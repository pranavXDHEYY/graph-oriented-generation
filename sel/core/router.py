"""
router.py — Orchestrates the full SEL pipeline.
Implements process(prompt) -> str.

Flow:
  1. Validate prompt is emotional/relational scope
  2. decompose(prompt) -> list[Primitive]
  3. reason(primitives) -> list[Concept]
  4. render(concepts, prompt) -> str

Observability:
  Every stage is timed with perf_counter and logged as a formatted trace block.
  process_debug() returns full timing data alongside intermediate results.

Example log output:
  [SEL] ══════════════════════════════════════════════════════
  [SEL] ▶  PIPELINE  "I miss my hometown"
  [SEL] ├─ scope_check    0.04 ms   IN SCOPE
  [SEL] ├─ decompose    523.11 ms   ollama:qwen2.5:0.5b  →  3 primitives
  [SEL] │              GRIEF(0b,0.85)  PLACE(0a,0.70)  NOSTALGIA(0b,0.90)
  [SEL] ├─ reason         1.38 ms   pure Python          →  2 concepts
  [SEL] │              homesickness(K,conf=0.72)  longing(A,conf=0.85)
  [SEL] ├─ membrane     876.44 ms   ollama:qwen2.5:0.5b  →  152 chars
  [SEL] └─ TOTAL       1400.97 ms
  [SEL] ══════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .decomposer import decompose, Primitive
from .reasoner import reason, Concept
from .membrane import render as membrane_render
from .template_renderer import render as template_render

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[SEL] %(message)s"
)
log = logging.getLogger("sel.router")

# ── Scope validation ───────────────────────────────────────────────────────
_EMOTIONAL_SIGNALS = re.compile(
    r'\b(miss|feel|felt|feeling|love|hate|afraid|scared|nervous|anxious|'
    r'happy|sad|angry|proud|ashamed|guilty|lonely|excited|worried|hope|'
    r'grief|loss|joy|fear|pain|hurt|tired|empty|full|grateful|resentful|'
    r'envy|jealous|nostalgic|homesick|numb|overwhelmed|moved|touched|'
    r'heartbroken|devastated|thrilled|relieved|confused|lost|belong|'
    r'still think|can\'t stop|can\'t get over|something about|'
    r'it\'s been|used to|remember when|long for|wish|if only|'
    r'something good|about to happen|about to|can\'t wait|'
    r'looking forward|missing|longing|aching|carrying|'
    # third-person loss (covers "passed away", "is gone", "completely gone")
    r'passed away|passed on|gone now|no longer here|no longer with|'
    r'lost my|losing my|losing her|losing him|losing them|'
    # anticipatory / life-event signals
    r'interview|waiting to hear|find out|test results|getting married|'
    r'move to|moving to|new city|new job|starting over|big day|'
    r'waiting for|about to find)\b',
    re.IGNORECASE
)

_OUT_OF_SCOPE_SIGNALS = re.compile(
    r'\b(write|generate|create|build|code|function|class|script|'
    r'capital of|population of|calculate|compute|convert|translate|'
    r'what is the|define|explain how|how do I|tutorial|recipe|'
    r'weather|news|schedule|price|stock|sports|election)\b',
    re.IGNORECASE
)

_OUT_OF_SCOPE_RESPONSE = (
    "The SEL is designed for emotional and relational language only — things like "
    "how you feel, who you miss, what you're carrying. For factual questions, "
    "code generation, or practical tasks, a general-purpose assistant would serve "
    "you better."
)


# ── Timing / trace ─────────────────────────────────────────────────────────
@dataclass
class StageResult:
    name:       str
    backend:    str          # "pure Python" | "ollama:model" | "regex"
    duration_ms: float
    output:     Any          # raw output for this stage
    error:      str = ""


@dataclass
class PipelineTrace:
    prompt:   str
    stages:   list[StageResult] = field(default_factory=list)
    _start:   float = field(default_factory=time.perf_counter, repr=False)

    def record(self, name: str, backend: str, start: float, output: Any,
               error: str = "") -> StageResult:
        ms = (time.perf_counter() - start) * 1000
        s = StageResult(name=name, backend=backend, duration_ms=ms,
                        output=output, error=error)
        self.stages.append(s)
        return s

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def log(self) -> None:
        """Emit a formatted trace block to the logger."""
        W = 56
        log.info("═" * W)
        log.info(f'▶  PIPELINE  "{self.prompt}"')
        for i, s in enumerate(self.stages):
            is_last = (i == len(self.stages) - 1)
            prefix = "└─" if is_last else "├─"
            status = f"ERROR: {s.error}" if s.error else _stage_summary(s)
            log.info(f"{prefix} {s.name:<14} {s.duration_ms:>8.2f} ms   {s.backend:<22}  →  {status}")
            detail = _stage_detail(s)
            if detail:
                cont = "  " if is_last else "│ "
                log.info(f"{cont}             {detail}")
        log.info(f"└─ {'TOTAL':<14} {self.total_ms:>8.2f} ms")
        log.info("═" * W)

    def to_dict(self) -> dict:
        return {
            "prompt":    self.prompt,
            "total_ms":  round(self.total_ms, 3),
            "stages": [
                {
                    "name":        s.name,
                    "backend":     s.backend,
                    "duration_ms": round(s.duration_ms, 3),
                    "error":       s.error,
                }
                for s in self.stages
            ],
        }


def _stage_summary(s: StageResult) -> str:
    if s.name == "scope_check":
        return "IN SCOPE" if s.output else "OUT OF SCOPE"
    if s.name == "decompose":
        n = len(s.output) if s.output else 0
        return f"{n} primitive{'s' if n != 1 else ''}"
    if s.name == "reason":
        n = len(s.output) if s.output else 0
        return f"{n} concept{'s' if n != 1 else ''}"
    if s.name == "membrane":
        n = len(s.output) if s.output else 0
        return f"{n} chars"
    return str(s.output)[:60]


def _stage_detail(s: StageResult) -> str:
    if s.name == "decompose" and s.output:
        parts = [f"{p.word}({p.layer},{p.weight:.2f})" for p in s.output]
        return "  ".join(parts)
    if s.name == "reason" and s.output:
        parts = [f"{c.name}({c.rule_class},conf={c.confidence:.2f})" for c in s.output]
        return "  ".join(parts)
    return ""


# ── Public API ─────────────────────────────────────────────────────────────
def process(prompt: str) -> str:
    """
    Full pipeline: prompt -> empathetic English response.
    Logs a timed trace block for every stage.
    """
    prompt = prompt.strip()
    if not prompt:
        return "I'm here — what's on your mind?"

    trace = PipelineTrace(prompt=prompt)

    # Stage 0: Scope check
    t = time.perf_counter()
    in_scope = _is_emotional_scope(prompt)
    trace.record("scope_check", "regex", t, in_scope)

    if not in_scope:
        trace.log()
        return _OUT_OF_SCOPE_RESPONSE

    # Stage 1: Decompose
    t = time.perf_counter()
    try:
        primitives = decompose(prompt)
        trace.record("decompose", "ollama:qwen2.5:0.5b", t, primitives)
    except Exception as e:
        trace.record("decompose", "ollama:qwen2.5:0.5b", t, [], error=str(e))
        primitives = []

    # Stage 2: Reason
    t = time.perf_counter()
    try:
        concepts = reason(primitives) if primitives else []
        trace.record("reason", "pure Python", t, concepts)
    except Exception as e:
        trace.record("reason", "pure Python", t, [], error=str(e))
        concepts = []

    # Stage 3: Template renderer (zero LLM) — falls back to membrane when needed
    t = time.perf_counter()
    try:
        response = template_render(concepts, prompt)
        if response is not None:
            trace.record("render", "zero-LLM:template", t, response)
        else:
            # Fallback: concept not in library, zero primitives, or confidence < 0.4
            t_m = time.perf_counter()
            response = membrane_render(concepts, prompt)
            trace.record("render", "ollama:qwen2.5:0.5b", t_m, response)
    except Exception as e:
        trace.record("render", "ollama:qwen2.5:0.5b", t, "", error=str(e))
        response = "Something in what you said reached me. I just want you to know that."

    trace.log()
    return response


def process_debug(prompt: str) -> dict:
    """
    Same as process() but returns a dict with all intermediate results AND timing.

    Return shape:
    {
      "prompt":      str,
      "in_scope":    bool,
      "primitives":  list[dict],
      "concepts":    list[dict],
      "response":    str,
      "timing": {
        "total_ms":  float,
        "stages": [
          {"name": "scope_check", "backend": "regex",               "duration_ms": 0.04},
          {"name": "decompose",   "backend": "ollama:qwen2.5:0.5b", "duration_ms": 523.11},
          {"name": "reason",      "backend": "pure Python",         "duration_ms": 1.38},
          {"name": "membrane",    "backend": "ollama:qwen2.5:0.5b", "duration_ms": 876.44},
        ]
      }
    }
    """
    prompt = prompt.strip()
    trace = PipelineTrace(prompt=prompt)

    result: dict = {
        "prompt":     prompt,
        "in_scope":   False,
        "primitives": [],
        "concepts":   [],
        "response":   "",
        "timing":     {},
    }

    # Stage 0: Scope check
    t = time.perf_counter()
    in_scope = _is_emotional_scope(prompt)
    trace.record("scope_check", "regex", t, in_scope)
    result["in_scope"] = in_scope

    if not in_scope:
        result["response"] = _OUT_OF_SCOPE_RESPONSE
        result["timing"] = trace.to_dict()
        trace.log()
        return result

    # Stage 1: Decompose
    t = time.perf_counter()
    try:
        primitives = decompose(prompt)
        trace.record("decompose", "ollama:qwen2.5:0.5b", t, primitives)
        result["primitives"] = [
            {"word": p.word, "layer": p.layer, "weight": p.weight}
            for p in primitives
        ]
    except Exception as e:
        trace.record("decompose", "ollama:qwen2.5:0.5b", t, [], error=str(e))
        primitives = []

    # Stage 2: Reason
    t = time.perf_counter()
    try:
        concepts = reason(primitives) if primitives else []
        trace.record("reason", "pure Python", t, concepts)
        result["concepts"] = [
            {
                "name":              c.name,
                "rule_class":        c.rule_class,
                "confidence":        c.confidence,
                "source_primitives": c.source_primitives,
                "output_type":       c.output_type,
                "validated":         c.validated,
            }
            for c in concepts
        ]
    except Exception as e:
        trace.record("reason", "pure Python", t, [], error=str(e))
        concepts = []

    # Stage 3: Template renderer (zero LLM) — falls back to membrane when needed
    t = time.perf_counter()
    try:
        response = template_render(concepts, prompt)
        if response is not None:
            trace.record("render", "zero-LLM:template", t, response)
        else:
            t_m = time.perf_counter()
            response = membrane_render(concepts, prompt)
            trace.record("render", "ollama:qwen2.5:0.5b", t_m, response)
        result["response"] = response
    except Exception as e:
        trace.record("render", "ollama:qwen2.5:0.5b", t, "", error=str(e))
        result["response"] = "Something reached me in what you said."

    result["timing"] = trace.to_dict()
    trace.log()
    return result


def _is_emotional_scope(prompt: str) -> bool:
    if _OUT_OF_SCOPE_SIGNALS.search(prompt):
        if not _EMOTIONAL_SIGNALS.search(prompt):
            return False
    if _EMOTIONAL_SIGNALS.search(prompt):
        return True
    words = prompt.split()
    if len(words) <= 12 and prompt.strip().endswith((".", "...", "?", "!")):
        return True
    if re.match(r'^(i|i\'m|i\'ve|i was|i feel|i think|i miss|i wish)', prompt, re.IGNORECASE):
        return True
    return False
