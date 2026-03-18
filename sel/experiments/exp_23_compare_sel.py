#!/usr/bin/env python3
"""
SEL Experiment 23 — A/B/C Comparison
LOCATION: sel/experiments/exp_23_comparison.py

THE EXPERIMENT:
Three-way comparison of emotional language response quality:

  Condition A: Direct LLM (llama3.2:1b, no pipeline)
               Baseline — what does direct prompting produce?

  Condition B: SEL Pipeline (decompose → graph → qwen0.5b membrane)
               Full SEL with small membrane

  Condition C: SEL Pipeline (decompose → graph → llama3.2:1b membrane)
               Full SEL with better membrane
               Isolates: does membrane quality matter independently?

MEASUREMENTS:
  - Response text
  - Latency (ms)
  - Primitives extracted (B/C only)
  - Rule class fired (B/C only)
  - Fallback rate (B/C only)
  - Raw scores saved for external judge analysis

JUDGE:
  Results saved to exp_23_results.json
  Send to Claude for blind evaluation across 4 dimensions:
    1. Emotional resonance  (0-10)
    2. Accuracy             (0-10)
    3. Specificity          (0-10)
    4. Naturalness          (0-10)

20 PROMPTS across 4 emotional categories:
  Category 1: Simple loss (5)
  Category 2: Anticipatory emotion (5)
  Category 3: Complex relational (5)
  Category 4: Existential (5)

HYPOTHESIS:
  SEL matches or exceeds direct LLM on Categories 1-2
  SEL falls behind on Categories 3-4 (decomposer limits)
  SEL is significantly faster on all categories
  SEL is more interpretable on all categories
"""
import json
import time
import sys
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# ── add sel to path ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sel.core.decomposer import decompose
from sel.core.reasoner import reason
from sel.core.membrane import render as sel_render
from sel.core.template_renderer import render_or_fallback as template_render_or_fallback

# ── test prompts ──────────────────────────────────────────────────────────────

PROMPTS = {
    "simple_loss": [
        "I miss my hometown",
        "My dog passed away last week",
        "I lost touch with my best friend",
        "My old neighborhood is completely gone now",
        "I wish I had spent more time with my grandparents",
    ],
    "anticipatory": [
        "I have a big interview tomorrow",
        "I'm getting married next month",
        "I find out my test results today",
        "I'm about to move to a new city alone",
        "I'm waiting to hear if I got the job",
    ],
    "complex_relational": [
        "I feel proud of my kids but also sad they're growing up so fast",
        "I deeply admire someone I can never be like",
        "I'm genuinely happy for my friend but I feel jealous too",
        "I feel closer to someone after they hurt me",
        "I miss someone I know I'm better off without",
    ],
    "existential": [
        "I still think about the life I could have lived",
        "I wonder if I made the right choices",
        "I feel like time is moving too fast lately",
        "I'm proud of how far I've come but scared of what's next",
        "I feel most alive when I'm about to lose something",
    ],
}

MODELS = {
    "direct_llama":  "llama3.2:1b",
    "membrane_qwen": "qwen2.5:0.5b",
    "membrane_llama": "llama3.2:1b",
}

OLLAMA_HOST = "http://localhost:11434"

# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    experiment_id:    str
    prompt:           str
    category:         str
    condition:        str        # A, B, C, D
    condition_label:  str
    response:         str
    latency_ms:       float
    # SEL-specific (empty for condition A)
    primitives:       list
    concepts:         list
    rule_classes:     list
    fallback_fired:   bool
    # Condition D-specific (empty for A/B/C)
    template_key:     str
    template_variant: str
    # scoring (filled by judge later)
    emotional_resonance: float
    accuracy:            float
    specificity:         float
    naturalness:         float
    composite_score:     float
    judge_winner:        str
    judge_notes:         str

# ── direct LLM (condition A) ──────────────────────────────────────────────────

SYSTEM_DIRECT = """You are a warm, empathetic conversational partner.
When someone shares an emotional experience with you, respond with
genuine understanding and care. Keep your response to 2-3 sentences.
Do not ask questions. Do not give advice. Just acknowledge and resonate."""

def run_direct_llm(prompt: str, model: str) -> tuple[str, float]:
    """Run prompt directly through LLM with no pipeline."""
    start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  model,
                "prompt": prompt,
                "system": SYSTEM_DIRECT,
                "stream": False,
            },
            timeout=120,
        )
        response = resp.json().get("response", "").strip()
        latency  = (time.time() - start) * 1000
        return response, latency
    except Exception as e:
        return f"[ERROR: {e}]", (time.time() - start) * 1000

# ── SEL pipeline (conditions B and C) ────────────────────────────────────────

def run_sel_pipeline(prompt: str,
                     membrane_model: str) -> tuple[str, float, dict]:
    """
    Run prompt through full SEL pipeline.
    Temporarily swaps membrane model if needed.
    Returns response, latency, and diagnostic info.
    """
    import sel.core.membrane as membrane_module

    original_model = membrane_module.OLLAMA_MODEL
    membrane_module.OLLAMA_MODEL = membrane_model

    diagnostics = {
        "primitives":     [],
        "concepts":       [],
        "rule_classes":   [],
        "fallback_fired": False,
    }

    start = time.time()
    try:
        # run decomposer
        primitives = decompose(prompt)
        diagnostics["primitives"] = [
            f"{p.word}({p.layer},{p.weight:.2f})"
            for p in primitives
        ]

        # run reasoner
        concepts = reason(primitives)
        diagnostics["concepts"] = [
            f"{c.name}(rule={c.rule_class},conf={c.confidence:.2f})"
            for c in concepts
        ]
        diagnostics["rule_classes"] = list(set(
            c.rule_class for c in concepts
        ))
        diagnostics["fallback_fired"] = any(
            c.rule_class == "__fallback__" for c in concepts
        )

        # render directly — avoids re-running scope gate and re-decomposing
        response = sel_render(concepts, prompt)
        latency  = (time.time() - start) * 1000

    except Exception as e:
        response = f"[ERROR: {e}]"
        latency  = (time.time() - start) * 1000

    finally:
        membrane_module.OLLAMA_MODEL = original_model

    return response, latency, diagnostics

# ── SEL + zero-LLM template renderer (condition D) ───────────────────────────

def run_sel_template(prompt: str) -> tuple[str, float, dict]:
    """
    Run prompt through decompose → reason → template_renderer.
    Zero Ollama calls. Returns response, latency_ms, diagnostic dict.
    Diagnostic dict includes template_key and template_variant.
    """
    diagnostics = {
        "primitives":      [],
        "concepts":        [],
        "rule_classes":    [],
        "fallback_fired":  False,
        "template_key":    "",
        "template_variant":"",
    }

    start = time.time()
    try:
        primitives = decompose(prompt)
        diagnostics["primitives"] = [
            f"{p.word}({p.layer},{p.weight:.2f})"
            for p in primitives
        ]

        concepts = reason(primitives)
        diagnostics["concepts"] = [
            f"{c.name}(rule={c.rule_class},conf={c.confidence:.2f})"
            for c in concepts
        ]
        diagnostics["rule_classes"] = list(set(
            c.rule_class for c in concepts
        ))
        diagnostics["fallback_fired"] = any(
            c.rule_class == "__fallback__" for c in concepts
        )

        response, template_key, template_variant = template_render_or_fallback(
            concepts, prompt
        )
        diagnostics["template_key"]     = template_key
        diagnostics["template_variant"] = template_variant
        latency = (time.time() - start) * 1000

    except Exception as e:
        response = f"[ERROR: {e}]"
        latency  = (time.time() - start) * 1000

    return response, latency, diagnostics


# ── run experiment ────────────────────────────────────────────────────────────

def run_experiment() -> list[TrialResult]:
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results: list[TrialResult] = []

    total_prompts = sum(len(v) for v in PROMPTS.values())
    total_trials  = total_prompts * 4  # A, B, C, D

    print(f"SEL Experiment 23 — A/B/C/D Comparison")
    print(f"Experiment ID: {experiment_id}")
    print(f"{total_prompts} prompts × 4 conditions = {total_trials} trials")
    print(f"\nCondition A: Direct llama3.2:1b")
    print(f"Condition B: SEL + qwen0.5b membrane")
    print(f"Condition C: SEL + llama3.2:1b membrane")
    print(f"Condition D: SEL + zero-LLM template renderer")
    print(f"\nResults will be saved for external judge evaluation.")
    print(f"{'═'*60}\n")

    for category, prompts in PROMPTS.items():
        print(f"\n── CATEGORY: {category.upper()} ──────────────────────────")

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n  [{prompt_idx+1}/{len(prompts)}] {prompt}")
            print(f"  {'─'*50}")

            # ── Condition A: Direct LLM ───────────────────────────────────
            print(f"  A) Direct llama3.2:1b...", end=" ", flush=True)
            response_a, latency_a = run_direct_llm(
                prompt, MODELS["direct_llama"]
            )
            print(f"{latency_a:.0f}ms")
            print(f"     → {response_a[:80]}...")

            result_a = TrialResult(
                experiment_id=experiment_id,
                prompt=prompt,
                category=category,
                condition="A",
                condition_label="Direct llama3.2:1b",
                response=response_a,
                latency_ms=round(latency_a, 1),
                primitives=[],
                concepts=[],
                rule_classes=[],
                fallback_fired=False,
                template_key="",
                template_variant="",
                emotional_resonance=0.0,
                accuracy=0.0,
                specificity=0.0,
                naturalness=0.0,
                composite_score=0.0,
                judge_winner="",
                judge_notes="",
            )
            all_results.append(result_a)

            # ── Condition B: SEL + qwen membrane ─────────────────────────
            print(f"  B) SEL + qwen0.5b...", end=" ", flush=True)
            response_b, latency_b, diag_b = run_sel_pipeline(
                prompt, MODELS["membrane_qwen"]
            )
            fallback_b = "FALLBACK" if diag_b["fallback_fired"] else \
                         f"Rule {diag_b['rule_classes']}"
            print(f"{latency_b:.0f}ms  [{fallback_b}]")
            print(f"     primitives: {', '.join(diag_b['primitives'])}")
            print(f"     concepts:   {', '.join(diag_b['concepts'])}")
            print(f"     → {response_b[:80]}...")

            result_b = TrialResult(
                experiment_id=experiment_id,
                prompt=prompt,
                category=category,
                condition="B",
                condition_label="SEL + qwen0.5b membrane",
                response=response_b,
                latency_ms=round(latency_b, 1),
                primitives=diag_b["primitives"],
                concepts=diag_b["concepts"],
                rule_classes=diag_b["rule_classes"],
                fallback_fired=diag_b["fallback_fired"],
                template_key="",
                template_variant="",
                emotional_resonance=0.0,
                accuracy=0.0,
                specificity=0.0,
                naturalness=0.0,
                composite_score=0.0,
                judge_winner="",
                judge_notes="",
            )
            all_results.append(result_b)

            # ── Condition C: SEL + llama membrane ────────────────────────
            print(f"  C) SEL + llama3.2:1b...", end=" ", flush=True)
            response_c, latency_c, diag_c = run_sel_pipeline(
                prompt, MODELS["membrane_llama"]
            )
            fallback_c = "FALLBACK" if diag_c["fallback_fired"] else \
                         f"Rule {diag_c['rule_classes']}"
            print(f"{latency_c:.0f}ms  [{fallback_c}]")
            print(f"     → {response_c[:80]}...")

            result_c = TrialResult(
                experiment_id=experiment_id,
                prompt=prompt,
                category=category,
                condition="C",
                condition_label="SEL + llama3.2:1b membrane",
                response=response_c,
                latency_ms=round(latency_c, 1),
                primitives=diag_c["primitives"],
                concepts=diag_c["concepts"],
                rule_classes=diag_c["rule_classes"],
                fallback_fired=diag_c["fallback_fired"],
                template_key="",
                template_variant="",
                emotional_resonance=0.0,
                accuracy=0.0,
                specificity=0.0,
                naturalness=0.0,
                composite_score=0.0,
                judge_winner="",
                judge_notes="",
            )
            all_results.append(result_c)

            # ── Condition D: SEL + zero-LLM template renderer ─────────────
            print(f"  D) SEL + template (zero LLM)...", end=" ", flush=True)
            response_d, latency_d, diag_d = run_sel_template(prompt)
            tmpl_d = (f"{diag_d['template_key']}:{diag_d['template_variant']}"
                      if diag_d["template_key"] else "FALLBACK")
            print(f"{latency_d:.0f}ms  [{tmpl_d}]")
            print(f"     → {response_d[:80]}...")

            result_d = TrialResult(
                experiment_id=experiment_id,
                prompt=prompt,
                category=category,
                condition="D",
                condition_label="SEL + zero-LLM template",
                response=response_d,
                latency_ms=round(latency_d, 1),
                primitives=diag_d["primitives"],
                concepts=diag_d["concepts"],
                rule_classes=diag_d["rule_classes"],
                fallback_fired=diag_d["fallback_fired"],
                template_key=diag_d["template_key"],
                template_variant=diag_d["template_variant"],
                emotional_resonance=0.0,
                accuracy=0.0,
                specificity=0.0,
                naturalness=0.0,
                composite_score=0.0,
                judge_winner="",
                judge_notes="",
            )
            all_results.append(result_d)

    # ── summary statistics ────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"EXPERIMENT COMPLETE — LATENCY SUMMARY")
    print(f"{'═'*60}\n")

    for condition in ["A", "B", "C", "D"]:
        condition_results = [r for r in all_results
                            if r.condition == condition]
        latencies = [r.latency_ms for r in condition_results]
        label = condition_results[0].condition_label
        avg   = sum(latencies) / len(latencies)
        mn    = min(latencies)
        mx    = max(latencies)
        print(f"  Condition {condition} ({label})")
        print(f"    avg: {avg:.0f}ms  min: {mn:.0f}ms  max: {mx:.0f}ms")

    print(f"\n── TEMPLATE HIT RATE (Condition D) ───────────────────────────")
    d_results = [r for r in all_results if r.condition == "D"]
    template_hits = sum(1 for r in d_results if r.template_key not in ("", "__fallback__"))
    print(f"  Template hits: {template_hits}/{len(d_results)} ({template_hits/len(d_results):.0%})")
    print(f"\n── TEMPLATE KEYS USED (Condition D) ──────────────────────────")
    tkey_counts: dict[str, int] = {}
    for r in d_results:
        k = f"{r.template_key}:{r.template_variant}" if r.template_key else "__fallback__"
        tkey_counts[k] = tkey_counts.get(k, 0) + 1
    for k, count in sorted(tkey_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<40} {count}")

    print(f"\n── FALLBACK RATE ─────────────────────────────────────────────")
    for condition in ["B", "C", "D"]:
        cond_results = [r for r in all_results
                       if r.condition == condition]
        fallbacks = sum(1 for r in cond_results if r.fallback_fired)
        total     = len(cond_results)
        print(f"  Condition {condition}: {fallbacks}/{total} fallbacks "
              f"({fallbacks/total:.0%})")

    print(f"\n── FALLBACK BY CATEGORY ──────────────────────────────────────")
    for category in PROMPTS:
        cat_results = [r for r in all_results
                      if r.category == category
                      and r.condition == "B"]
        fallbacks = sum(1 for r in cat_results if r.fallback_fired)
        total     = len(cat_results)
        print(f"  {category:<25} {fallbacks}/{total} fallbacks")

    print(f"\n── RULE CLASSES FIRED ────────────────────────────────────────")
    all_rule_classes: dict[str, int] = {}
    for r in all_results:
        if r.condition == "B":
            for rc in r.rule_classes:
                all_rule_classes[rc] = all_rule_classes.get(rc, 0) + 1
    for rc, count in sorted(all_rule_classes.items(),
                             key=lambda x: x[1], reverse=True):
        print(f"  Rule {rc:<15} fired {count} times")

    return all_results

# ── save results ──────────────────────────────────────────────────────────────

def save_results(results: list[TrialResult],
                 base="sel/experiments/results_exp_23"):
    Path("sel/experiments").mkdir(parents=True, exist_ok=True)

    # full results for judge
    json_path = Path(f"{base}_full.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # judge-ready format — responses only, no condition labels
    # this is what you send to Claude for blind evaluation
    judge_path = Path(f"{base}_for_judge.json")
    judge_data = []
    prompts_seen = []

    for r in results:
        if r.prompt not in prompts_seen:
            prompts_seen.append(r.prompt)
            # get all three conditions for this prompt
            prompt_results = [x for x in results
                             if x.prompt == r.prompt]
            judge_data.append({
                "prompt":   r.prompt,
                "category": r.category,
                "responses": {
                    # label order is fixed A→D to allow D comparison focus
                    "response_1": next(x.response for x in prompt_results
                                      if x.condition == "A"),
                    "response_2": next(x.response for x in prompt_results
                                      if x.condition == "B"),
                    "response_3": next(x.response for x in prompt_results
                                      if x.condition == "C"),
                    "response_4": next(x.response for x in prompt_results
                                      if x.condition == "D"),
                },
                # hidden from judge — revealed after scoring
                "condition_map": {
                    "response_1": "A — Direct llama3.2:1b",
                    "response_2": "B — SEL + qwen0.5b",
                    "response_3": "C — SEL + llama3.2:1b",
                    "response_4": "D — SEL + zero-LLM template",
                }
            })

    with open(judge_path, "w") as f:
        json.dump(judge_data, f, indent=2)

    # latency summary CSV
    import csv
    csv_path = Path(f"{base}_latency.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "category", "condition",
                        "latency_ms", "fallback", "rule_classes"])
        for r in results:
            writer.writerow([
                r.prompt, r.category, r.condition,
                r.latency_ms, r.fallback_fired,
                "|".join(r.rule_classes)
            ])

    print(f"\n── SAVED ─────────────────────────────────────────────────────")
    print(f"  {json_path}     ← full results")
    print(f"  {judge_path}    ← send THIS to Claude for blind judging")
    print(f"  {csv_path}      ← latency data")
    print(f"\n  NEXT STEP:")
    print(f"  Paste the contents of {judge_path} into Claude")
    print(f"  and ask for blind evaluation of each response set.")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_experiment()
    save_results(results)