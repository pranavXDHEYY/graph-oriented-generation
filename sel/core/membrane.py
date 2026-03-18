"""
membrane.py — Stage 3 of the SEL pipeline.
Renders Layer 1 concepts into natural empathetic English using
qwen2.5:0.5b via Ollama.

Role: SEMANTIC RENDERER ONLY — not reasoning, not explaining, not defining.
The output must feel human. Warm. Present.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b"

_SYSTEM_PROMPT = """You are a warm human voice. Speak like a thoughtful friend, not a therapist.

Input: emotional concepts + a person's message.
Task: write 2-3 sentences that resonate with what they feel.

HARD RULES — never break these:
NEVER start with "You are experiencing", "I understand that you", "It sounds like you", or "You feel".
NEVER name the emotion as a label: not "homesickness", not "grief", not "displacement".
NEVER explain or define. Never advise unless asked.
ALWAYS speak in warm, present, natural English — like someone who quietly gets it.

BAD (forbidden): "You are experiencing homesickness and longing."
BAD (forbidden): "I understand that you miss your hometown."
GOOD: "That pull toward a place that shaped you — it never quite leaves you."
GOOD: "There's something about being away from where you belong that makes the distance feel bigger than miles."

Write your response now:"""


@dataclass
class Concept:
    name: str
    rule_class: str
    confidence: float
    source_primitives: list = None  # type: ignore[assignment]
    output_type: str = ""
    validated: bool = False


def render(concepts: list[Concept], context: str) -> str:
    """
    Input:  [exile, homesickness], "I miss my hometown"
    Output: "That longing for a place that shaped you —
             it's one of the most human feelings there is."
    """
    if not concepts:
        return _render_no_concepts(context)

    concept_summary = _summarize_concepts(concepts)
    user_message = _build_user_message(concept_summary, context)
    raw = _call_ollama(user_message)

    response = raw.strip()
    if not response:
        return _fallback_render(concepts, context)

    return response


def _summarize_concepts(concepts: list[Concept]) -> str:
    """Build a readable concept list for the model."""
    lines = []
    for c in concepts:
        validated_marker = " (confirmed)" if c.validated else " (inferred)"
        lines.append(f"- {c.name}{validated_marker}: {c.output_type}")
    return "\n".join(lines)


def _build_user_message(concept_summary: str, original_prompt: str) -> str:
    return (
        f"Emotional concepts detected:\n{concept_summary}\n\n"
        f"Person's original message: \"{original_prompt}\"\n\n"
        f"Your response:"
    )


def _call_ollama(user_message: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{_SYSTEM_PROMPT}\n\n{user_message}",
        "stream": False,
        "options": {
            "temperature": 0.75,
            "num_predict": 200,
            "top_p": 0.9,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=45)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception:
        return ""


def _render_no_concepts(context: str) -> str:
    """Render when no concepts were extracted — minimal graceful response."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Emotional concepts detected: (none clearly identified)\n\n"
            f"Person's original message: \"{context}\"\n\n"
            f"Your response:"
        ),
        "stream": False,
        "options": {"temperature": 0.75, "num_predict": 150},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=45)
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        if result:
            return result
    except Exception:
        pass
    return "That sounds like something that's sitting with you. Whatever it is, it sounds real."


def _fallback_render(concepts: list[Concept], context: str) -> str:
    """Static fallback when model returns empty."""
    top_concept = concepts[0].name if concepts else "that feeling"
    templates = {
        "yearning":       "That reaching toward something that's no longer within reach — there's a particular ache to that kind of wanting.",
        "anticipation":   "Something about what's coming has you lit up inside. That kind of feeling is worth sitting with.",
        "mourning":       "There's a weight to carrying something for a long time. It doesn't mean you're stuck — it means it mattered.",
        "recognition":    "Seeing something truly remarkable in another person is its own quiet gift.",
        "momentum":       "You're in motion and it feels right. That kind of aliveness is rare — let it carry you.",
        "understanding":  "Something finally clicked into place, didn't it. That moment of clarity is its own reward.",
        "homesickness":   "That pull toward a place that shaped you — it never quite leaves you, does it.",
        "exile":          "There's something about being away from where you belong that makes the distance feel bigger than miles.",
        "nostalgia":      "The past has a way of staying present in you, especially the parts that meant something.",
        "anxiety":        "That tension between wanting something and being afraid of it — it's exhausting to carry both at once.",
        "hope":           "That open feeling of possibility, held carefully — it's fragile but it's real.",
        "worry":          "Your mind is working overtime on this. That's not weakness — it's care.",
        "joy":            "That lightness in you right now — don't rush past it.",
        "compassion":     "Feeling someone else's weight as if it were your own is a rare kind of love.",
    }
    return templates.get(top_concept, "Whatever you're carrying right now — it sounds like it matters.")
