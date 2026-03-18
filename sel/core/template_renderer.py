"""
template_renderer.py — Zero-LLM stage 3 renderer for the SEL pipeline.

Replaces membrane.py as the default renderer.
Input:  list[Concept] (from reasoner) + original prompt (for context signals)
Output: selected template string, or None if no template applies

Logic:
  1. Take highest-confidence concept from reasoner output
  2. Map concept name → template library key
  3. Detect context signals from prompt to select best variant
  4. Pick deterministically from variant's response list

Returns None (→ router falls back to membrane.py) when:
  - No concepts provided (zero-primitive case)
  - Top concept not in template library
  - Top concept confidence < 0.4
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


# ── Template loading ───────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sel/
_TEMPLATE_PATH = os.path.join(_HERE, "templates.json")

_templates: dict = {}


def _load() -> None:
    global _templates
    if not _templates:
        with open(_TEMPLATE_PATH, "r") as f:
            data = json.load(f)
        _templates = data["templates"]


# ── Concept dataclass (mirrors reasoner.Concept) ──────────────────────────

@dataclass
class Concept:
    name: str
    rule_class: str
    confidence: float
    source_primitives: list[str] = field(default_factory=list)
    output_type: str = ""
    validated: bool = False


# ── Concept name → template key ───────────────────────────────────────────
# Reasoner produces graph concept names; templates.json uses its own keys.

_CONCEPT_TO_TEMPLATE: dict[str, str] = {
    "homesickness":          "homesickness",
    "exile":                 "homesickness",
    "nostalgia":             "nostalgia",
    "reminiscence":          "nostalgia",
    "longing":               "longing",
    "yearning":              "longing",
    "mourning":              "mourning",
    "grief":                 "mourning",
    "anticipation":          "anticipation",
    "admiration":            "recognition",
    "recognition":           "recognition",
    "momentum":              "momentum",
    "understanding":         "understanding",
    "insight":               "understanding",
    "apprehension":          "anxiety",
    "anxiety":               "anxiety",
    "dread_of_possibility":  "worry",
    "worry":                 "worry",
    "joy":                   "anticipation",   # joy_attractor routes here; wedding/nervous variants apply
    "hope":                  "hope",
    "envy":                  "envy",
    "displacement":          "displacement",
    "transcendence":         "transcendence",
    "awe":                   "transcendence",
    "contentment":           "contentment",
    "celebration":           "celebration",
    "social_pain":           "social_pain",
    "compassion":            "compassion",
    "beauty_experience":     "beauty_experience",
    "beauty":                "beauty_experience",
    "disorientation":        "disorientation",
    "shame":                 "shame",
    "guilt":                 "shame",
    "bitter_grief":          "bitter_grief",
    "mortality":             "mortality",
    "existential_dread":     "mortality",
}


# ── Variant selection signals ──────────────────────────────────────────────
# For each template key: ordered list of (variant_name, compiled pattern).
# First matching pattern wins. Falls through to "default".

_VARIANT_SIGNALS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "homesickness": [
        ("pet_death",    re.compile(
            r'\b(passed away|passed on|died|dead|death|dog|cat|pet|bird|hamster|fish)\b', re.I)),
        ("gone_place",   re.compile(
            r'\b(gone now|no longer|completely gone|disappeared|torn down|demolished)\b', re.I)),
        ("neighborhood", re.compile(
            r'\b(neighborhood|neighbourhood|street|block|district)\b', re.I)),
        ("hometown",     re.compile(
            r'\b(hometown|home town|where I grew|where I was born|grew up)\b', re.I)),
    ],
    "nostalgia": [
        ("people",       re.compile(
            r'\b(friend|friends|family|grandparent|someone|person|people|they)\b', re.I)),
        ("time_period",  re.compile(
            r'\b(used to|childhood|years ago|back then|once|when I was)\b', re.I)),
    ],
    "longing": [
        ("grandparents", re.compile(
            r'grandparent|grandmother|grandfather|\bgran\b|grandma|grandpa', re.I)),
        ("past_self",    re.compile(
            r'\b(could have|might have|would have|what if|life I could|could\'ve)\b', re.I)),
        ("person",       re.compile(
            r'\b(friend|someone|person|partner|they|him|her)\b', re.I)),
    ],
    "mourning": [
        ("life_unlived", re.compile(
            r'\b(could have lived|could have been|life I could|what could have)\b', re.I)),
        ("extended",     re.compile(
            r'\b(still|years|long time|always|every day|keeps coming back)\b', re.I)),
    ],
    "anticipation": [
        ("wedding",              re.compile(
            r'\b(married|wedding|ceremony|big day|getting married|get married)\b', re.I)),
        ("job_news",             re.compile(
            r'\b(got the job|job offer|position|waiting to hear|hear if I got)\b', re.I)),
        ("nervous_anticipation", re.compile(
            r'\b(nervous|scared|anxious|worried|interview|test results|results)\b', re.I)),
    ],
    "recognition": [
        ("children",    re.compile(
            r'\b(kids|children|child|son|daughter|my kids|my children)\b', re.I)),
        ("unreachable", re.compile(
            r'\b(never be|can never|never be like|impossible|can\'t be like)\b', re.I)),
    ],
    "momentum": [
        ("mixed_fear", re.compile(
            r'\b(nervous|scared|afraid|terrified|worried|but also|and also)\b', re.I)),
        ("new_start",  re.compile(
            r'\b(new city|new job|starting over|new chapter|moving|move)\b', re.I)),
    ],
    "understanding": [
        ("regret_understanding", re.compile(
            r'\b(choices|decisions|right choice|wonder if|should have|regret)\b', re.I)),
        ("self_understanding",   re.compile(
            r'\b(myself|who I am|who I\'ve|i\'ve become|my own)\b', re.I)),
    ],
    "anxiety": [
        ("choices", re.compile(
            r'\b(choices|wonder if|right choices|decisions|should have|made the)\b', re.I)),
        ("waiting", re.compile(
            r'\b(waiting|wait|hear|find out|results|interview|news)\b', re.I)),
    ],
    "hope": [
        ("future", re.compile(
            r'\b(future|next|will|going to|someday|one day|ahead)\b', re.I)),
    ],
    "envy": [
        ("friend_success", re.compile(
            r'\b(friend|friends|they|their|someone I know|someone else)\b', re.I)),
    ],
    "displacement": [
        ("ambivalent", re.compile(
            r'\b(better off|know I\'m|without|had to|right decision|should)\b', re.I)),
    ],
    "transcendence": [
        ("mortality", re.compile(
            r'\b(lose|losing|about to lose|end|ends|finite|death|die|alive when)\b', re.I)),
    ],
    "celebration": [
        ("milestone", re.compile(
            r'\b(how far|come so far|how much|proud of|accomplished|made it)\b', re.I)),
    ],
}

# Static fallbacks for concepts not in template library (keeps Condition D zero-LLM)
_STATIC_FALLBACKS: dict[str, str] = {
    "joy":              "That lightness you're feeling right now — let yourself stay in it.",
    "grief":            "Whatever you're carrying, it's real. That's worth acknowledging.",
    "fear":             "Fear about something that matters — that's your system taking it seriously.",
    "sadness":          "Some things deserve to be felt fully before anything else.",
    "negated_emotion":  "The absence of a feeling you expected — that's its own kind of experience.",
    "menace":           "That unease has a weight to it. You're right to notice it.",
    "territorial_fear": "Feeling uncertain about where you stand — that discomfort is real.",
    "pride":            "What you've built and who you've become — those are real things.",
    "someone":          "The people who matter to us stay with us in ways that don't need explaining.",
    "know":             "Sitting with something you can't quite name yet — that's its own kind of wisdom.",
    "think":            "That persistent thought — the fact that it keeps returning means something.",
    "die":              "Being close to something that reminds you what's real — that lands differently.",
    "live":             "The feeling of being alive in a particular moment — some of them deserve to be noticed.",
    "move":             "Movement and change have their own particular weight, even when they're chosen.",
    "anxiety":          "The tension of not knowing yet — it doesn't need to be resolved to be real.",
    "apprehension":     "That careful feeling before something uncertain — it's just honesty.",
}
_DEFAULT_FALLBACK = "Whatever you're carrying right now — it sounds like it matters."


# ── Public API ─────────────────────────────────────────────────────────────

def render(concepts: list[Concept], context: str) -> str | None:
    """
    Returns a template string, or None if no template applies.

    None signals the router to fall back to membrane.py.
    """
    _load()

    if not concepts:
        return None

    top = concepts[0]

    if top.confidence < 0.4:
        return None

    template_key = _CONCEPT_TO_TEMPLATE.get(top.name)
    if template_key is None or template_key not in _templates:
        return None

    # Context remap: homesickness when person signals dominate over place signals
    # routes to displacement (which has the ambivalent/person-oriented variants)
    if template_key == "homesickness":
        has_person = re.search(
            r'\b(someone|person|friend|they|him|her|partner|people)\b', context, re.I)
        has_place  = re.search(
            r'\b(home|hometown|town|city|neighborhood|place|here|back|there)\b', context, re.I)
        if has_person and not has_place:
            template_key = "displacement"

    variant_key   = _select_variant(template_key, context)
    variant_list  = (
        _templates[template_key]["variants"].get(variant_key)
        or _templates[template_key]["variants"]["default"]
    )

    return _pick(variant_list, context)


def render_or_fallback(concepts: list[Concept], context: str) -> tuple[str, str, str]:
    """
    Zero-LLM rendering — always returns a string.

    Returns (response, template_key, variant_key).
    template_key is "__fallback__" when no template matched.
    """
    _load()

    result = render(concepts, context)
    if result is not None:
        top          = concepts[0]
        template_key = _CONCEPT_TO_TEMPLATE.get(top.name, top.name)
        variant_key  = _select_variant(template_key, context)
        return result, template_key, variant_key

    # Static fallback — zero LLM calls, always succeeds
    if concepts:
        top      = concepts[0]
        response = _STATIC_FALLBACKS.get(top.name, _DEFAULT_FALLBACK)
    else:
        response = _DEFAULT_FALLBACK

    return response, "__fallback__", "__static__"


# ── Internals ──────────────────────────────────────────────────────────────

def _select_variant(template_key: str, prompt: str) -> str:
    """Return the best variant name for this concept given prompt text."""
    for variant_name, pattern in _VARIANT_SIGNALS.get(template_key, []):
        if pattern.search(prompt):
            return variant_name
    return "default"


def _pick(responses: list[str], seed_text: str) -> str:
    """
    Deterministically select from a response list.
    Uses sum-of-ordinals (stable across Python processes unlike hash()).
    """
    idx = sum(ord(c) for c in seed_text) % len(responses)
    return responses[idx]
