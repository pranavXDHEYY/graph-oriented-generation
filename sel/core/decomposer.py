"""
decomposer.py — Stage 1 of the SEL pipeline.
Rule-based parser — zero Ollama calls, <10ms per prompt.

Three layers applied in order, merged by highest weight:

  Layer 1 — Direct primitive keywords          weight 0.90
    Words that ARE Wierzbicka primitives / emotion seeds.
    Handles morphological variants (miss → GRIEF, etc.)

  Layer 2 — Signal word clusters               weight 0.80
    Words that IMPLY primitives.
    Source: taxonomy.json decomposer_hints + exp 23 failure analysis.
    Handles both single-word and multi-word phrases.

  Layer 3 — Structural regex patterns          weight 0.85
    Syntactic patterns that carry compositional meaning.
    "I miss X", "about to", "I feel X but Y", etc.

Deduplication: when same primitive appears in multiple layers,
keep the highest weight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Fix 2: death signals — used in decompose() to strip PLACE from death contexts
_DEATH_SIGNALS = re.compile(
    r'\b(passed away|passed on|died|dead|death|lost my)\b', re.I
)

# Fix 3: epistemic "wonder" idiom — "wonder if/whether/why/how/when" means doubt/anxiety,
# not aesthetic wonder. Used in decompose() to strip the WONDER primitive.
_WONDER_EPISTEMIC = re.compile(
    r'\bwonder\s+(?:if|whether|why|how|when|what)\b', re.I
)


# ── Public interface ───────────────────────────────────────────────────────

@dataclass
class Primitive:
    word: str
    layer: str   # "0a" or "0b"
    weight: float


def decompose(prompt: str) -> list[Primitive]:
    """
    Input:  "I miss my hometown"
    Output: [
        Primitive(word="GRIEF",    layer="0b", weight=0.85),
        Primitive(word="NOSTALGIA",layer="0b", weight=0.85),
        Primitive(word="PLACE",    layer="0a", weight=0.80),
    ]
    """
    text = prompt.strip()
    if not text:
        return []

    # Collect {word → weight} from all three layers
    collected: dict[str, float] = {}

    _apply_layer3(text, collected)   # patterns  (0.85)
    _apply_layer2(text, collected)   # signals   (0.80) — lower, so L3 wins ties
    _apply_layer1(text, collected)   # direct    (0.90) — highest, always wins ties

    if not collected:
        return []

    # Fix 2: death context override — remove PLACE when explicit death signals
    # are present so "my dog passed away" maps to pure GRIEF not GRIEF+PLACE.
    if _DEATH_SIGNALS.search(text) and "PLACE" in collected:
        del collected["PLACE"]

    # Fix 3: "wonder if/whether/why/how/when" is the epistemic idiom "I'm not sure",
    # not aesthetic wonder. Remove WONDER so ANXIETY+KNOW from Layer 3 route correctly.
    if _WONDER_EPISTEMIC.search(text) and "WONDER" in collected:
        del collected["WONDER"]

    # Build Primitive objects, assign canonical layer, sort by weight
    result = [
        Primitive(word=word, layer=_layer(word), weight=round(weight, 2))
        for word, weight in collected.items()
    ]
    result.sort(key=lambda p: -p.weight)
    return result


# ── Layer assignment tables ────────────────────────────────────────────────

_LAYER_0A = frozenset({
    "KNOW", "TIME", "PLACE", "SOMEONE", "WANT", "FEEL", "THINK",
    "MOVE", "DO", "HAPPEN", "LIVE", "DIE", "GOOD", "BAD",
    "NOT", "MAYBE", "IF", "BECAUSE", "VERY", "MORE", "FAR", "NEAR",
    "BEFORE", "AFTER", "HERE", "GO", "SEE", "HEAR",
    "YOU", "I", "ALL", "SOME",
})

_LAYER_0B = frozenset({
    "GRIEF", "FEAR", "JOY", "NOSTALGIA", "SADNESS", "ANXIETY", "EXCITEMENT",
    "SATISFACTION", "ADMIRATION", "PRIDE", "SHAME", "GUILT", "RELIEF",
    "ENVY", "CONTEMPT", "AWE", "WONDER", "HORROR", "CONFUSION", "EMPATHY",
    "CALMNESS", "HOPE",
})


def _layer(word: str) -> str:
    if word in _LAYER_0A:
        return "0a"
    if word in _LAYER_0B:
        return "0b"
    return "0a"


def _add(collected: dict[str, float], word: str, weight: float) -> None:
    """Insert or raise weight — never lower an existing entry."""
    if collected.get(word, 0.0) < weight:
        collected[word] = weight


# ── Layer 1: Direct keyword map ────────────────────────────────────────────
# Surface word → canonical primitive.  Covers morphological variants.

_DIRECT: dict[str, str] = {
    # NSM operators
    "want":       "WANT",  "wants":    "WANT",  "wanted":   "WANT",
    "know":       "KNOW",  "knows":    "KNOW",  "knowing":  "KNOW",  "knew": "KNOW",
    "think":      "THINK", "thinks":   "THINK", "thought":  "THINK", "thinking": "THINK",
    "feel":       "FEEL",  "feels":    "FEEL",  "felt":     "FEEL",  "feeling":  "FEEL",
    "see":        "SEE",   "saw":      "SEE",   "seeing":   "SEE",
    "hear":       "HEAR",  "heard":    "HEAR",  "hearing":  "HEAR",
    "happen":     "HAPPEN","happens":  "HAPPEN","happened": "HAPPEN",
    "live":       "LIVE",  "living":   "LIVE",
    "die":        "DIE",   "died":     "DIE",   "dying":    "DIE",
    "good":       "GOOD",  "bad":      "BAD",
    "not":        "NOT",   "maybe":    "MAYBE", "if":       "IF",
    "because":    "BECAUSE","very":    "VERY",  "more":     "MORE",
    "far":        "FAR",   "near":     "NEAR",  "before":   "BEFORE","after":   "AFTER",
    "here":       "HERE",  "some":     "SOME",  "all":      "ALL",
    "go":         "GO",    "going":    "GO",
    "someone":    "SOMEONE","somebody":"SOMEONE",
    "place":      "PLACE", "where":    "PLACE",
    "time":       "TIME",  "when":     "TIME",
    "move":       "MOVE",  "moved":    "MOVE",
    # Emotion seeds
    "grief":       "GRIEF",  "grieving":  "GRIEF",
    "fear":        "FEAR",   "fearful":   "FEAR",   "afraid":   "FEAR",
    "joy":         "JOY",    "joyful":    "JOY",
    "nostalgia":   "NOSTALGIA","nostalgic":"NOSTALGIA",
    "sad":         "SADNESS","sadness":   "SADNESS","unhappy":  "SADNESS",
    "anxiety":     "ANXIETY","anxious":   "ANXIETY","nervous":  "ANXIETY",
    "excited":     "EXCITEMENT","excitement":"EXCITEMENT",
    "satisfaction":"SATISFACTION","satisfied":"SATISFACTION",
    "admiration":  "ADMIRATION","admire":  "ADMIRATION","admiring":"ADMIRATION",
    "pride":       "PRIDE",  "proud":     "PRIDE",
    "shame":       "SHAME",  "ashamed":   "SHAME",
    "guilt":       "GUILT",  "guilty":    "GUILT",
    "relief":      "RELIEF", "relieved":  "RELIEF",
    "envy":        "ENVY",   "envious":   "ENVY",   "jealous":  "ENVY",
    "contempt":    "CONTEMPT",
    "awe":         "AWE",
    "wonder":      "WONDER", "wondrous":  "WONDER",
    "horror":      "HORROR", "horrified": "HORROR",
    "confusion":   "CONFUSION","confused": "CONFUSION",
    "empathy":     "EMPATHY","empathic":  "EMPATHY","empathetic":"EMPATHY",
    "calm":        "CALMNESS","calmness": "CALMNESS","peaceful": "CALMNESS",
    "hope":        "HOPE",   "hopeful":   "HOPE",   "hoping":   "HOPE",
}

_W1 = 0.90

def _apply_layer1(text: str, collected: dict[str, float]) -> None:
    """Tokenize and match each word against the direct map."""
    # Strip punctuation, split on whitespace
    tokens = re.sub(r"[^\w\s']", " ", text).lower().split()
    for tok in tokens:
        prim = _DIRECT.get(tok)
        if prim:
            _add(collected, prim, _W1)


# ── Layer 2: Signal word clusters ─────────────────────────────────────────
# Single-word signals and multi-word phrases → list of implied primitives.
# taxonomy.json decomposer_hints + exp 23 failure analysis.

# Single-word signals (checked after tokenisation)
_SIGNAL_WORDS: dict[str, list[str]] = {
    # Loss
    "miss":         ["GRIEF", "NOSTALGIA"],
    "missing":      ["GRIEF", "NOSTALGIA"],
    "missed":       ["GRIEF", "NOSTALGIA"],
    "lost":         ["GRIEF"],
    "loss":         ["GRIEF"],
    "lose":         ["GRIEF"],
    "losing":       ["GRIEF"],
    "mourn":        ["GRIEF"],
    "mourning":     ["GRIEF"],
    "gone":         ["GRIEF"],
    "passed":       ["GRIEF"],
    "without":      ["GRIEF"],
    "heartache":    ["GRIEF"],
    "hollow":       ["GRIEF"],
    "ache":         ["GRIEF"],
    "heavy heart":  ["GRIEF"],
    "numb":         ["GRIEF"],
    "void":         ["GRIEF"],
    "torn up":      ["GRIEF"],
    "can't bear it":["GRIEF"],
    "raw":          ["GRIEF"],
    # Threat
    "scared":       ["FEAR"],
    "worried":      ["ANXIETY"],
    "worry":        ["ANXIETY"],
    "dread":        ["FEAR"],
    "terror":       ["FEAR"],
    "terrified":    ["FEAR"],
    "frightened":   ["FEAR"],
    "apprehensive": ["ANXIETY"],
    # Desire
    "wish":         ["WANT"],
    "wishing":      ["WANT"],
    "wished":       ["WANT"],
    "crave":        ["WANT"],
    "craving":      ["WANT"],
    "need":         ["WANT"],
    "needs":        ["WANT"],
    "longing":      ["WANT", "GRIEF"],
    "desire":       ["WANT"],
    "desires":      ["WANT"],
    # Temporal
    "still":        ["TIME"],
    "always":       ["TIME"],
    "never":        ["TIME", "GRIEF"],
    "since":        ["TIME"],
    "years":        ["TIME"],
    "ago":          ["TIME", "NOSTALGIA"],
    "lately":       ["TIME"],
    "soon":         ["TIME"],
    "tomorrow":     ["TIME", "ANXIETY"],
    "today":        ["TIME"],
    "yesterday":    ["TIME", "NOSTALGIA"],
    "childhood":    ["TIME", "NOSTALGIA"],
    "anymore":      ["TIME", "GRIEF"],
    "once":         ["TIME", "NOSTALGIA"],
    # Spatial
    "home":         ["PLACE"],
    "hometown":     ["PLACE", "NOSTALGIA"],
    "neighborhood": ["PLACE"],
    "city":         ["PLACE"],
    "there":        ["PLACE"],
    "away":         ["PLACE", "GRIEF"],
    "back":         ["PLACE", "NOSTALGIA"],
    "inside":       ["PLACE"],
    # Agent
    "friend":       ["SOMEONE"],
    "friends":      ["SOMEONE"],
    "family":       ["SOMEONE"],
    "kids":         ["SOMEONE"],
    "children":     ["SOMEONE"],
    "child":        ["SOMEONE"],
    "partner":      ["SOMEONE"],
    "grandparents": ["SOMEONE"],
    "grandparent":  ["SOMEONE"],
    "parent":       ["SOMEONE"],
    "parents":      ["SOMEONE"],
    "dog":          ["SOMEONE"],
    "pet":          ["SOMEONE"],
    "they":         ["SOMEONE"],
    "he":           ["SOMEONE"],
    "she":          ["SOMEONE"],
    "stranger":     ["SOMEONE"],
    # Epistemic
    "realize":      ["KNOW"],
    "realized":     ["KNOW"],
    "understand":   ["KNOW"],
    "understood":   ["KNOW"],
    "learn":        ["KNOW"],
    "learned":      ["KNOW"],
    "believe":      ["KNOW"],
    # Action
    "run":          ["MOVE"],
    "try":          ["MOVE", "WANT"],
    "trying":       ["MOVE", "WANT"],
    "act":          ["MOVE"],
    "change":       ["MOVE"],
    "work":         ["MOVE"],
    # Positive / activation
    "happy":        ["JOY"],
    "happiness":    ["JOY"],
    "glad":         ["JOY"],
    "love":         ["JOY", "SOMEONE"],
    "wonderful":    ["JOY"],
    "amazing":      ["JOY"],
    "alive":        ["LIVE", "EXCITEMENT"],
    # Negative
    "hurt":         ["GRIEF", "SADNESS"],
    "hurts":        ["GRIEF", "SADNESS"],
    "pain":         ["GRIEF", "SADNESS"],
    "broken":       ["GRIEF"],
    "awful":        ["SADNESS"],
    "terrible":     ["SADNESS"],
    "hate":         ["SADNESS", "BAD"],
    # Exp 23 additions
    "interview":    ["ANXIETY", "TIME"],
    "married":      ["JOY", "EXCITEMENT", "TIME"],
    "wedding":      ["JOY", "EXCITEMENT"],
    "job":          ["ANXIETY", "WANT"],
    "results":      ["ANXIETY", "FEAR"],
    "result":       ["ANXIETY", "FEAR"],
    "choices":      ["WANT", "ANXIETY"],
    "choice":       ["WANT", "ANXIETY"],
    "jealousy":     ["ENVY", "SADNESS"],
    "admired":      ["ADMIRATION", "SOMEONE"],
    "wondering":    ["KNOW", "ANXIETY"],
    "scary":        ["FEAR"],
    "overwhelming": ["ANXIETY"],
    "overwhelmed":  ["ANXIETY"],
    "lonely":       ["SADNESS"],
    "loneliness":   ["SADNESS"],
    "alone":        ["ANXIETY", "SADNESS"],
}

# Multi-word phrases — checked as substrings in lowercased prompt
# Ordered longest-first so "wish i had" matches before "wish"
_SIGNAL_PHRASES: list[tuple[str, list[str]]] = [
    ("passed away",    ["GRIEF"]),
    ("passed on",      ["GRIEF"]),
    ("no longer",      ["GRIEF", "TIME"]),
    ("used to",        ["TIME", "NOSTALGIA"]),
    ("long for",       ["WANT", "GRIEF"]),
    ("growing up",     ["TIME", "SADNESS"]),
    ("grew up",        ["TIME", "NOSTALGIA"]),
    ("wish i had",     ["WANT", "GRIEF"]),
    ("still think",    ["TIME", "GRIEF"]),
    ("better off",     ["KNOW", "GRIEF"]),
    ("hurt me",        ["GRIEF", "SOMEONE"]),
    ("moving to",      ["ANXIETY", "EXCITEMENT", "PLACE"]),
    ("move to",        ["ANXIETY", "EXCITEMENT", "PLACE"]),
    ("about to move",  ["ANXIETY", "EXCITEMENT", "PLACE"]),
    ("waiting to hear",["ANXIETY", "TIME"]),
    ("find out",       ["KNOW", "ANXIETY"]),
    ("makes me",       ["FEEL"]),
    ("getting married",["JOY", "EXCITEMENT", "TIME"]),
    ("get married",    ["JOY", "EXCITEMENT", "TIME"]),
]

_W2 = 0.80

def _apply_layer2(text: str, collected: dict[str, float]) -> None:
    lower = text.lower()

    # Multi-word phrases first (substring match)
    for phrase, prims in _SIGNAL_PHRASES:
        if phrase in lower:
            for p in prims:
                _add(collected, p, _W2)

    # Single-word signals (token match)
    tokens = re.sub(r"[^\w\s']", " ", lower).split()
    for tok in tokens:
        prims = _SIGNAL_WORDS.get(tok)
        if prims:
            for p in prims:
                _add(collected, p, _W2)


# ── Layer 3: Structural regex patterns ────────────────────────────────────
# Syntactic constructions that carry compositional meaning regardless of
# the specific content words.

# Maps pattern → fixed primitive list
_REGEX_RULES: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\bI miss\b",           re.I), ["GRIEF", "NOSTALGIA"]),
    (re.compile(r"\bI wish I had\b",     re.I), ["WANT", "GRIEF"]),
    (re.compile(r"\bI still\b",          re.I), ["TIME", "GRIEF"]),
    (re.compile(r"\babout to\b",         re.I), ["TIME", "EXCITEMENT"]),
    (re.compile(r"\bI'?m waiting\b",     re.I), ["ANXIETY", "TIME"]),
    (re.compile(r"\bI could have\b",     re.I), ["WANT", "GRIEF", "TIME"]),
    # Fix 3: WONDER disambiguation — "wonder if" = epistemic doubt, "wonder at" = aesthetic awe
    (re.compile(r"\bI wonder if\b",               re.I), ["ANXIETY", "KNOW"]),
    (re.compile(r"\bI wonder at\b",               re.I), ["AWE"]),
    (re.compile(r"\bI wonder\b(?!\s+(?:if|at)\b)",re.I), ["ANXIETY", "KNOW"]),
    (re.compile(r"\bpassed away\b",      re.I), ["GRIEF"]),
    (re.compile(r"\bno longer\b",        re.I), ["GRIEF", "TIME"]),
    (re.compile(r"\bused to\b",          re.I), ["TIME", "NOSTALGIA"]),
    (re.compile(r"\bbetter off without\b",re.I),["KNOW", "GRIEF"]),
    (re.compile(r"\bgrew up\b|\bgrowing up\b", re.I), ["TIME", "SADNESS"]),
    (re.compile(r"\bI'?m proud\b",       re.I), ["PRIDE"]),
    (re.compile(r"\bI feel proud\b",     re.I), ["PRIDE"]),
    (re.compile(r"\bI'?m scared\b",      re.I), ["FEAR"]),
    (re.compile(r"\bI'?m nervous\b",     re.I), ["ANXIETY", "FEAR"]),
    (re.compile(r"\bI'?m excited\b",     re.I), ["EXCITEMENT"]),
    (re.compile(r"\bI'?m getting married\b", re.I), ["JOY", "EXCITEMENT", "TIME"]),
    (re.compile(r"\bI have .{0,20} interview\b", re.I), ["ANXIETY", "TIME"]),
    (re.compile(r"\bfind out .{0,30} today\b",  re.I), ["KNOW", "ANXIETY", "TIME"]),
]

# Special compound pattern: "I feel X but (also) Y"
_FEEL_BUT = re.compile(
    r"\bI feel (.+?)\s+but(?:\s+also)?\s+(.+)", re.I
)

# Inline emotion words extractable from compound clauses
_INLINE_EMOTIONS: dict[str, str] = {
    "proud":     "PRIDE",    "pride":       "PRIDE",
    "sad":       "SADNESS",  "sadness":     "SADNESS",   "unhappy": "SADNESS",
    "happy":     "JOY",      "happiness":   "JOY",       "glad":    "JOY",
    "jealous":   "ENVY",     "envious":     "ENVY",
    "angry":     "SADNESS",  "anger":       "SADNESS",
    "scared":    "FEAR",     "afraid":      "FEAR",      "fearful": "FEAR",
    "excited":   "EXCITEMENT","enthusiasm":  "EXCITEMENT",
    "anxious":   "ANXIETY",  "nervous":     "ANXIETY",   "worried": "ANXIETY",
    "guilty":    "GUILT",    "guilt":       "GUILT",
    "ashamed":   "SHAME",    "shame":       "SHAME",
    "alive":     "LIVE",
    "lonely":    "SADNESS",  "loneliness":  "SADNESS",
    "hopeful":   "HOPE",     "hope":        "HOPE",
    "grateful":  "SATISFACTION",
    "relieved":  "RELIEF",   "relief":      "RELIEF",
    "amazed":    "AWE",      "awe":         "AWE",       "wonderful": "JOY",
    "closer":    "SOMEONE",  "close":       "SOMEONE",
    "hurt":      "GRIEF",
    "nostalgic": "NOSTALGIA","wistful":     "NOSTALGIA",
    "lost":      "GRIEF",
    "overwhelmed":"ANXIETY",
}

_W3 = 0.85

def _apply_layer3(text: str, collected: dict[str, float]) -> None:
    # Fixed-list regex rules
    for pattern, prims in _REGEX_RULES:
        if pattern.search(text):
            for p in prims:
                _add(collected, p, _W3)

    # Compound "I feel X but Y" — extract emotion words from both halves
    m = _FEEL_BUT.search(text)
    if m:
        for half in (m.group(1), m.group(2)):
            tokens = re.sub(r"[^\w\s]", " ", half).lower().split()
            for tok in tokens:
                prim = _INLINE_EMOTIONS.get(tok)
                if prim:
                    _add(collected, prim, _W3)
