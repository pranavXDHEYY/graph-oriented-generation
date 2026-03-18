"""
reasoner.py — Stage 2 of the SEL pipeline.
Pure Python — NO Ollama calls.

Loads primitive_graph.json at startup.
For each pair of primitives, looks up edges in the graph.
Applies rule classes from taxonomy.json.
Handles special cases: JOY attractor, FEEL redundancy, BAD instability,
NOT complexity, and INTENSIFIER modifiers.

Input:  list[Primitive]
Output: list[Concept]
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from itertools import combinations

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_GRAPH_PATH = os.path.join(_HERE, "primitive_graph.json")
_RULES_PATH = os.path.join(_HERE, "composition_rules.json")

# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class Primitive:
    word: str
    layer: str
    weight: float


@dataclass
class Concept:
    name: str
    rule_class: str
    confidence: float
    source_primitives: list[str] = field(default_factory=list)
    output_type: str = ""
    validated: bool = False


# ── Graph loader ───────────────────────────────────────────────────────────
_graph: dict = {}
_rules: dict = {}


def _load() -> None:
    global _graph, _rules
    if not _graph:
        with open(_GRAPH_PATH, "r") as f:
            _graph = json.load(f)
    if not _rules:
        with open(_RULES_PATH, "r") as f:
            _rules = json.load(f)


# ── Public API ─────────────────────────────────────────────────────────────
def reason(primitives: list[Primitive]) -> list[Concept]:
    """
    Input:  [GRIEF, PLACE, NOSTALGIA]
    Output: [
        Concept(name="homesickness", rule_class="K", confidence=0.72),
        Concept(name="exile",        rule_class="K", confidence=0.68),
    ]
    """
    _load()

    # Pre-process: apply special cases before graph traversal
    primitives, flags = _preprocess(primitives)

    if not primitives:
        return []

    # If NOT complexity was flagged, return a sentinel concept for membrane
    if flags.get("not_complexity"):
        return [Concept(
            name="negated_emotion",
            rule_class="X",
            confidence=0.40,
            source_primitives=[p.word for p in primitives],
            output_type="absence_or_inversion",
        )]

    # If JOY attractor was detected, bypass composition
    if flags.get("joy_attractor"):
        return [Concept(
            name="joy",
            rule_class="H",
            confidence=1.00,
            source_primitives=["JOY"],
            output_type="positive_emotion",
            validated=True,
        )]

    concepts = []

    # Try all pairwise combinations first
    words = [p.word for p in primitives]
    weight_map = {p.word: p.weight for p in primitives}

    concepts.extend(_lookup_edges(words, weight_map, flags))

    # Try rule-class inference for any primitives not matched by edges
    matched_words = {w for c in concepts for w in c.source_primitives}
    unmatched = [p for p in primitives if p.word not in matched_words]
    if unmatched and len(unmatched) >= 2:
        concepts.extend(_infer_by_rule_class(unmatched, flags))

    # Deduplicate by concept name, keeping highest confidence
    concepts = _deduplicate(concepts)

    # Fix 5: resolve conflicting rule classes — when 3+ distinct rules fire
    # simultaneously the membrane loses focus; keep only the highest-confidence concept
    concepts = _resolve_conflicts(concepts)

    # Apply intensifier scaling
    if flags.get("intensifier"):
        mult = _graph["special_cases"]["INTENSIFIER_MODIFIER"]["multiplier"]
        for c in concepts:
            c.confidence = min(1.0, c.confidence * mult)

    return concepts if concepts else _fallback(primitives)


# ── Preprocessing ──────────────────────────────────────────────────────────
def _preprocess(primitives: list[Primitive]) -> tuple[list[Primitive], dict]:
    """Apply special cases. Returns modified list and flag dict."""
    flags: dict = {}
    words = [p.word for p in primitives]

    # FEEL redundancy — strip it
    feel_stripped = [p for p in primitives if p.word != "FEEL"]
    if len(feel_stripped) < len(primitives):
        primitives = feel_stripped
        words = [p.word for p in primitives]

    # NOT complexity — flag and continue (don't strip)
    if "NOT" in words:
        flags["not_complexity"] = True

    # JOY attractor — detect JOY as seed with any operator.
    # Fix 1: when social_negative (ENVY, CONTEMPT) co-occurs with JOY,
    # skip the attractor so the pair routes to Rule I (social pain) instead.
    joy_present = any(p.word == "JOY" for p in primitives)
    social_negative_present = any(
        _get_prim_type(p.word) == "social_negative"
        for p in primitives if p.word != "JOY"
    )
    operators_present = any(
        _get_prim_type(p.word) in {
            "desire", "temporal", "spatial", "agent",
            "epistemic", "action", "evaluative", "logical", "existential"
        }
        for p in primitives if p.word != "JOY"
    )
    if joy_present and operators_present and not social_negative_present:
        flags["joy_attractor"] = True

    # BAD instability — require referent
    bad_words = [p for p in primitives if p.word == "BAD"]
    if bad_words:
        has_referent = any(
            _get_prim_type(p.word) in {"agent", "loss_emotion", "threat_emotion"}
            for p in primitives if p.word != "BAD"
        )
        if not has_referent:
            primitives = [p for p in primitives if p.word != "BAD"]
            words = [p.word for p in primitives]

    # Intensifier — track but don't remove
    if any(p.word in {"VERY", "MORE", "ALL"} for p in primitives):
        flags["intensifier"] = True

    return primitives, flags


# ── Edge lookup ────────────────────────────────────────────────────────────
def _lookup_edges(words: list[str], weight_map: dict, _flags: dict) -> list[Concept]:
    """Look up all word pairs in both validated and theoretical edge tables."""
    concepts = []
    all_edges = {}
    all_edges.update(_graph["edges"]["validated"])
    all_edges.update(_graph["edges"]["theoretical"])

    # Fix 4: Rule L (Space × Threat) only fires when a physical location
    # primitive (PLACE or HERE) is explicitly present — not just FAR.
    location_present = any(w in {"PLACE", "HERE"} for w in words)

    # Try all 2-combinations of words
    for w1, w2 in combinations(words, 2):
        for key in [f"{w1}+{w2}", f"{w2}+{w1}"]:
            if key in all_edges:
                edge = all_edges[key]
                if edge.get("rule_class") == "L" and not location_present:
                    continue  # Fix 4: skip spatial menace without explicit location
                avg_weight = (weight_map.get(w1, 0.5) + weight_map.get(w2, 0.5)) / 2
                edge_weight = edge["weight"]
                confidence = round(edge_weight * avg_weight * 1.2, 3)
                confidence = min(1.0, confidence)

                concept_name = edge["concept"]
                concept_node = _graph["nodes"]["concepts"].get(concept_name, {})
                concepts.append(Concept(
                    name=concept_name,
                    rule_class=edge["rule_class"],
                    confidence=confidence,
                    source_primitives=[w1, w2],
                    output_type=concept_node.get("output_type", ""),
                    validated=edge.get("validated", False),
                ))

    # Try 3-word combined keys too
    if len(words) >= 3:
        for triple in combinations(words, 3):
            key = "+".join(sorted(triple))
            if key in all_edges:
                edge = all_edges[key]
                if edge.get("rule_class") == "L" and not location_present:
                    continue  # Fix 4
                avg_weight = sum(weight_map.get(w, 0.5) for w in triple) / 3
                confidence = round(edge["weight"] * avg_weight * 1.2, 3)
                confidence = min(1.0, confidence)
                concept_name = edge["concept"]
                concept_node = _graph["nodes"]["concepts"].get(concept_name, {})
                concepts.append(Concept(
                    name=concept_name,
                    rule_class=edge["rule_class"],
                    confidence=confidence,
                    source_primitives=list(triple),
                    output_type=concept_node.get("output_type", ""),
                    validated=edge.get("validated", False),
                ))

    return concepts


# ── Rule-class inference ───────────────────────────────────────────────────
def _infer_by_rule_class(primitives: list[Primitive], _flags: dict) -> list[Concept]:
    """
    When no direct edge match exists, infer concept type from operator + seed
    type using the composition_rules.json rule classes.
    """
    concepts = []
    rules = _rules["rules"]
    type_map = _rules["primitive_type_map"]

    # Fix 4: Rule L (Space × Threat) requires explicit physical location primitive
    all_words = {p.word for p in primitives}
    location_present = bool(all_words & {"PLACE", "HERE"})

    for prim_a, prim_b in combinations(primitives, 2):
        type_a = type_map.get(prim_a.word, "")
        type_b = type_map.get(prim_b.word, "")
        if not type_a or not type_b:
            continue

        for rule_id, rule in rules.items():
            # Fix 4: skip spatial menace without explicit physical location
            if rule_id == "L" and not location_present:
                continue

            op_type = rule["operator_type"]
            seed_type = rule["seed_type"]

            matched = (
                (type_a == op_type and type_b == seed_type) or
                (type_b == op_type and type_a == seed_type)
            )
            if not matched:
                continue

            rule_confidence = rule["confidence"]
            avg_prim = (prim_a.weight + prim_b.weight) / 2
            confidence = round(rule_confidence * avg_prim, 3)
            if confidence < _graph["traversal"]["minimum_confidence"]:
                continue

            # Pick the best output concept for this rule class
            output_type = rule["output_type"]
            concept_name = _best_concept_for_rule(rule_id, output_type)

            concepts.append(Concept(
                name=concept_name,
                rule_class=rule_id,
                confidence=confidence,
                source_primitives=[prim_a.word, prim_b.word],
                output_type=output_type,
                validated=rule["validated"],
            ))
            break  # one rule match per pair is enough

    return concepts


def _best_concept_for_rule(rule_id: str, output_type: str) -> str:
    """Return the highest-confidence validated concept for a rule class."""
    best_name = output_type.replace("_", " ")
    best_conf = -1.0
    for name, node in _graph["nodes"]["concepts"].items():
        if node.get("rule_class") == rule_id:
            conf = node.get("confidence", 0.0)
            if conf > best_conf:
                best_conf = conf
                best_name = name
    return best_name


# ── Helpers ────────────────────────────────────────────────────────────────
def _get_prim_type(word: str) -> str:
    type_map = _rules["primitive_type_map"] if _rules else {}
    return type_map.get(word, "")


def _resolve_conflicts(concepts: list[Concept]) -> list[Concept]:
    """
    Fix 5: when 3+ distinct rule classes fire simultaneously the membrane
    loses focus and underperforms. Keep only the highest-confidence concept.
    Two-class combinations (e.g. PRIDE + SADNESS) are valid mixed states — kept.

    Tie-breaking: threat/loss/anxiety rule classes (C, G, K, L, O, R, U) take
    priority over activation/positive classes (B, E, H, T) so that prompts like
    "moving to a new city alone" route to mourning/displacement rather than
    momentum — ALONE's anxiety signal overrides the MOVE+EXCITEMENT edge.
    """
    _THREAT_LOSS = {"C", "G", "K", "L", "O", "R", "U"}
    distinct_rules = {c.rule_class for c in concepts if c.rule_class != "__fallback__"}
    if len(distinct_rules) >= 3:
        return [max(concepts, key=lambda c: (c.rule_class in _THREAT_LOSS, c.confidence))]
    return concepts


def _deduplicate(concepts: list[Concept]) -> list[Concept]:
    seen: dict[str, Concept] = {}
    for c in concepts:
        if c.name not in seen or c.confidence > seen[c.name].confidence:
            seen[c.name] = c
    return sorted(seen.values(), key=lambda x: -x.confidence)


def _fallback(primitives: list[Primitive]) -> list[Concept]:
    """When no graph path found, return a generic concept signaling direct membrane routing."""
    if not primitives:
        return []
    top = max(primitives, key=lambda p: p.weight)
    return [Concept(
        name=top.word.lower(),
        rule_class="__fallback__",
        confidence=top.weight * 0.6,
        source_primitives=[top.word],
        output_type="raw_primitive",
    )]
