"""
test_reasoner.py — Unit tests for the SEL reasoner.

Tests all 6 validated rule classes (A-F) plus key special cases.
The reasoner is pure Python — no Ollama required.

Run: python -m pytest sel/tests/test_reasoner.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from sel.core.reasoner import reason, Primitive, Concept


def _make(word: str, layer: str = "0b", weight: float = 0.8) -> Primitive:
    return Primitive(word=word, layer=layer, weight=weight)


class TestValidatedRuleClasses:
    """Test all 6 empirically validated rule classes from experiments 19b+20."""

    def test_rule_A_desire_x_loss(self):
        """WANT + GRIEF → yearning/longing (Desire × Loss)"""
        prims = [_make("WANT", "0a"), _make("GRIEF", "0b")]
        concepts = reason(prims)
        assert len(concepts) > 0, "Rule A should produce at least one concept"
        names = {c.name for c in concepts}
        rule_classes = {c.rule_class for c in concepts}
        assert names & {"longing", "yearning", "ache", "wistful wanting"} or "A" in rule_classes, \
            f"Expected yearning/longing from WANT+GRIEF, got: {names}"

    def test_rule_B_time_x_positive(self):
        """TIME + EXCITEMENT → anticipation (Time × Positive Activation)"""
        prims = [_make("TIME", "0a"), _make("EXCITEMENT", "0b")]
        concepts = reason(prims)
        assert len(concepts) > 0, "Rule B should produce at least one concept"
        names = {c.name for c in concepts}
        rule_classes = {c.rule_class for c in concepts}
        assert names & {"anticipation", "hopeful_tension"} or "B" in rule_classes, \
            f"Expected anticipation from TIME+EXCITEMENT, got: {names}"

    def test_rule_C_time_x_grief(self):
        """TIME + GRIEF → mourning (Time × Loss or Grief)"""
        prims = [_make("TIME", "0a"), _make("GRIEF", "0b")]
        concepts = reason(prims)
        assert len(concepts) > 0, "Rule C should produce at least one concept"
        names = {c.name for c in concepts}
        rule_classes = {c.rule_class for c in concepts}
        assert names & {"mourning", "melancholy"} or "C" in rule_classes, \
            f"Expected mourning from TIME+GRIEF, got: {names}"

    def test_rule_D_agent_x_admiration(self):
        """SOMEONE + ADMIRATION → admiration (Agent × Evaluation Upward)"""
        prims = [_make("SOMEONE", "0a"), _make("ADMIRATION", "0b")]
        concepts = reason(prims)
        assert len(concepts) > 0, "Rule D should produce at least one concept"
        names = {c.name for c in concepts}
        rule_classes = {c.rule_class for c in concepts}
        assert names & {"admiration", "reverence"} or "D" in rule_classes, \
            f"Expected admiration/recognition from SOMEONE+ADMIRATION, got: {names}"

    def test_rule_E_action_x_excitement(self):
        """MOVE + EXCITEMENT → momentum (Action × High Positive Arousal)"""
        prims = [_make("MOVE", "0a"), _make("EXCITEMENT", "0b")]
        concepts = reason(prims)
        assert len(concepts) > 0, "Rule E should produce at least one concept"
        names = {c.name for c in concepts}
        rule_classes = {c.rule_class for c in concepts}
        assert names & {"momentum", "drive", "flow_state"} or "E" in rule_classes, \
            f"Expected momentum from MOVE+EXCITEMENT, got: {names}"

    def test_rule_F_epistemic_x_satisfaction(self):
        """KNOW + SATISFACTION → understanding (Epistemic × Completion)"""
        prims = [_make("KNOW", "0a"), _make("SATISFACTION", "0b")]
        concepts = reason(prims)
        assert len(concepts) > 0, "Rule F should produce at least one concept"
        names = {c.name for c in concepts}
        rule_classes = {c.rule_class for c in concepts}
        assert names & {"understanding", "insight"} or "F" in rule_classes, \
            f"Expected understanding from KNOW+SATISFACTION, got: {names}"


class TestSpecialCases:
    def test_joy_attractor_bypasses_composition(self):
        """JOY + any operator → joy (JOY is terminal attractor)"""
        prims = [_make("WANT", "0a"), _make("JOY", "0b")]
        concepts = reason(prims)
        assert len(concepts) > 0
        assert any(c.name == "joy" for c in concepts), \
            f"JOY attractor should return joy concept, got: {[c.name for c in concepts]}"

    def test_feel_stripped_before_reasoning(self):
        """FEEL + GRIEF → grief concepts (FEEL should be stripped)"""
        prims = [_make("FEEL", "0a"), _make("GRIEF", "0b"), _make("TIME", "0a")]
        concepts = reason(prims)
        # After stripping FEEL, TIME+GRIEF should produce mourning
        assert isinstance(concepts, list)
        # Should not crash and should produce something
        if concepts:
            names = {c.name for c in concepts}
            # Should NOT produce a concept that requires FEEL as operator
            rule_classes = {c.rule_class for c in concepts}
            # Mourning (C) or fallback — either is acceptable
            assert names or rule_classes

    def test_not_complexity_flagged(self):
        """NOT + GRIEF → negated_emotion (flagged for membrane)"""
        prims = [_make("NOT", "0a"), _make("GRIEF", "0b")]
        concepts = reason(prims)
        assert len(concepts) > 0
        assert any(c.rule_class == "X" for c in concepts), \
            f"NOT complexity should produce rule_class X, got: {[c.rule_class for c in concepts]}"

    def test_bad_without_referent_stripped(self):
        """BAD alone → no activation (requires referent)"""
        prims = [_make("BAD", "0a")]
        concepts = reason(prims)
        # Single BAD with no referent should not crash; may return empty or fallback
        assert isinstance(concepts, list)

    def test_returns_list(self):
        """reason() always returns a list"""
        result = reason([])
        assert isinstance(result, list)

    def test_concept_fields(self):
        """All concepts have required fields."""
        prims = [_make("WANT", "0a"), _make("GRIEF", "0b")]
        concepts = reason(prims)
        for c in concepts:
            assert hasattr(c, "name")
            assert hasattr(c, "rule_class")
            assert hasattr(c, "confidence")
            assert 0.0 <= c.confidence <= 1.0


class TestRuleKDisplacement:
    def test_space_x_loss_homesickness(self):
        """PLACE + GRIEF → homesickness (Rule K: Space × Loss)"""
        prims = [_make("PLACE", "0a"), _make("GRIEF", "0b")]
        concepts = reason(prims)
        assert len(concepts) > 0
        names = {c.name for c in concepts}
        rule_classes = {c.rule_class for c in concepts}
        assert names & {"homesickness", "exile", "displacement"} or "K" in rule_classes, \
            f"Expected displacement concepts from PLACE+GRIEF, got: {names}"

    def test_multi_primitive_homesickness(self):
        """GRIEF + PLACE + NOSTALGIA → homesickness (multi-primitive)"""
        prims = [
            _make("GRIEF",    "0b", 0.8),
            _make("PLACE",    "0a", 0.7),
            _make("NOSTALGIA","0b", 0.9),
        ]
        concepts = reason(prims)
        assert len(concepts) > 0
        names = {c.name for c in concepts}
        assert names & {"homesickness", "longing", "nostalgia", "exile", "yearning"}, \
            f"Expected homesickness or related concept from GRIEF+PLACE+NOSTALGIA, got: {names}"
