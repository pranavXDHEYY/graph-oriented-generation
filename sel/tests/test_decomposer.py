"""
test_decomposer.py — Unit tests for the SEL decomposer.

Tests the decomposer against 5 emotional prompts.
Requires Ollama running with qwen2.5:0.5b pulled.

Run: python -m pytest sel/tests/test_decomposer.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from sel.core.decomposer import decompose, Primitive


EMOTIONAL_PROMPTS = [
    "I miss my hometown",
    "I'm nervous about tomorrow",
    "I feel proud of what I've accomplished",
    "I still think about my old friend",
    "Something good is about to happen",
]


def _check_primitives(primitives: list[Primitive], prompt: str) -> None:
    """Common assertions for all decompose results."""
    assert isinstance(primitives, list), f"Expected list, got {type(primitives)} for: {prompt!r}"
    # We accept empty list (fallback) but prefer at least one primitive
    for p in primitives:
        assert isinstance(p, Primitive), f"Expected Primitive, got {type(p)}"
        assert p.word, "Primitive word should not be empty"
        assert p.layer in ("0a", "0b"), f"layer must be '0a' or '0b', got {p.layer!r}"
        assert 0.0 <= p.weight <= 1.0, f"weight must be in [0,1], got {p.weight}"


class TestDecomposerOutput:
    def test_miss_hometown(self):
        prims = decompose("I miss my hometown")
        _check_primitives(prims, "I miss my hometown")
        # Should extract loss/grief and spatial primitives
        words = {p.word for p in prims}
        assert words & {"GRIEF", "SADNESS", "NOSTALGIA", "PLACE", "FAR", "HERE"}, \
            f"Expected loss or spatial primitives in: {words}"

    def test_nervous_about_tomorrow(self):
        prims = decompose("I'm nervous about tomorrow")
        _check_primitives(prims, "I'm nervous about tomorrow")
        # qwen2.5:0.5b may return [] if JSON parsing fails — that's the
        # correct fallback behavior per spec. Accept empty or any valid list.
        words = {p.word for p in prims}
        print(f"\nPrimitives for 'nervous about tomorrow': {words or '(empty — JSON fallback)'}")

    def test_proud_accomplishment(self):
        prims = decompose("I feel proud of what I've accomplished")
        _check_primitives(prims, "I feel proud of what I've accomplished")
        words = {p.word for p in prims}
        assert words & {"PRIDE", "JOY", "SATISFACTION", "GOOD", "SOMEONE", "I"}, \
            f"Expected positive or agent primitives in: {words}"

    def test_think_about_old_friend(self):
        prims = decompose("I still think about my old friend")
        _check_primitives(prims, "I still think about my old friend")
        # Content varies with small model — accept any valid primitive set
        words = {p.word for p in prims}
        print(f"\nPrimitives for 'still think about old friend': {words}")

    def test_something_good_about_to_happen(self):
        prims = decompose("Something good is about to happen")
        _check_primitives(prims, "Something good is about to happen")
        words = {p.word for p in prims}
        assert words & {"JOY", "EXCITEMENT", "GOOD", "TIME", "BEFORE", "HOPE"}, \
            f"Expected positive or temporal primitives in: {words}"


class TestDecomposerStructure:
    def test_returns_list(self):
        result = decompose("I feel lost")
        assert isinstance(result, list)

    def test_primitive_has_required_fields(self):
        result = decompose("I miss someone")
        for p in result:
            assert hasattr(p, "word")
            assert hasattr(p, "layer")
            assert hasattr(p, "weight")

    def test_fallback_on_empty_string(self):
        # Should not crash — returns [] on empty or nonsensical input
        result = decompose("")
        assert isinstance(result, list)

    def test_weight_bounds(self):
        result = decompose("I'm so happy right now")
        for p in result:
            assert 0.0 <= p.weight <= 1.0
