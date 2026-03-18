# SEL — Semantic Emotional Layer

An application layer that takes emotional and relational natural language prompts and processes them through three stages: primitive decomposition, graph-based reasoning, and membrane rendering.

Built on top of 22 experiments validating the existence of semantic primitives in small language models.

---

## Architecture

```
prompt
  │
  ▼
decomposer.py     → [GRIEF(0b,0.8), PLACE(0a,0.7), NOSTALGIA(0b,0.9)]
  │                  Uses qwen2.5:0.5b via Ollama — structured JSON output
  ▼
reasoner.py       → [homesickness(rule_class=K), longing(rule_class=A)]
  │                  Pure Python — graph traversal, NO LLM calls
  │                  Source of truth: core/primitive_graph.json
  ▼
membrane.py       → "That pull toward a place that shaped you —
                      it never quite leaves you, does it."
                     Uses qwen2.5:0.5b via Ollama — renderer only
```

### Components

| File | Role | LLM |
|------|------|-----|
| `core/decomposer.py` | Prompt → semantic primitives | qwen2.5:0.5b |
| `core/reasoner.py` | Primitives → Layer 1 concepts | None (pure Python) |
| `core/membrane.py` | Concepts → empathetic English | qwen2.5:0.5b |
| `core/router.py` | Pipeline orchestration + scope validation | None |
| `core/primitive_graph.json` | Knowledge graph (source of truth) | — |
| `core/composition_rules.json` | Rule class index for reasoner | — |

### The graph

`core/primitive_graph.json` contains:
- **Validated edges** (weight ≥ 0.75): confirmed by experiments 19b and 20 with multi-model agreement
- **Theoretical edges** (weight 0.38–0.55): derived from taxonomy.json rule classes G–Z, awaiting empirical validation
- **Special case handlers**: JOY attractor, FEEL redundancy, BAD instability, NOT complexity

### Scope

SEL handles **emotional and relational language only**. It will return a helpful message for out-of-scope requests (code generation, factual queries, etc.).

---

## Installation

```bash
pip install requests
```

SEL has no other Python dependencies beyond the standard library.

---

## Setup

You need [Ollama](https://ollama.com) running locally with the model pulled:

```bash
# Install Ollama, then:
ollama pull qwen2.5:0.5b
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

---

## Usage

### Full pipeline

```python
import sys
sys.path.insert(0, "/path/to/graph-oriented-generation")

from sel.core.router import process

response = process("I miss my hometown")
print(response)
# → "That pull toward a place that shaped you — it never quite leaves you, does it."
```

### Debug mode (see all stages)

```python
from sel.core.router import process_debug

result = process_debug("I miss my hometown")
print(result["primitives"])  # [{'word': 'GRIEF', 'layer': '0b', 'weight': 0.85}, ...]
print(result["concepts"])    # [{'name': 'homesickness', 'rule_class': 'K', ...}, ...]
print(result["response"])    # Natural empathetic English
```

### Individual stages

```python
from sel.core.decomposer import decompose
from sel.core.reasoner   import reason
from sel.core.membrane   import render

primitives = decompose("I miss my hometown")
# [Primitive(word='GRIEF', layer='0b', weight=0.85), ...]

concepts = reason(primitives)
# [Concept(name='homesickness', rule_class='K', confidence=0.72), ...]

response = render(concepts, "I miss my hometown")
# "That pull toward a place that shaped you..."
```

---

## Example: Full pipeline trace

**Input:** `"I miss my hometown"`

```
[SEL] PROMPT: 'I miss my hometown'
[SEL] PRIMITIVES (3): GRIEF(0b,0.85) PLACE(0a,0.70) NOSTALGIA(0b,0.90)
[SEL] CONCEPTS (2): homesickness(rule=K,conf=0.65) longing(rule=A,conf=0.72)
[SEL] RESPONSE: "That pull toward a place that shaped you — it never quite
                  leaves you, does it. Something about it still lives in you."
```

**What happened:**
1. Decomposer identified `GRIEF + PLACE + NOSTALGIA` from "miss" and "hometown"
2. Reasoner matched `PLACE + GRIEF` → `homesickness` via Rule K (Space × Loss)
3. Reasoner matched `WANT/GRIEF + NOSTALGIA` → `longing` via Rule A (Desire × Loss)
4. Membrane rendered the concepts as natural, empathetic English

---

## Testing

```bash
# From the repo root
python -m pytest sel/tests/ -v

# Individual test files
python -m pytest sel/tests/test_reasoner.py -v    # No Ollama needed
python -m pytest sel/tests/test_decomposer.py -v  # Requires Ollama
python -m pytest sel/tests/test_membrane.py -v    # Requires Ollama
python -m pytest sel/tests/test_pipeline.py -v    # Requires Ollama
```

> `test_reasoner.py` is pure Python — no Ollama required. Safe for CI.

---

## Rule classes

| Class | Name | Pattern | Status |
|-------|------|---------|--------|
| A | Desire × Loss | WANT + grief → longing | Validated |
| B | Time × Positive | TIME + excitement → anticipation | Validated |
| C | Time × Loss | TIME + grief → mourning | Validated |
| D | Agent × Upward | SOMEONE + admiration → recognition | Validated |
| E | Action × Arousal | MOVE + excitement → momentum | Validated |
| F | Epistemic × Completion | KNOW + satisfaction → understanding | Validated |
| G–Z | (theoretical) | Various operator × seed combinations | Unvalidated |

---

## Special cases

| Case | Trigger | Handling |
|------|---------|----------|
| JOY attractor | JOY as seed | Bypass composition → render joy directly |
| FEEL redundancy | FEEL as operator | Strip FEEL, use seed directly |
| BAD instability | BAD without referent | Skip composition |
| NOT complexity | NOT in primitives | Flag for membrane only — skip graph |

---

## Data files

| File | Contents |
|------|----------|
| `data/wierzbicka_primitives.json` | Full 65 NSM primitives with SEL type assignments |
| `data/cowen_keltner_emotions.json` | 27 Cowen & Keltner emotion categories → seed types |
| `data/ground_truth_layer1.json` | 9 validated compositions from experiments 19b+20 |
| `data/validated_compositions.json` | Full validated edge list with confidence scores |

---

## Constraints

- **Membrane** is qwen2.5:0.5b — intentionally small, rendering only
- **Reasoner** is pure Python — no LLM calls in the reasoning step
- **Scope** is emotional/relational language only
- **Graph** is the source of truth — logic lives in `primitive_graph.json`, not hardcoded
