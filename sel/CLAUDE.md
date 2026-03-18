# SEL — Semantic Emotional Layer

## What this is
The SEL is an application layer built on top of 22 
experiments validating the existence of semantic 
primitives in small language models. It takes 
emotional/relational natural language prompts and 
processes them through three stages:
1. Primitive decomposition
2. Graph-based reasoning  
3. Membrane rendering

## What this is NOT
- Not a general-purpose LLM replacement
- Not a code generation system
- Not a factual reasoning system
Scope: emotional and relational language ONLY

## Architecture
See architecture spec in README.md
Four components: decomposer, reasoner, membrane, router

## The graph
primitive_graph.json is the core knowledge artifact
Do not modify it manually — it is built from validated
experimental data

## The membrane
Use qwen2.5:0.5b via Ollama as default membrane
It is intentionally small — rendering only, not reasoning
System prompt is critical — see membrane.py

## Testing philosophy
Every component has unit tests
The pipeline has integration tests
Test prompts should be emotional/relational ONLY
Example valid prompts:
  "I miss my old friends"
  "I'm nervous about tomorrow"
  "I feel proud of what I've accomplished"
Example invalid prompts (out of scope):
  "Write me a Python function"
  "What is the capital of France"

## Key files from prior research
/data/ground_truth_layer1.json  — validated compositions
/data/wierzbicka_primitives.json — full 65 primitives
/data/cowen_keltner_emotions.json — 27 emotions

## Success criteria
Given: "I miss my hometown"
Expected pipeline:
  decompose → [GRIEF, PLACE, NOSTALGIA]
  reason    → [homesickness, exile]  
  render    → empathetic English response
  
The response should feel natural and emotionally resonant
NOT like a definition or explanation
```

---

**Before you hand this to Claude Code, two things to do here first:**

**Thing 1 — Let me generate the full rule taxonomy.**

I have six rule classes sketched. The full taxonomy needs 20-30 to cover the entire emotional-relational space. That's a 30-minute conversation here that saves Claude Code hours of guessing. Want me to generate it now?

**Thing 2 — Decide on the decomposer strategy.**

The decomposer is the hardest component. Three options:
```
Option A: Rule-based keyword matching
          Fast, deterministic, brittle
          "miss" → GRIEF + NOSTALGIA
          "nervous" → FEAR + ANXIETY
          
Option B: Small classifier model
          More flexible, requires training data
          Fine-tuned on primitive labeling task
          
Option C: Use Ollama for decomposition too
          Feed prompt to qwen with structured output
          Extract primitives from JSON response
          Slower but more accurate initially

`taxonomy.json`  contains the entire reasoning layer broken down to use for the experiments in the 'semantic-emotional layer'