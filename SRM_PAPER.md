# Symbolic Reasoning Membranes: An Empirical Investigation of Semantic Primitives in Small Language Models

---

## Abstract

We began with a deceptively simple question: can the meaning carried by natural language be compressed into symbolic representations, and if so, can a language model serve as the bridge between those symbols and fluent English output?

This question sits at an uncomfortable intersection. Language models are trained to predict tokens, not to reason symbolically. Yet they produce outputs that appear to carry meaning. Something is happening between input and output that resembles understanding — but nobody has systematically probed what that something is, or whether it can be made explicit, controllable, and reproducible.

The standard assumption in the field is that language models are black boxes: you put language in, you get language out, and the internal process is opaque by design. We took a different position. We treated the model not as a black box but as a **membrane** — a medium that transforms input into output according to rules that might be discoverable through systematic experimentation.

Our approach was empirical rather than theoretical. Rather than proposing an architecture and proving it mathematically, we ran experiments. We fed language models inputs ranging from random phonemes to universal semantic primitives proposed by linguist Anna Wierzbicka in the 1970s. We measured outputs, classified behaviors, and let findings accumulate into a theory. Seventeen experiments later, that theory has a name: the **Symbolic Reasoning Membrane**, or **SRM**.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Membrane: Discovering Behavioral States (Experiments 1–9)](#2-the-membrane-discovering-behavioral-states-experiments-19)
3. [Structure as the Control Variable (Experiments 10–13)](#3-structure-as-the-control-variable-experiments-1013)
4. [Validating the Architecture (Experiments 14–15)](#4-validating-the-architecture-experiments-1415)
5. [The Primitive Layer: Finding the Atoms of Meaning (Experiments 16–18)](#5-the-primitive-layer-finding-the-atoms-of-meaning-experiments-1618)
6. [Discussion](#6-discussion)
7. [Open Questions](#7-open-questions)
8. [Conclusion](#8-conclusion)
9. [Acknowledgments](#9-acknowledgments)
10. [References](#references)

---

## 1. Introduction

The SRM is not a single model. It is an **architecture** — a pipeline in which:

1. A **symbolic reasoning layer** generates structured representations
2. A **membrane layer** translates those representations into language
3. A **language model** renders the final output

The central claim of this paper is that this architecture is not merely plausible. It is empirically grounded. The experiments described here provide evidence for three findings that surprised us:

> **Finding 1:** Language model behavior under unusual inputs follows a reproducible taxonomy of states — not random noise, but classifiable patterns we term **Collapse, Synthesis, Overflow, Metacognition, Linguistic Drift**, and others.

> **Finding 2:** Structure is a more reliable control variable than content. Telling a model *how* to structure its reasoning produces more consistent outputs than telling it *what* to reason about.

> **Finding 3:** The semantic primitive framework proposed by Wierzbicka — a set of universal concepts that underlie all human languages — appears as measurable activation patterns in small language models across four different architectures. Some primitives activate the membrane phenomenologically. Others function as pure operators. This distinction is consistent, reproducible, and architecture-independent.

We do not claim to have built the SRM. We claim to have found the theoretical foundation on which it can be built. The experiments are reproducible, the code is open source, and the findings are preliminary by design. What follows is an honest account of what we looked for, what we found, and what we don't yet understand.

---

## 2. The Membrane: Discovering Behavioral States (Experiments 1–9)

The first question was almost naive: what happens when you feed a small language model inputs that have no meaning?

We started with **phonemes** — fragments like `bri`, `thal`, `morn`, `vel` — chosen specifically because they carry no semantic content. No definitions, no connotations, no prior context. We fed ten random phonemes at a time to a 500 million parameter model running locally via Ollama, and asked it a single question: find whatever meaning emerges.

We expected noise. We got something else entirely.

> **The model produced Chinese.** On the first run: 失望, meaning disappointment. On the second: 惊讶, meaning surprise — a single word, confidence 1.0. On the third: *The fisherman asked the woman about the fish.* A complete sentence. A narrative. From ten random phoneme fragments with no semantic content.

This was not a failure. This was the first finding.

The model was not predicting likely next tokens in the conventional sense. It was doing something that looked more like **reaching** — searching for a pattern, finding the nearest meaningful configuration, and rendering it in whatever language felt most natural for that configuration. When the phonemes triggered something that felt more naturally expressed in Mandarin than English, the model used Mandarin. The output language was not controlled by the input language. It was controlled by something else — something closer to the semantic weight of the pattern itself.

### The Behavioral Taxonomy

We ran more experiments. We replaced phonemes with real English words. Then with emotionally charged words. Then with function words — *the*, *and*, *of*, *in* — stripped of all content. Then with invented morphemes designed to feel harsh or neutral. Eighteen configurations across nine experiments, each one probing a different dimension of the model's response space.

What accumulated was a taxonomy. The model's behavior under unusual inputs was not random. It fell into recognizable, reproducible classes:

| State | Description | Frequency |
|-------|-------------|-----------|
| **Synthesis** | The model finds a unified interpretation that connects all inputs into a single coherent concept | 50-60% of runs |
| **Collapse** | The model produces a single word at maximum confidence. Input overwhelm | — |
| **Overflow** | The model begins listing and cannot stop. A repetitive spiral that loses coherence | Most common with high semantic similarity inputs |
| **Metacognition** | The model steps outside the task and describes the experiment itself | Most consistent with pure reference words (pronouns) |
| **Linguistic Drift** | The model exits English unprompted and produces output in another language (usually Mandarin) | 7/16 primitives tested |
| **Pure Structure** | The model perceives grammatical and relational patterns without any semantic content | Produced by function words (prepositions) |

> **Key Insight:** Once you know the classes exist, you can ask a more interesting question: *what controls which class appears?*

Experiment 9 gave us the first strong hint. Pure function words — words with no semantic content but rich relational structure — consistently produced **Pure Structure** responses. The membrane was not responding to meaning. It was responding to **shape**. **Structure, not content, appeared to be the primary control variable.**

That finding drove the next four experiments entirely.

---

## 3. Structure as the Control Variable (Experiments 10–13)

The finding from experiment 9 changed the direction of the research entirely.

If structure — not content, not emotional weight, not word frequency — was the primary control variable for membrane behavior, then the implications for the SRM architecture were significant. A reasoning system that could generate structure deterministically, without relying on language at all, might be able to control membrane output reliably. The membrane would not need to understand the reasoning. It would only need to receive its shape.

### Experiment 10: The Recognition Gradient

We introduced what we called the **Recognition Gradient** — the spectrum from fully recognized inputs to completely novel ones:

- **Function words** → fully recognized (model knows exactly what they are)
- **Invented morphemes** → completely novel (model has never seen them)
- **Latinate roots** (e.g., *malum*, *mortis*, *noxius*) → partially recognized

> **Counterintuitive Finding:** We expected fully recognized inputs to produce the most coherent outputs. The opposite was closer to the truth. Partially recognized inputs — the Latinate roots — produced the highest rate of UNKNOWN classifications. Completely novel inputs forced the membrane into **pure synthesis**.

This was the **Recognition Gradient hypothesis**: tension between known and unknown forces synthesis. A membrane given something it half-knows and something it doesn't know at all must bridge the gap — and that bridging act produces the most genuinely novel outputs.

### Experiment 11: The Voice-Giving Prompt

We shifted the prompt frame rather than the input content. Instead of asking the model to *find meaning*, we asked it to *give them a voice* — to treat the inputs as sounds that wanted to express something.

> **Most important methodological finding:** The membrane's behavior is shaped more by *how it is invited to respond* than by *what it is given*. The input content matters less than the permission structure of the prompt. The membrane is not a passive translator. It is a **role-player**.

### Experiment 12: Single Membrane vs. Pipeline

We compared two architectures:

| Architecture | Full Output Rate | Failure Mode |
|--------------|------------------|--------------|
| Single membrane (3 roles simultaneously) | 85% | Sometimes filled role slots by *echoing* input |
| Pipeline (3 specialized membranes) | 90% | Broke at meaning stage on high-emotional inputs |

> **Key Insight:** The failure modes pointed toward a conclusion: the membrane needs its role baked in at the **system level**, not requested in the user prompt.

### Experiment 13: User-Prompt vs. System-Prompt Conditioning

> **Unambiguous Result:** System-conditioned structure perception produced genuine structural outputs in **9 of 9 runs**. User-stacked role prompting produced the structure role being confused with content.

**Structure prompting works. It works reliably. It works regardless of input content. And it works best when the role is established *before* the input arrives.**

This was the architectural insight that made the SRM coherent as a design rather than a hypothesis. If structure can be deterministically prompted into the membrane through system-level conditioning, then a reasoning layer that generates structure as pure code output has a reliable channel into the language output layer.

---

## 4. Validating the Architecture (Experiments 14–15)

At this point in the research, we had:

- ✅ A taxonomy of membrane behaviors
- ✅ A theory about structure as the control variable
- ✅ A proposed architecture

What we did **not** have was evidence that the architecture produced better outcomes than simply asking a language model to reason directly.

### Experiment 14: Direct Prompting vs. SRM

We constructed **twenty reasoning problems** spanning:

- Logic
- Mathematics
- Spatial reasoning
- Causal inference
- Pattern recognition
- Conditional deduction

We ran each problem twice:

1. **Direct prompt** — simply asking the question
2. **SRM prompt** — extract logical structure → apply reasoning → render answer

| Metric | Direct Prompt | SRM Prompt |
|--------|---------------|------------|
| Correctness | 80% | 70% |
| Visible reasoning steps | 50% | **100%** |

> **The finding that matters for interpretability:** The SRM architecture does not make the model smarter. It makes the model's reasoning **visible**. When the model fails under SRM prompting, you can see exactly where it failed.

### Experiment 15: Larger Model Scale

We ran the same twenty problems against a significantly more capable model:

| Metric | Direct Prompt | SRM Prompt |
|--------|---------------|------------|
| Correctness | 95% | 95% |
| Visible reasoning steps | ~50% | **100%** |

> **Confirmed:** The SRM does not add intelligence — it adds **transparency**. On small models, the additional cognitive overhead costs a small amount of accuracy. On capable models, there is no cost at all. The structure is free.

---

## 5. The Primitive Layer: Finding the Atoms of Meaning (Experiments 16–18)

The question: are there specific, individual words that the membrane responds to with such consistency and clarity that they function as **atomic units of meaning**?

### Theoretical Foundations

Two frameworks guided this search:

1. **Anna Wierzbicka's Natural Semantic Metalanguage** — ~65 irreducible semantic primitives (WANT, KNOW, FEEL, GOOD, BAD, DO, HAPPEN, SOMEONE, TIME, PLACE...)
2. **Cowen & Keltner's 27 Emotions** — Awe, Nostalgia, Craving, Entrancement, Empathic Pain, Relief, Grief, Joy, Fear, Anger...

### Experiment 16: Semantic Density

We took sixty seed words and generated 150 two-word combinations, running each three times.

> **Result:** 25% of combinations were dense (38 pairs produced consistent outputs with confidence > 0.85).

The most consistent anchor words: *threshold, yearning, bloom, flow, structure, conflict, decay, stillness*

> **Insight:** These words looked less like arbitrary vocabulary and more like **Layer 2 concepts** — things that emerge from combinations of simpler primitives.

### Experiment 17: Wierzbicka's Primitives

We selected ten semantic primitives and ten emotional primitives, fed each individually to the membrane five times.

| Result | Primitives |
|--------|------------|
| **Stable responses** | WANT, HAPPEN, ANGER, RELIEF, JOY, NOSTALGIA (6) |
| **Semi-stable/unstable** | 14 |
| **Most unstable** | BAD (could not process pure negativity without a referent) |

### Experiment 17b: Layer 0a vs. Layer 0b

**Hypothesis:** Some primitives function as **scaffolding** (organize and relate meaning but carry no phenomenological content), while others function as **content seeds** (carry felt, experiential meaning).

> **Result:** A gap of **+0.313** between the two sublayers.

| Layer | Type | Example Primitives | Activation Score |
|-------|------|-------------------|------------------|
| **Layer 0a** | Scaffolding | SOMEONE, TIME, PLACE | 0.342 |
| **Layer 0b** | Content seeds | JOY, FEAR | 0.654 |

> - SOMEONE, TIME, PLACE scored **0.00** activation — perfectly abstract
> - JOY, FEAR scored **1.00** — maximum phenomenological content
> - GRIEF produced **Mandarin** as modal output — the primitive exists below English

### Experiment 17c: Cross-Architecture Validation

We ran the identical experiment on **four models**:

| Model | Parameters | Gap |
|-------|------------|-----|
| SmolLM2 | 360M | +0.356 |
| Qwen | 500M | +0.313 |
| LLaMA 3.2 | 1B | +0.188 |
| Gemma 3 | 1B | +0.125 |

> **Scaling Hypothesis:** The gap is largest in the smallest model and narrows as model size increases. Larger models appear to **converge toward a more coherent internal representation of the primitive layer**.

### Experiment 18: Primitive Composition

We selected:
- **Operators (Layer 0a):** KNOW, PLACE, FEEL, WANT, TIME
- **Content seeds (Layer 0b):** GRIEF, FEAR, JOY, ANGER, RELIEF, NOSTALGIA

We pre-registered **30 predictions** and ran all combinations across all four models.

> **Result:** **11 combinations** matched predictions in 3/4 models — we term these **majority combinations**:

| Combination | Output |
|-------------|--------|
| KNOW + FEAR | dread or awareness |
| FEEL + GRIEF | heartbreak or sorrow |
| WANT + GRIEF | longing or yearning |
| WANT + FEAR | anxiety |
| WANT + ANGER | ambition |
| TIME + GRIEF | mourning or melancholy |
| TIME + RELIEF | healing or recovery |
| TIME + NOSTALGIA | memory or reminiscence |

> **Special Finding:** JOY resisted composition entirely — it absorbed whatever it was combined with and returned itself. Joy may be the most **self-contained emotional state** in the human repertoire.

### Key Findings (16–18)

1. **Semantic density exists** — some word combinations produce stable, consistent membrane responses
2. **The Layer 0a/0b distinction is architecture-independent** — consistent across four models from four organizations
3. **Primitive composition produces predictable Layer 1 concepts** — the compositional rules appear discoverable

---

## 6. Discussion

### On the Behavioral Taxonomy

The six membrane behavioral classes are empirically grounded. The most natural interpretation is that these classes reflect different modes of the model's **internal search process**:

- **Synthesis** → stable attractor found
- **Overflow** → inputs too similar to distinguish, model loops
- **Metacognition** → self-referential loop triggered
- **Linguistic Drift** → relevant semantic region more densely populated in another language

### On Structure as the Control Variable

> **Isn't this just prompt engineering?**

In one sense, yes. But what distinguishes the SRM approach is the **separation of structure generation from language generation**. In chain-of-thought prompting, the same model generates both reasoning and answer. In the SRM architecture, structure is generated by a **separate layer** and passed to the language model as pre-formed input.

This separation matters for interpretability. When structure comes from outside, it can be inspected, modified, and validated independently.

### On the Primitive Layer Findings

The connection between Wierzbicka's primitives and measurable activation patterns in language models is the finding we are most cautious about — and the one we find most theoretically significant.

> The Layer 0a/0b activation gap appeared consistently across **four models from four different organizations**. If it were an artifact of a specific training distribution, we would expect it to disappear or reverse. It did not.

---

## 7. Open Questions

### The Composition Map is Incomplete

We tested 30 combinations out of 390 possible. The immediate next step is systematic: run all combinations of Layer 0b content primitives, build the full two-primitive composition table, identify which combinations are universal, majority, or model-specific.

### The PLACE Problem

PLACE performed poorly as an operator (21% match rate). Wierzbicka's richer spatial vocabulary (HERE, WHERE, ABOVE, BELOW, FAR, NEAR, SIDE, INSIDE) might perform better.

### Minimum Viable Membrane

SmolLM2 (360M) could recognize primitives but not compose them. Qwen (500M) could compose. Finding that threshold precisely matters for the SRM architecture.

### The Mechanistic Question

Everything in this research is behavioral. We did not look inside the models. Mechanistic interpretability research has the tools to answer *why* the findings hold.

### The Symbolic Layer Question

We have not built the system that generates primitive combinations from pure code, feeds them to the membrane, and produces complete reasoning output. The GOG research program is one candidate for this symbolic layer.

### The Interpretability Application

If you can map any model output back to the primitive combinations that generated it, you have a **human-readable explanation** of what the model was representing — interpretability at the output level rather than the weight level.

---

## 8. Conclusion

We set out to ask whether language could be distilled into symbols. Eighteen experiments later, we have a more precise version of that question and the beginning of an answer.

> **Language cannot be distilled into symbols by brute force** — the combinatorial explosion is too severe, the semantic space too vast. But it **may be distillable into primitives** — a finite set of atomic concepts that underlie all human meaning-making.

The SRM is an architecture built on that possibility:

- **Structure** comes from reasoning and can be deterministically generated
- **Language** is emergent and does not need to be controlled
- The **membrane** carries structure into language space and lets emergence do the rest

The primitive layer findings are the deepest result:

1. The Layer 0a/0b distinction appeared consistently across **four architectures**
2. The **eleven validated Layer 1 compositions** emerged from pre-registered predictions
3. The **scaling observation** — gap narrows as models grow — suggests convergence toward the primitive layer

None of this is finished. But foundations do not need to be finished to be load-bearing.

> **The atoms of meaning are there. We have found several of them. The rest are waiting.**

---

## 9. Acknowledgments

This research was conducted by a single human researcher working in collaboration with **Claude**, an AI assistant developed by Anthropic. The experimental design, theoretical framework, core insights, and research direction were conceived and driven by the human researcher. Claude contributed to experimental implementation, code generation, data interpretation, and the writing of this paper.

We treat this collaboration as analogous to a researcher working with a highly capable instrument — one that can write code, suggest experimental variations, and articulate findings, but cannot have the insight that language might compress into primitives, cannot notice that the membrane is reaching for something, and cannot decide that Wierzbicka's 1970s linguistic theory might map onto the internal geometry of modern neural networks.

The code for all eighteen experiments is open source. The data is available. The methodology is documented. Every open question described in this paper is tractable with the tools already established.

---

## References

1. Wierzbicka, A. (1996). *Semantics: Primes and Universals*. Oxford University Press.

2. Wierzbicka, A. (1972). *Semantic Primitives*. Athenäum.

3. Cowen, A. S., & Keltner, D. (2017). Self-report captures 27 distinct categories of emotion bridged by continuous gradients. *Proceedings of the National Academy of Sciences*, 114(38), E7900–E7909.

4. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35.

5. Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901.

6. Chisholm, Derek R. (2026). *Graph-Oriented Generation (GOG): Offloading AI Reasoning to Deterministic Symbolic Graphs*. https://github.com/dchisholm125/graph-oriented-generation

---

*This paper represents Phase 2 of an ongoing research program into symbolic reasoning and language model architectures.*
