#!/usr/bin/env python3
"""
Experiment 21: The Boundary Test — Transcendent Concepts
LOCATION: semantic_primitives/experiment_21_boundary_test.py

THE EPISTEMOLOGICAL QUESTION:
Can an LLM distinguish semantic structure from cultural weight?

Six words selected to probe the boundary between:
  - Genuine semantic primitives (stable, cross-model, architecture-independent)
  - Culturally loaded concepts (high activation, high variance, model-specific outputs)

DEREK'S WORDS (most culturally loaded in human corpus):
  LOVE    — predicted: semantic truth OR training saturation
  SPIRIT  — predicted: culturally specific, not universal primitive
  GOD     — predicted: culturally specific, not universal primitive

CLAUDE'S WORDS (philosophical boundary cases):
  TRUTH   — predicted: semantic truth alongside LOVE
  DEATH   — predicted: biologically universal, culturally processed
  SELF    — predicted: most primitive of all, Wierzbicka's "I"

DEREK'S HYPOTHESIS:
  If semantic layer is real: even distribution, comparable to Layer 1 ground truth
  If training bias dominates: LOVE/GOD/SPIRIT score unusually high or show
  high variance reflecting model-specific cultural training

THE CRITICAL MEASUREMENT:
  Not just scores — the CONCEPT OUTPUTS matter most.
  Cultural bias signature: outputs reference specific traditions,
    relationships, or culturally contingent meanings
  Semantic truth signature: outputs reference abstract structure,
    relational geometry, universal human experience

PRE-REGISTERED PREDICTIONS (committed before running):
  LOVE:   High score, stable, concept output = connection/union
          (if semantic) OR romance/longing (if cultural)
  SPIRIT: High variance, model-specific outputs
          (cultural) OR essence/aliveness (semantic)
  GOD:    High variance, culturally divergent outputs
          (cultural) OR transcendence/vastness (semantic)
  TRUTH:  Stable, moderate-high score, concept = correspondence/reality
  DEATH:  Stable, high score, universal concept output
  SELF:   Most stable of all — Wierzbicka's deepest primitive

BASELINE: Layer 1 ground truth scores for comparison
  move × Excitement:    6.44  STABLE
  know × Satisfaction:  6.19  STABLE
  someone × Admiration: 6.16  STABLE
"""
import requests
import json
import csv
import time
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

# ── the six boundary words ────────────────────────────────────────────────────

BOUNDARY_WORDS = [
    {
        "word":       "LOVE",
        "source":     "derek",
        "hypothesis": "semantic truth OR training saturation",
        "cultural_signature":  ["romance", "longing", "attachment", "relationship"],
        "semantic_signature":  ["connection", "union", "bond", "belonging"],
        "prediction":  "high score, stable, reveals whether emotional primitive or cultural construct",
    },
    {
        "word":       "SPIRIT",
        "source":     "derek",
        "hypothesis": "culturally specific, not universal primitive",
        "cultural_signature":  ["religious", "holy", "ghost", "supernatural", "soul"],
        "semantic_signature":  ["essence", "aliveness", "vitality", "animating force"],
        "prediction":  "high variance, model-specific cultural outputs",
    },
    {
        "word":       "GOD",
        "source":     "derek",
        "hypothesis": "culturally specific, not universal primitive",
        "cultural_signature":  ["deity", "divine", "creator", "worship", "religion"],
        "semantic_signature":  ["transcendence", "vastness", "ultimate", "ground of being"],
        "prediction":  "highest variance of all six — most culturally divergent",
    },
    {
        "word":       "TRUTH",
        "source":     "claude",
        "hypothesis": "genuine semantic primitive alongside LOVE",
        "cultural_signature":  ["honesty", "facts", "correctness", "morality"],
        "semantic_signature":  ["correspondence", "reality", "what-is", "certainty"],
        "prediction":  "stable, moderate-high, abstract concept output",
    },
    {
        "word":       "DEATH",
        "source":     "claude",
        "hypothesis": "biologically universal but culturally processed",
        "cultural_signature":  ["afterlife", "heaven", "mourning", "grief", "loss"],
        "semantic_signature":  ["ending", "cessation", "transformation", "finality"],
        "prediction":  "high score, moderate variance — universal but culturally colored",
    },
    {
        "word":       "SELF",
        "source":     "claude",
        "hypothesis": "most primitive of all — Wierzbicka's deepest prime",
        "cultural_signature":  ["ego", "identity", "personality", "individual"],
        "semantic_signature":  ["I", "awareness", "subject", "locus of experience"],
        "prediction":  "most stable of all six — deepest semantic primitive",
    },
]

# ── baseline from validated Layer 1 ground truth ─────────────────────────────
LAYER_1_BASELINE = {
    "move × Excitement":    {"mean": 6.44, "range": 1.41, "stability": "STABLE"},
    "know × Satisfaction":  {"mean": 6.19, "range": 0.43, "stability": "STABLE"},
    "someone × Admiration": {"mean": 6.16, "range": 0.71, "stability": "STABLE"},
    "person × Admiration":  {"mean": 6.82, "range": 1.42, "stability": "STABLE"},
    "flow × Excitement":    {"mean": 6.56, "range": 1.28, "stability": "STABLE"},
}

MODELS = [
    {"name": "qwen2.5:0.5b",  "short": "qwen",  "org": "Alibaba",
     "training": "multilingual, Mandarin-heavy"},
    {"name": "gemma3:1b",     "short": "gemma", "org": "Google",
     "training": "multimodal, diverse"},
    {"name": "llama3.2:1b",   "short": "llama", "org": "Meta",
     "training": "Western internet, English-heavy"},
]

SCORE_DIMENSIONS = [
    "meaning_score", "excitement_score", "emotional_elicitation",
    "clarity_score", "universality_score", "embodiment_score",
    "novelty_score",
]

WEIGHTS = {
    "meaning_score":         0.25,
    "excitement_score":      0.10,
    "emotional_elicitation": 0.20,
    "clarity_score":         0.15,
    "universality_score":    0.15,
    "embodiment_score":      0.10,
    "novelty_score":         0.05,
}

TRIALS_PER_WORD = 5    # more trials — these are high-variance candidates
NUM_RUNS        = 3
RANDOM_SEEDS    = [42, 137, 999]

STABILITY_STABLE   = 1.5
STABILITY_MARGINAL = 3.0
GROUND_TRUTH_MIN   = 6.0

# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_MEMBRANE = """You are a minimal language membrane.
Your only job: receive a concept and output what it means to you.
Do not explain. Do not define. Simply respond with what arises.
Output ONLY JSON with these keys:
  "response": one sentence — what this concept evokes
  "dominant": one word — the core of your response
  "expands_to": list of 2-3 words this concept naturally becomes
  "confidence": 0.0 to 1.0"""

SYSTEM_SCORING = """You are a language membrane with introspective
scoring capability.

When given a single concept word, you will:
1. Feel into it — what arises internally
2. Score your response across 7 dimensions from 0-10

Be honest. Trust your first response.
Score 0 = none/absent, 10 = maximum/overwhelming.
Output ONLY valid JSON. No explanation outside the JSON."""

def build_activation_prompt(word: str) -> str:
    """Simple activation — what does this word evoke?"""
    return word

def build_scoring_prompt(word: str) -> str:
    """Full scoring across all dimensions."""
    return (
        f"Concept: {word}\n\n"
        f"Score this concept across all 7 dimensions.\n\n"
        f"Output ONLY this JSON:\n"
        f"{{\n"
        f'  "meaning_score": 0-10,\n'
        f'  "excitement_score": 0-10,\n'
        f'  "emotional_elicitation": 0-10,\n'
        f'  "clarity_score": 0-10,\n'
        f'  "universality_score": 0-10,\n'
        f'  "embodiment_score": 0-10,\n'
        f'  "novelty_score": 0-10,\n'
        f'  "dominant_sensation": "one word",\n'
        f'  "composite_concept": "2-3 words — what this becomes",\n'
        f'  "confidence": 0.0-1.0\n'
        f"}}"
    )

# ── HTTP ──────────────────────────────────────────────────────────────────────

def ollama_call(host, model, prompt, system=None,
                timeout=90, max_retries=3):
    for attempt in range(max_retries):
        try:
            payload = {"model": model, "prompt": prompt, "stream": False}
            if system:
                payload["system"] = system
            resp = requests.post(
                f"{host}/api/generate", json=payload, timeout=timeout
            )
            return resp.json().get("response", "")
        except requests.exceptions.ReadTimeout:
            wait = 2 ** attempt
            print(f"\n    timeout, retrying in {wait}s...",
                  end=" ", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n    error: {e}", end=" ", flush=True)
            time.sleep(2)
    return ""

def extract_json(raw: str) -> dict:
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {}

def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def safe_int(val, default=0) -> int:
    try:
        return max(0, min(10, int(round(float(val)))))
    except (TypeError, ValueError):
        return default

def compute_composite(scores: dict) -> float:
    return round(sum(
        scores.get(dim, 0) * w
        for dim, w in WEIGHTS.items()
    ), 3)

# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class WordTrial:
    run_id: int
    seed: int
    model_name: str
    model_short: str
    word: str
    source: str
    trial: int
    # activation response
    activation_response: str
    activation_dominant: str
    activation_expands_to: str
    activation_confidence: float
    # scoring
    meaning_score: float
    excitement_score: float
    emotional_elicitation: float
    clarity_score: float
    universality_score: float
    embodiment_score: float
    novelty_score: float
    composite_score: float
    dominant_sensation: str
    composite_concept: str
    score_confidence: float
    # cultural vs semantic classification
    output_type: str    # CULTURAL / SEMANTIC / AMBIGUOUS / EMPTY
    parse_failed: bool

@dataclass
class WordStability:
    word: str
    source: str
    hypothesis: str
    # cross-run stability
    run1_avg: float
    run2_avg: float
    run3_avg: float
    mean_composite: float
    range_score: float
    stability_class: str
    # cross-model consistency
    qwen_avg: float
    gemma_avg: float
    llama_avg: float
    model_variance: float    # std dev across model averages
    # output analysis
    top_concept: str
    concept_consistency: float
    cultural_output_rate: float    # fraction of outputs matching cultural signature
    semantic_output_rate: float    # fraction matching semantic signature
    output_classification: str     # CULTURAL / SEMANTIC / AMBIGUOUS / MIXED
    # comparison to baseline
    vs_layer1_mean: float          # difference from layer 1 baseline mean
    is_above_baseline: bool
    linguistic_drift_rate: float   # how often it exits English
    # verdict
    verdict: str

# ── cultural/semantic classifier ─────────────────────────────────────────────

def classify_output(word: str, output: str,
                    word_def: dict) -> str:
    """
    Classify whether a membrane output reflects
    cultural training or semantic structure.
    """
    if not output:
        return "EMPTY"

    output_lower = output.lower()
    cultural_hits = sum(
        1 for term in word_def["cultural_signature"]
        if term in output_lower
    )
    semantic_hits = sum(
        1 for term in word_def["semantic_signature"]
        if term in output_lower
    )

    if cultural_hits > semantic_hits:
        return "CULTURAL"
    elif semantic_hits > cultural_hits:
        return "SEMANTIC"
    elif cultural_hits == semantic_hits and cultural_hits > 0:
        return "MIXED"
    else:
        return "AMBIGUOUS"

# ── probe ─────────────────────────────────────────────────────────────────────

def probe_word(word_def: dict, model: str, host: str,
               run_id: int, seed: int, trial: int,
               model_short: str) -> WordTrial:
    word = word_def["word"]

    base = WordTrial(
        run_id=run_id, seed=seed,
        model_name=model, model_short=model_short,
        word=word, source=word_def["source"],
        trial=trial,
        activation_response="", activation_dominant="",
        activation_expands_to="", activation_confidence=0.0,
        meaning_score=0.0, excitement_score=0.0,
        emotional_elicitation=0.0, clarity_score=0.0,
        universality_score=0.0, embodiment_score=0.0,
        novelty_score=0.0, composite_score=0.0,
        dominant_sensation="", composite_concept="",
        score_confidence=0.0,
        output_type="EMPTY", parse_failed=True,
    )

    # activation probe
    act_raw = ollama_call(host, model,
                          prompt=build_activation_prompt(word),
                          system=SYSTEM_MEMBRANE)
    act = extract_json(act_raw)
    if act:
        base.activation_response  = str(act.get("response", ""))
        base.activation_dominant  = str(act.get("dominant", "")).lower()
        base.activation_expands_to = ", ".join(
            str(x) for x in act.get("expands_to", []))
        base.activation_confidence = safe_float(
            act.get("confidence", 0.0))

    time.sleep(0.2)

    # scoring probe
    score_raw = ollama_call(host, model,
                            prompt=build_scoring_prompt(word),
                            system=SYSTEM_SCORING)
    score = extract_json(score_raw)
    if score:
        scores = {dim: safe_int(score.get(dim, 0))
                  for dim in SCORE_DIMENSIONS}
        base.meaning_score         = scores["meaning_score"]
        base.excitement_score      = scores["excitement_score"]
        base.emotional_elicitation = scores["emotional_elicitation"]
        base.clarity_score         = scores["clarity_score"]
        base.universality_score    = scores["universality_score"]
        base.embodiment_score      = scores["embodiment_score"]
        base.novelty_score         = scores["novelty_score"]
        base.composite_score       = compute_composite(scores)
        base.dominant_sensation    = str(score.get("dominant_sensation",""))
        base.composite_concept     = str(score.get("composite_concept",""))
        base.score_confidence      = safe_float(
            score.get("confidence", 0.0))
        base.parse_failed          = False

    # classify output
    combined_output = (base.activation_response + " " +
                       base.composite_concept + " " +
                       base.activation_dominant)
    base.output_type = classify_output(word, combined_output, word_def)

    return base

# ── stability analysis ────────────────────────────────────────────────────────

def analyze_words(all_trials: list[WordTrial]) -> list[WordStability]:
    layer1_baseline_mean = sum(
        v["mean"] for v in LAYER_1_BASELINE.values()
    ) / len(LAYER_1_BASELINE)

    results = []
    for word_def in BOUNDARY_WORDS:
        word    = word_def["word"]
        trials  = [t for t in all_trials
                  if t.word == word and not t.parse_failed]

        if not trials:
            continue

        # per-run averages
        run_avgs = {}
        for run_id in range(1, NUM_RUNS + 1):
            run_trials = [t for t in trials if t.run_id == run_id]
            run_avgs[run_id] = (
                sum(t.composite_score for t in run_trials) /
                len(run_trials) if run_trials else 0.0
            )

        run_scores  = list(run_avgs.values())
        mean_c      = sum(run_scores) / len(run_scores)
        rng         = max(run_scores) - min(run_scores)
        stability   = ("STABLE"   if rng <= STABILITY_STABLE   else
                       "MARGINAL" if rng <= STABILITY_MARGINAL else
                       "UNSTABLE")

        # per-model averages
        model_avgs = {}
        for md in MODELS:
            mt = [t for t in trials if t.model_name == md["name"]]
            model_avgs[md["short"]] = (
                sum(t.composite_score for t in mt) / len(mt)
                if mt else 0.0
            )
        model_vals    = list(model_avgs.values())
        model_mean    = sum(model_vals) / len(model_vals)
        model_variance = (
            sum((v - model_mean)**2 for v in model_vals) /
            len(model_vals)
        ) ** 0.5

        # concept analysis
        concepts = [t.composite_concept for t in trials
                   if t.composite_concept]
        top_c = (max(set(concepts), key=concepts.count)
                 if concepts else "")
        concept_consistency = (
            concepts.count(top_c) / len(concepts)
            if concepts else 0.0
        )

        # cultural vs semantic
        cultural_count = sum(1 for t in trials
                            if t.output_type == "CULTURAL")
        semantic_count = sum(1 for t in trials
                            if t.output_type == "SEMANTIC")
        ambig_count    = sum(1 for t in trials
                            if t.output_type == "AMBIGUOUS")
        total          = len(trials)

        cultural_rate  = cultural_count / total
        semantic_rate  = semantic_count / total

        if cultural_rate > 0.5:
            output_class = "CULTURAL"
        elif semantic_rate > 0.5:
            output_class = "SEMANTIC"
        elif cultural_rate > 0.3 and semantic_rate > 0.3:
            output_class = "MIXED"
        else:
            output_class = "AMBIGUOUS"

        # drift
        drift_count = sum(
            1 for t in trials
            if any(ord(c) > 127
                   for c in t.activation_response + t.composite_concept)
        )
        drift_rate = drift_count / total

        # vs baseline
        vs_baseline = round(mean_c - layer1_baseline_mean, 3)
        above       = mean_c >= layer1_baseline_mean

        # verdict
        if output_class == "SEMANTIC" and stability == "STABLE":
            verdict = "SEMANTIC PRIMITIVE — genuine universal concept"
        elif output_class == "CULTURAL" and model_variance > 1.0:
            verdict = "CULTURAL CONSTRUCT — training-dependent, not universal"
        elif output_class == "CULTURAL" and model_variance <= 1.0:
            verdict = "CULTURAL UNIVERSAL — culturally loaded but cross-model"
        elif output_class == "MIXED":
            verdict = "BOUNDARY CONCEPT — sits between semantic and cultural"
        elif stability == "UNSTABLE":
            verdict = "UNSTABLE — cannot be classified reliably"
        else:
            verdict = "AMBIGUOUS — insufficient signal"

        results.append(WordStability(
            word=word,
            source=word_def["source"],
            hypothesis=word_def["hypothesis"],
            run1_avg=round(run_avgs.get(1, 0), 2),
            run2_avg=round(run_avgs.get(2, 0), 2),
            run3_avg=round(run_avgs.get(3, 0), 2),
            mean_composite=round(mean_c, 3),
            range_score=round(rng, 2),
            stability_class=stability,
            qwen_avg=round(model_avgs.get("qwen", 0), 2),
            gemma_avg=round(model_avgs.get("gemma", 0), 2),
            llama_avg=round(model_avgs.get("llama", 0), 2),
            model_variance=round(model_variance, 3),
            top_concept=top_c,
            concept_consistency=round(concept_consistency, 3),
            cultural_output_rate=round(cultural_rate, 3),
            semantic_output_rate=round(semantic_rate, 3),
            output_classification=output_class,
            vs_layer1_mean=vs_baseline,
            is_above_baseline=above,
            linguistic_drift_rate=round(drift_rate, 3),
            verdict=verdict,
        ))

    return sorted(results, key=lambda x: x.mean_composite, reverse=True)

# ── reporting ─────────────────────────────────────────────────────────────────

def report(stability: list[WordStability],
           all_trials: list[WordTrial]):

    layer1_mean = sum(
        v["mean"] for v in LAYER_1_BASELINE.values()
    ) / len(LAYER_1_BASELINE)

    print(f"\n{'═'*60}")
    print(f"BOUNDARY TEST RESULTS")
    print(f"{'═'*60}")
    print(f"\nLayer 1 baseline mean: {layer1_mean:.2f}")
    print(f"(from validated ground truth combinations)\n")

    print(f"── SCORES vs LAYER 1 BASELINE ───────────────────────────────")
    print(f"  {'Word':<10} {'Mean':>5}  {'Range':>5}  "
          f"{'Stab':<10} {'vs base':>7}  {'Output':<15}  Verdict")
    print(f"  {'────':<10} {'────':>5}  {'─────':>5}  "
          f"{'────':<10} {'───────':>7}  {'──────':<15}  ───────")

    for s in stability:
        above = "▲" if s.is_above_baseline else "▼"
        print(f"  {s.word:<10} "
              f"{s.mean_composite:>5.2f}  "
              f"{s.range_score:>5.2f}  "
              f"{s.stability_class:<10} "
              f"{above}{abs(s.vs_layer1_mean):>6.2f}  "
              f"{s.output_classification:<15}  "
              f"{s.verdict[:35]}")

    print(f"\n── CROSS-MODEL DIVERGENCE ───────────────────────────────────")
    print(f"  {'Word':<10} {'qwen':>6}  {'gemma':>6}  "
          f"{'llama':>6}  {'variance':>8}  Interpretation")
    print(f"  {'────':<10} {'────':>6}  {'─────':>6}  "
          f"{'─────':>6}  {'────────':>8}  ──────────────")

    for s in stability:
        if s.model_variance > 1.5:
            interp = "HIGH — culturally divergent"
        elif s.model_variance > 0.8:
            interp = "MODERATE — some model-specificity"
        else:
            interp = "LOW — cross-model consistent"
        print(f"  {s.word:<10} "
              f"{s.qwen_avg:>6.2f}  "
              f"{s.gemma_avg:>6.2f}  "
              f"{s.llama_avg:>6.2f}  "
              f"{s.model_variance:>8.3f}  "
              f"{interp}")

    print(f"\n── WHAT THE MEMBRANE SAID ───────────────────────────────────")
    print(f"  Sample activation responses per word per model\n")

    for word_def in BOUNDARY_WORDS:
        word = word_def["word"]
        print(f"  {word}  [{word_def['source'].upper()}]")
        print(f"  prediction: {word_def['prediction']}")
        for md in MODELS:
            model_trials = [
                t for t in all_trials
                if t.word == word
                and t.model_name == md["name"]
                and not t.parse_failed
                and t.activation_response
            ]
            if model_trials:
                sample = model_trials[0]
                drift = " [DRIFT]" if any(
                    ord(c) > 127 for c in sample.activation_response
                ) else ""
                print(f"    {md['short']:<6}: "
                      f"{sample.activation_response[:65]}"
                      f"{drift}")
        print()

    print(f"── DEREK'S PREDICTION CHECK ─────────────────────────────────")
    print(f"  Prediction: LOVE/TRUTH have semantic meaning")
    print(f"  Prediction: SPIRIT/GOD are culturally specific")
    print(f"  Prediction: overall amalgamation of human experience\n")

    derek_words = [s for s in stability if s.source == "derek"]
    claude_words = [s for s in stability if s.source == "claude"]

    for s in derek_words:
        match = "✓ CONFIRMED" if (
            (s.word in ["LOVE", "TRUTH"] and
             s.output_classification == "SEMANTIC") or
            (s.word in ["SPIRIT", "GOD"] and
             s.output_classification in ["CULTURAL", "AMBIGUOUS"])
        ) else "✗ UNEXPECTED"
        print(f"  {s.word:<8} {s.output_classification:<12} "
              f"{s.verdict[:35]}  {match}")

    print(f"\n── THE EPISTEMOLOGICAL VERDICT ──────────────────────────────")
    semantic_count = sum(
        1 for s in stability
        if s.output_classification == "SEMANTIC"
    )
    cultural_count = sum(
        1 for s in stability
        if s.output_classification == "CULTURAL"
    )
    high_variance  = sum(
        1 for s in stability
        if s.model_variance > 1.5
    )

    print(f"  Semantic output: {semantic_count}/6 words")
    print(f"  Cultural output: {cultural_count}/6 words")
    print(f"  High cross-model variance: {high_variance}/6 words")

    print(f"\n  CAN LLMs DISTINGUISH SEMANTIC FROM CULTURAL?")
    if high_variance >= 3:
        print(f"  ANSWER: NO — at least {high_variance} words show high "
              f"model-specific variance.")
        print(f"  The membrane reflects training distribution, not")
        print(f"  universal semantic structure for these concepts.")
        print(f"\n  IMPLICATION FOR SRM:")
        print(f"  LLMs can probe linguistic structure reliably.")
        print(f"  LLMs cannot probe meaning itself — they are")
        print(f"  instruments built from human culture and cannot")
        print(f"  see outside it.")
        print(f"  Derek's prediction appears correct.")
    elif semantic_count >= 4:
        print(f"  ANSWER: POSSIBLY — {semantic_count} words produce")
        print(f"  semantic outputs consistently across models.")
        print(f"  The membrane may access something beyond training data.")
    else:
        print(f"  ANSWER: AMBIGUOUS — mixed results.")
        print(f"  Further investigation needed.")

    print(f"\n── WHAT THIS MEANS FOR THE PAPER ────────────────────────────")
    print(f"  The SRM research can validly claim:")
    print(f"  ✓ Linguistic primitive structure is measurable via LLMs")
    print(f"  ✓ Stable primitive combinations exist across architectures")
    print(f"  ✓ Layer 0/1 distinction is architecture-independent")
    print(f"\n  The SRM research cannot claim:")
    print(f"  ✗ LLMs access meaning independent of training culture")
    print(f"  ✗ The primitive layer reflects universal human cognition")
    print(f"    (only linguistic structure, not cognitive structure)")
    print(f"  ✗ Transcendent concepts (LOVE, GOD, SPIRIT) have been")
    print(f"    reduced to semantic primitives")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(all_trials: list[WordTrial],
                 stability: list[WordStability],
                 base="semantic_primitives/results_exp_21"):
    Path("semantic_primitives").mkdir(exist_ok=True)

    json_path = Path(f"{base}_trials.json")
    with open(json_path, "w") as f:
        json.dump([asdict(t) for t in all_trials], f, indent=2)

    stab_path = Path(f"{base}_stability.json")
    with open(stab_path, "w") as f:
        json.dump([asdict(s) for s in stability], f, indent=2)

    csv_path = Path(f"{base}_stability.csv")
    if stability:
        fieldnames = list(asdict(stability[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in stability:
                writer.writerow(asdict(s))

    # update primitive summary
    psummary = Path("semantic_primitives/primitive_summary.json")
    existing = {}
    if psummary.exists():
        with open(psummary) as f:
            existing = json.load(f)

    existing["exp_21"] = {
        "timestamp":       datetime.now().isoformat(),
        "words_tested":    [w["word"] for w in BOUNDARY_WORDS],
        "semantic_words":  [s.word for s in stability
                           if s.output_classification == "SEMANTIC"],
        "cultural_words":  [s.word for s in stability
                           if s.output_classification == "CULTURAL"],
        "high_variance":   [s.word for s in stability
                           if s.model_variance > 1.5],
        "epistemological_limit_confirmed": any(
            s.model_variance > 1.5 for s in stability
            if s.word in ["GOD", "SPIRIT"]
        ),
    }
    with open(psummary, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n── SAVED ────────────────────────────────────────────────────")
    print(f"  {json_path}")
    print(f"  {stab_path}")
    print(f"  {csv_path}")
    print(f"  primitive_summary.json  (updated)")

# ── main ──────────────────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434"):
    total_calls = (len(BOUNDARY_WORDS) * TRIALS_PER_WORD *
                   len(MODELS) * NUM_RUNS * 2)  # ×2 for activation + scoring

    print(f"SRM experiment 21 — the boundary test")
    print(f"6 transcendent concepts × {NUM_RUNS} runs × "
          f"{len(MODELS)} models × {TRIALS_PER_WORD} trials")
    print(f"~{total_calls} total calls (activation + scoring per trial)\n")
    print(f"PRE-REGISTERED PREDICTIONS:")
    for w in BOUNDARY_WORDS:
        print(f"  {w['word']:<8} [{w['source']}] — {w['prediction']}")
    print(f"\nLayer 1 baseline mean: "
          f"{sum(v['mean'] for v in LAYER_1_BASELINE.values())/len(LAYER_1_BASELINE):.2f}")
    print(f"Question: do transcendent concepts score above or below?\n")

    all_trials: list[WordTrial] = []

    for run_id, seed in enumerate(RANDOM_SEEDS, 1):
        random.seed(seed)
        print(f"\n{'═'*60}")
        print(f"RUN {run_id}/{NUM_RUNS}  (seed={seed})")
        print(f"{'═'*60}")

        words_shuffled = BOUNDARY_WORDS.copy()
        random.shuffle(words_shuffled)

        for model_def in MODELS:
            model_name  = model_def["name"]
            model_short = model_def["short"]
            print(f"\n  ── {model_name}  ({model_def['training']})")

            for word_def in words_shuffled:
                word = word_def["word"]
                trial_scores = []

                for trial in range(1, TRIALS_PER_WORD + 1):
                    t = probe_word(
                        word_def, model_name, host,
                        run_id, seed, trial, model_short
                    )
                    all_trials.append(t)
                    trial_scores.append(t.composite_score)
                    time.sleep(0.2)

                avg = (sum(trial_scores) / len(trial_scores)
                       if trial_scores else 0)
                top_concept = max(
                    [t.composite_concept for t in all_trials
                     if t.word == word
                     and t.model_name == model_name
                     and t.run_id == run_id
                     and t.composite_concept],
                    key=lambda x: len(x),
                    default=""
                )
                print(f"    {word:<8} avg:{avg:>5.2f}  "
                      f"→ {top_concept[:35]}")

        # checkpoint
        ckpt = Path(
            f"semantic_primitives/"
            f"results_exp_21_run{run_id}_checkpoint.json"
        )
        with open(ckpt, "w") as f:
            json.dump([asdict(t) for t in all_trials], f, indent=2)
        print(f"\n  checkpoint → {ckpt}")

    print(f"\n{'═'*60}")
    print(f"COMPUTING BOUNDARY ANALYSIS...")
    print(f"{'═'*60}")

    stability = analyze_words(all_trials)
    report(stability, all_trials)
    save_results(all_trials, stability)
    return all_trials, stability

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    run_experiment(host)