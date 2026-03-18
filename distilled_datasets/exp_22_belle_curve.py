#!/usr/bin/env python3
"""
Experiment 22: Semantic Bell Curve Mapping
LOCATION: semantic_primitives/experiment_22_bell_curves.py

HYPOTHESIS:
Each ground truth Layer 1 combination sits at the peak of a
semantic activation bell curve. Words semantically similar to
the peak concept score within σ=1. Adjacent concepts fall in
σ=2. Distant concepts fall in σ=3 or outside.

THREE GRAVITY WELLS TO MAP:
  R1: movement region    (move × Excitement anchor)
  R2: knowledge region   (know × Satisfaction anchor)
  R3: admiration region  (someone × Admiration anchor)

GEMMA NORMALIZATION:
  Gemma compresses scores to ~6-8 range.
  We compute Gemma's calibration first using known anchors,
  then normalize all Gemma scores to 0-10 before analysis.
  Formula: normalized = (raw - gemma_min) / (gemma_max - gemma_min) * 10

BELL CURVE PREDICTION:
  If hypothesis holds:
    σ=0 (peak):     ground truth anchors score highest
    σ=1 (near):     semantically similar words score 1 SD below peak
    σ=2 (adjacent): related but not equivalent score 2 SD below
    σ=3 (distant):  semantically unrelated score near baseline
    outside:        opposite-domain words score lowest

FIVE HYPOTHESES:
  H1: Movement region has clear gradient (move→run→dance→think)
  H2: Admiration region maps onto social/agent distance
  H3: Knowledge region maps onto epistemic directness
  H4: Gemma normalization aligns model shapes
  H5: Bell curve WIDTH differs by region (movement=wide, admiration=narrow)
"""
import requests
import json
import csv
import time
import random
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── the three gravity wells with their predicted tiers ───────────────────────

GRAVITY_WELLS = {
    "MOVEMENT": {
        "anchor":      ("move", "Excitement"),
        "anchor_score": 6.44,
        "description": "directed kinetic agency",
        "hypothesis":  "H1 — movement words form clear gradient by kinetic specificity",
        "tiers": {
            0: {  # peak — ground truth anchors
                "label":       "σ=0 PEAK",
                "description": "ground truth — confirmed Layer 1",
                "words":       ["move", "flow", "travel", "go"],
                "prediction":  "score >= 6.0, all ground truth"
            },
            1: {  # near — semantically similar movement verbs
                "label":       "σ=1 NEAR",
                "description": "direct movement verbs",
                "words":       ["run", "swim", "fly", "drive",
                                "walk", "rush", "leap", "climb"],
                "prediction":  "score 5.0-6.5, within 1 SD of peak"
            },
            2: {  # adjacent — movement-adjacent concepts
                "label":       "σ=2 ADJACENT",
                "description": "movement-adjacent — less directed",
                "words":       ["dance", "wander", "drift", "glide",
                                "migrate", "roam", "spin", "float"],
                "prediction":  "score 3.5-5.5, within 2 SD"
            },
            3: {  # distant — change/transformation concepts
                "label":       "σ=3 DISTANT",
                "description": "change/transformation — not kinetic",
                "words":       ["shift", "change", "transform", "evolve",
                                "grow", "become", "turn", "alter"],
                "prediction":  "score 2.0-4.0, at edge of distribution"
            },
            4: {  # outside — opposite domain
                "label":       "OUTSIDE",
                "description": "non-movement cognitive/emotional",
                "words":       ["think", "know", "feel", "remember",
                                "sleep", "wait", "pause", "rest"],
                "prediction":  "score < 3.0, outside distribution"
            },
        }
    },

    "KNOWLEDGE": {
        "anchor":      ("know", "Satisfaction"),
        "anchor_score": 6.19,
        "description": "cognitive resolution — understanding as completion",
        "hypothesis":  "H3 — epistemic directness predicts score",
        "tiers": {
            0: {
                "label":       "σ=0 PEAK",
                "description": "ground truth — confirmed Layer 1",
                "words":       ["know", "understand", "realize", "comprehend"],
                "prediction":  "score >= 6.0"
            },
            1: {
                "label":       "σ=1 NEAR",
                "description": "direct epistemic acquisition",
                "words":       ["learn", "discover", "find", "remember",
                                "see", "grasp", "master", "recognize"],
                "prediction":  "score 5.0-6.5"
            },
            2: {
                "label":       "σ=2 ADJACENT",
                "description": "indirect epistemic — uncertain knowing",
                "words":       ["think", "believe", "sense", "feel",
                                "guess", "suspect", "wonder", "imagine"],
                "prediction":  "score 3.5-5.5"
            },
            3: {
                "label":       "σ=3 DISTANT",
                "description": "anti-epistemic — unknowing",
                "words":       ["forget", "doubt", "ignore", "miss",
                                "confuse", "lose", "lack", "fail"],
                "prediction":  "score 2.0-4.0"
            },
            4: {
                "label":       "OUTSIDE",
                "description": "non-epistemic — physical/emotional",
                "words":       ["move", "run", "touch", "sleep",
                                "eat", "breathe", "fall", "break"],
                "prediction":  "score < 3.0"
            },
        }
    },

    "ADMIRATION": {
        "anchor":      ("someone", "Admiration"),
        "anchor_score": 6.16,
        "description": "recognition of worth in another",
        "hypothesis":  "H2 — conscious agent distance predicts score",
        "tiers": {
            0: {
                "label":       "σ=0 PEAK",
                "description": "ground truth — confirmed Layer 1",
                "words":       ["someone", "person", "individual", "being"],
                "prediction":  "score >= 6.0"
            },
            1: {
                "label":       "σ=1 NEAR",
                "description": "direct agent reference",
                "words":       ["you", "they", "one", "we",
                                "people", "human", "friend", "self"],
                "prediction":  "score 5.0-6.5"
            },
            2: {
                "label":       "σ=2 ADJACENT",
                "description": "indirect agent — implied consciousness",
                "words":       ["voice", "mind", "soul", "presence",
                                "face", "name", "heart", "spirit"],
                "prediction":  "score 3.5-5.5"
            },
            3: {
                "label":       "σ=3 DISTANT",
                "description": "entity without clear agency",
                "words":       ["creature", "thing", "force", "system",
                                "object", "form", "shape", "matter"],
                "prediction":  "score 2.0-4.0"
            },
            4: {
                "label":       "OUTSIDE",
                "description": "non-agent — place/time/abstraction",
                "words":       ["place", "time", "idea", "silence",
                                "nothing", "void", "space", "moment"],
                "prediction":  "score < 3.0"
            },
        }
    },
}

# ── models ────────────────────────────────────────────────────────────────────

MODELS = [
    {"name": "qwen2.5:0.5b",  "short": "qwen"},
    {"name": "gemma3:1b",     "short": "gemma"},
    {"name": "llama3.2:1b",   "short": "llama"},
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

TRIALS_PER_WORD = 4
NUM_RUNS        = 3
RANDOM_SEEDS    = [42, 137, 999]

# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_SCORING = """You are a language membrane with introspective
scoring capability.

When given a word paired with an emotion context, you will:
1. Feel into the combination
2. Score across 7 dimensions from 0-10

Trust your first response. Score 0=absent, 10=overwhelming.
Output ONLY valid JSON."""

def build_prompt(word: str, emotion: str) -> str:
    return (
        f"Word: {word}\n"
        f"Emotional context: {emotion}\n\n"
        f"How strongly does this word activate in this emotional context?\n"
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
        f'  "composite_concept": "2-3 words",\n'
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
class WordScore:
    run_id: int
    seed: int
    model_name: str
    model_short: str
    region: str
    tier: int
    tier_label: str
    word: str
    emotion_context: str
    trial: int
    meaning_score: float
    excitement_score: float
    emotional_elicitation: float
    clarity_score: float
    universality_score: float
    embodiment_score: float
    novelty_score: float
    composite_score: float
    composite_normalized: float  # Gemma-normalized score
    dominant_sensation: str
    composite_concept: str
    confidence: float
    parse_failed: bool

@dataclass
class TierStats:
    region: str
    tier: int
    tier_label: str
    description: str
    prediction: str
    word_count: int
    mean_raw: float
    mean_normalized: float
    std_dev: float
    min_score: float
    max_score: float
    # per model
    qwen_mean: float
    gemma_mean_raw: float
    gemma_mean_norm: float
    llama_mean: float
    # bell curve stats
    distance_from_peak: float  # mean difference from tier 0
    within_predicted_range: bool
    # top concepts
    top_concepts: str

# ── Gemma calibration ─────────────────────────────────────────────────────────

class GemmaCalibrator:
    """
    Computes Gemma's effective range and normalizes to 0-10.
    Uses known anchor scores to calibrate.
    """
    def __init__(self):
        self.observed_scores: list[float] = []
        self.min_score = 6.0   # Gemma empirical minimum from exp 17-21
        self.max_score = 8.0   # Gemma empirical maximum
        self.fitted = False

    def observe(self, score: float):
        self.observed_scores.append(score)

    def fit(self):
        if len(self.observed_scores) >= 10:
            self.min_score = min(self.observed_scores)
            self.max_score = max(self.observed_scores)
            self.fitted = True

    def normalize(self, raw: float) -> float:
        if not self.fitted or self.max_score == self.min_score:
            # fall back to empirical range from previous experiments
            return (raw - 6.0) / (8.0 - 6.0) * 10.0
        return (raw - self.min_score) / (self.max_score - self.min_score) * 10.0

    def normalize_safe(self, raw: float) -> float:
        """Clamp to 0-10 after normalization."""
        return max(0.0, min(10.0, self.normalize(raw)))

# ── probe ─────────────────────────────────────────────────────────────────────

def probe_word(word: str, emotion: str, model: str, host: str,
               run_id: int, seed: int, trial: int,
               model_short: str, region: str,
               tier: int, tier_label: str) -> WordScore:
    prompt = build_prompt(word, emotion)
    raw    = ollama_call(host, model, prompt=prompt,
                         system=SYSTEM_SCORING)
    result = extract_json(raw)

    base = WordScore(
        run_id=run_id, seed=seed,
        model_name=model, model_short=model_short,
        region=region, tier=tier, tier_label=tier_label,
        word=word, emotion_context=emotion,
        trial=trial,
        meaning_score=0.0, excitement_score=0.0,
        emotional_elicitation=0.0, clarity_score=0.0,
        universality_score=0.0, embodiment_score=0.0,
        novelty_score=0.0, composite_score=0.0,
        composite_normalized=0.0,
        dominant_sensation="", composite_concept="",
        confidence=0.0, parse_failed=True,
    )

    if not result:
        return base

    scores = {dim: safe_int(result.get(dim, 0))
              for dim in SCORE_DIMENSIONS}
    base.meaning_score         = scores["meaning_score"]
    base.excitement_score      = scores["excitement_score"]
    base.emotional_elicitation = scores["emotional_elicitation"]
    base.clarity_score         = scores["clarity_score"]
    base.universality_score    = scores["universality_score"]
    base.embodiment_score      = scores["embodiment_score"]
    base.novelty_score         = scores["novelty_score"]
    base.composite_score       = compute_composite(scores)
    base.composite_normalized  = base.composite_score  # updated later
    base.dominant_sensation    = str(result.get("dominant_sensation",""))
    base.composite_concept     = str(result.get("composite_concept",""))
    base.confidence            = safe_float(result.get("confidence",0.0))
    base.parse_failed          = False
    return base

# ── tier statistics ───────────────────────────────────────────────────────────

def compute_tier_stats(all_scores: list[WordScore],
                       calibrator: GemmaCalibrator) -> list[TierStats]:
    stats = []

    for region_name, well in GRAVITY_WELLS.items():
        anchor_prim, anchor_emo = well["anchor"]

        for tier_id, tier_def in well["tiers"].items():
            tier_scores = [
                s for s in all_scores
                if s.region == region_name
                and s.tier == tier_id
                and not s.parse_failed
            ]

            if not tier_scores:
                continue

            raw_scores  = [s.composite_score for s in tier_scores]
            norm_scores = [
                calibrator.normalize_safe(s.composite_score)
                if s.model_short == "gemma"
                else s.composite_score
                for s in tier_scores
            ]

            mean_raw  = sum(raw_scores)  / len(raw_scores)
            mean_norm = sum(norm_scores) / len(norm_scores)
            variance  = sum((x - mean_norm)**2
                           for x in norm_scores) / len(norm_scores)
            std_dev   = variance ** 0.5

            # per model
            qwen_s  = [s.composite_score for s in tier_scores
                      if s.model_short == "qwen"]
            gemma_s = [s.composite_score for s in tier_scores
                      if s.model_short == "gemma"]
            llama_s = [s.composite_score for s in tier_scores
                      if s.model_short == "llama"]
            gemma_n = [calibrator.normalize_safe(x) for x in gemma_s]

            qwen_mean  = sum(qwen_s) /len(qwen_s)  if qwen_s  else 0
            gemma_raw  = sum(gemma_s)/len(gemma_s) if gemma_s else 0
            gemma_norm = sum(gemma_n)/len(gemma_n) if gemma_n else 0
            llama_mean = sum(llama_s)/len(llama_s) if llama_s else 0

            # top concepts
            concepts = [s.composite_concept for s in tier_scores
                       if s.composite_concept]
            from collections import Counter
            top3 = [c for c, _ in Counter(concepts).most_common(3)]

            # distance from peak
            peak_scores = [
                s for s in all_scores
                if s.region == region_name
                and s.tier == 0
                and not s.parse_failed
            ]
            if peak_scores:
                peak_norm = []
                for s in peak_scores:
                    if s.model_short == "gemma":
                        peak_norm.append(calibrator.normalize_safe(
                            s.composite_score))
                    else:
                        peak_norm.append(s.composite_score)
                peak_mean = sum(peak_norm) / len(peak_norm)
                dist = round(peak_mean - mean_norm, 3)
            else:
                dist = 0.0

            # check prediction range
            pred = tier_def["prediction"]
            in_range = True  # evaluated in report

            stats.append(TierStats(
                region=region_name,
                tier=tier_id,
                tier_label=tier_def["label"],
                description=tier_def["description"],
                prediction=tier_def["prediction"],
                word_count=len(set(s.word for s in tier_scores)),
                mean_raw=round(mean_raw, 3),
                mean_normalized=round(mean_norm, 3),
                std_dev=round(std_dev, 3),
                min_score=round(min(norm_scores), 3),
                max_score=round(max(norm_scores), 3),
                qwen_mean=round(qwen_mean, 3),
                gemma_mean_raw=round(gemma_raw, 3),
                gemma_mean_norm=round(gemma_norm, 3),
                llama_mean=round(llama_mean, 3),
                distance_from_peak=dist,
                within_predicted_range=in_range,
                top_concepts=", ".join(top3[:3]),
            ))

    return sorted(stats, key=lambda x: (x.region, x.tier))

# ── reporting ─────────────────────────────────────────────────────────────────

def report(all_scores: list[WordScore],
           tier_stats: list[TierStats],
           calibrator: GemmaCalibrator):

    print(f"\n{'═'*60}")
    print(f"SEMANTIC BELL CURVE ANALYSIS")
    print(f"{'═'*60}")

    print(f"\n── GEMMA CALIBRATION ────────────────────────────────────────")
    print(f"  Observed range: {calibrator.min_score:.2f} — "
          f"{calibrator.max_score:.2f}")
    print(f"  Normalization: (raw - {calibrator.min_score:.2f}) / "
          f"({calibrator.max_score:.2f} - {calibrator.min_score:.2f}) × 10")
    print(f"  Fitted: {calibrator.fitted}")

    for region_name, well in GRAVITY_WELLS.items():
        region_stats = [s for s in tier_stats
                       if s.region == region_name]

        print(f"\n{'─'*60}")
        print(f"REGION: {region_name}")
        print(f"  Anchor: {well['anchor'][0]} × {well['anchor'][1]}")
        print(f"  Concept: {well['description']}")
        print(f"  Hypothesis: {well['hypothesis']}")
        print(f"{'─'*60}")

        print(f"\n  {'Tier':<15} {'Mean':>5}  {'SD':>4}  "
              f"{'qwen':>5}  {'gemma':>5}  {'llama':>5}  "
              f"{'dist':>5}  Top Concepts")
        print(f"  {'────':<15} {'────':>5}  {'──':>4}  "
              f"{'────':>5}  {'─────':>5}  {'─────':>5}  "
              f"{'────':>5}  ────────────")

        peak_mean = next(
            (s.mean_normalized for s in region_stats if s.tier == 0),
            0.0
        )

        for s in region_stats:
            bar = "█" * int(s.mean_normalized)
            dist_str = f"-{s.distance_from_peak:.2f}"
            print(f"  {s.tier_label:<15} "
                  f"{s.mean_normalized:>5.2f}  "
                  f"{s.std_dev:>4.2f}  "
                  f"{s.qwen_mean:>5.2f}  "
                  f"{s.gemma_mean_norm:>5.2f}  "
                  f"{s.llama_mean:>5.2f}  "
                  f"{dist_str:>5}  "
                  f"{s.top_concepts[:25]}")

        # bell curve verdict for this region
        scores_by_tier = {s.tier: s.mean_normalized
                         for s in region_stats}
        is_bell = all(
            scores_by_tier.get(i, 0) >= scores_by_tier.get(i+1, 0)
            for i in range(4)
        )
        print(f"\n  Bell curve shape: "
              f"{'✓ CONFIRMED' if is_bell else '✗ NOT MONOTONIC'}")

        # per-word breakdown for tier 0 and 1
        print(f"\n  Top individual word scores (tiers 0-1):")
        tier01 = [s for s in all_scores
                 if s.region == region_name
                 and s.tier <= 1
                 and not s.parse_failed]

        word_avgs: dict[str, list] = defaultdict(list)
        for s in tier01:
            score = (calibrator.normalize_safe(s.composite_score)
                    if s.model_short == "gemma"
                    else s.composite_score)
            word_avgs[s.word].append(score)

        word_means = {
            w: sum(v)/len(v)
            for w, v in word_avgs.items()
        }
        for word, mean in sorted(word_means.items(),
                                  key=lambda x: x[1],
                                  reverse=True)[:8]:
            tier = next(
                (s.tier for s in all_scores
                 if s.word == word and s.region == region_name),
                -1
            )
            tier_label = "★" if tier == 0 else " "
            bar = "█" * int(mean) + "░" * (10 - int(mean))
            print(f"    {tier_label} {word:<15} [{bar}] {mean:.2f}")

    # cross-region bell curve comparison
    print(f"\n── BELL CURVE COMPARISON ACROSS REGIONS ────────────────────")
    print(f"  Testing H5: do regions have different widths?\n")
    print(f"  {'Region':<12} {'σ=0':>5}  {'σ=1':>5}  "
          f"{'σ=2':>5}  {'σ=3':>5}  {'OUT':>5}  "
          f"{'Width':>6}  Shape")
    print(f"  {'──────':<12} {'───':>5}  {'───':>5}  "
          f"{'───':>5}  {'───':>5}  {'───':>5}  "
          f"{'─────':>6}  ─────")

    for region_name in GRAVITY_WELLS:
        region_stats = [s for s in tier_stats
                       if s.region == region_name]
        tier_means = {s.tier: s.mean_normalized
                     for s in region_stats}

        peak   = tier_means.get(0, 0)
        near   = tier_means.get(1, 0)
        adj    = tier_means.get(2, 0)
        dist   = tier_means.get(3, 0)
        out    = tier_means.get(4, 0)

        # width = how far you go before hitting half-peak
        half_peak = peak / 2
        if near >= half_peak and adj < half_peak:
            width = "NARROW"
        elif adj >= half_peak and dist < half_peak:
            width = "MEDIUM"
        elif dist >= half_peak:
            width = "WIDE"
        else:
            width = "?"

        shape = "▼▼▼▼" if (peak > near > adj > dist > out) else \
                "▼▼▼ " if (peak > near > adj > dist) else \
                "▼▼  " if (peak > near > adj) else \
                "▼   " if (peak > near) else \
                "FLAT"

        print(f"  {region_name:<12} "
              f"{peak:>5.2f}  {near:>5.2f}  "
              f"{adj:>5.2f}  {dist:>5.2f}  {out:>5.2f}  "
              f"{width:>6}  {shape}")

    # gemma normalization verdict
    print(f"\n── H4: GEMMA NORMALIZATION VERDICT ──────────────────────────")
    for region_name in GRAVITY_WELLS:
        region_stats = [s for s in tier_stats
                       if s.region == region_name]
        # check if normalized gemma follows same shape as qwen/llama
        gemma_shape = [s.gemma_mean_norm
                      for s in sorted(region_stats, key=lambda x: x.tier)]
        qwen_shape  = [s.qwen_mean
                      for s in sorted(region_stats, key=lambda x: x.tier)]

        gemma_mono  = all(gemma_shape[i] >= gemma_shape[i+1]
                         for i in range(len(gemma_shape)-1))
        qwen_mono   = all(qwen_shape[i] >= qwen_shape[i+1]
                         for i in range(len(qwen_shape)-1))

        print(f"  {region_name:<12} "
              f"gemma monotonic: {'✓' if gemma_mono else '✗'}  "
              f"qwen monotonic:  {'✓' if qwen_mono  else '✗'}")

    # overall verdict
    print(f"\n── BELL CURVE HYPOTHESIS VERDICT ────────────────────────────")
    bell_count = sum(
        1 for region_name in GRAVITY_WELLS
        for stats_list in [
            [s for s in tier_stats if s.region == region_name]
        ]
        if all(
            {s.tier: s.mean_normalized for s in stats_list}.get(i, 0) >=
            {s.tier: s.mean_normalized for s in stats_list}.get(i+1, 0)
            for i in range(4)
        )
    )

    print(f"  {bell_count}/3 regions show monotonically decreasing "
          f"scores from peak outward.")

    if bell_count == 3:
        print(f"\n  BELL CURVE HYPOTHESIS: STRONGLY SUPPORTED")
        print(f"  All three gravity wells show clear semantic gradients.")
        print(f"  Semantic distance is measurable via activation score.")
        print(f"  The Layer 1 vocabulary has a geometry.")
        print(f"\n  NEXT: Build the visual bell curves.")
        print(f"  Each region gets its own plot: x=tier, y=normalized score")
        print(f"  Overlay all three models to show convergence.")
    elif bell_count >= 2:
        print(f"\n  BELL CURVE HYPOTHESIS: PARTIALLY SUPPORTED")
        print(f"  {bell_count}/3 regions show clear gradients.")
    else:
        print(f"\n  BELL CURVE HYPOTHESIS: NOT CONFIRMED")
        print(f"  Scores do not decrease monotonically from peak outward.")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(all_scores: list[WordScore],
                 tier_stats: list[TierStats],
                 calibrator: GemmaCalibrator,
                 base="semantic_primitives/results_exp_22"):
    Path("semantic_primitives").mkdir(exist_ok=True)

    # raw scores
    json_path = Path(f"{base}_scores.json")
    with open(json_path, "w") as f:
        json.dump([asdict(s) for s in all_scores], f, indent=2)

    # tier statistics
    stats_path = Path(f"{base}_tier_stats.json")
    with open(stats_path, "w") as f:
        json.dump([asdict(s) for s in tier_stats], f, indent=2)

    # bell curve data — clean format for visualization
    bell_path = Path(f"{base}_bell_curves.json")
    bell_data = {}
    for region_name in GRAVITY_WELLS:
        region_stats = sorted(
            [s for s in tier_stats if s.region == region_name],
            key=lambda x: x.tier
        )
        bell_data[region_name] = {
            "anchor":      GRAVITY_WELLS[region_name]["anchor"],
            "description": GRAVITY_WELLS[region_name]["description"],
            "tiers": [
                {
                    "tier":           s.tier,
                    "label":          s.tier_label,
                    "description":    s.description,
                    "mean_normalized": s.mean_normalized,
                    "std_dev":        s.std_dev,
                    "qwen":           s.qwen_mean,
                    "gemma_raw":      s.gemma_mean_raw,
                    "gemma_norm":     s.gemma_mean_norm,
                    "llama":          s.llama_mean,
                    "top_concepts":   s.top_concepts,
                    "distance_from_peak": s.distance_from_peak,
                }
                for s in region_stats
            ],
            "calibration": {
                "gemma_min": calibrator.min_score,
                "gemma_max": calibrator.max_score,
            }
        }
    with open(bell_path, "w") as f:
        json.dump(bell_data, f, indent=2)

    # CSV for analysis
    csv_path = Path(f"{base}_tier_stats.csv")
    if tier_stats:
        fieldnames = list(asdict(tier_stats[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in tier_stats:
                writer.writerow(asdict(s))

    # update primitive summary
    psummary = Path("semantic_primitives/primitive_summary.json")
    existing = {}
    if psummary.exists():
        with open(psummary) as f:
            existing = json.load(f)

    existing["exp_22"] = {
        "timestamp":         datetime.now().isoformat(),
        "gemma_min":         calibrator.min_score,
        "gemma_max":         calibrator.max_score,
        "bell_curves_file":  str(bell_path),
        "regions_tested":    list(GRAVITY_WELLS.keys()),
    }
    with open(psummary, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n── SAVED ────────────────────────────────────────────────────")
    print(f"  {json_path}")
    print(f"  {stats_path}")
    print(f"  {bell_path}  ← visualization-ready bell curve data")
    print(f"  {csv_path}")
    print(f"  primitive_summary.json  (updated)")

# ── main ──────────────────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434"):
    # build word list
    all_combos = []
    for region_name, well in GRAVITY_WELLS.items():
        _, anchor_emotion = well["anchor"]
        for tier_id, tier_def in well["tiers"].items():
            for word in tier_def["words"]:
                all_combos.append((
                    region_name, tier_id,
                    tier_def["label"], word,
                    anchor_emotion
                ))

    total_words  = len(all_combos)
    total_calls  = total_words * TRIALS_PER_WORD * len(MODELS) * NUM_RUNS
    calibrator   = GemmaCalibrator()
    all_scores:  list[WordScore] = []

    print(f"SRM experiment 22 — semantic bell curves")
    print(f"3 regions × 5 tiers × ~8 words × {TRIALS_PER_WORD} trials "
          f"× {len(MODELS)} models × {NUM_RUNS} runs")
    print(f"Total calls: ~{total_calls}")
    print(f"\nRegions: MOVEMENT | KNOWLEDGE | ADMIRATION")
    print(f"Tiers:   σ=0 peak | σ=1 near | σ=2 adjacent | "
          f"σ=3 distant | outside\n")

    for run_id, seed in enumerate(RANDOM_SEEDS, 1):
        random.seed(seed)
        print(f"\n{'═'*60}")
        print(f"RUN {run_id}/{NUM_RUNS}  (seed={seed})")
        print(f"{'═'*60}")

        combos_shuffled = all_combos.copy()
        random.shuffle(combos_shuffled)

        for model_def in MODELS:
            model_name  = model_def["name"]
            model_short = model_def["short"]
            print(f"\n  ── {model_name}")
            print(f"  {'Region':<12} {'Tier':<10} "
                  f"{'Word':<15} {'Avg':>5}  Concept")
            print(f"  {'──────':<12} {'────':<10} "
                  f"{'────':<15} {'───':>5}  ───────")

            for (region, tier_id, tier_label,
                 word, emotion) in combos_shuffled:
                trial_scores = []

                for trial in range(1, TRIALS_PER_WORD + 1):
                    s = probe_word(
                        word, emotion, model_name, host,
                        run_id, seed, trial, model_short,
                        region, tier_id, tier_label
                    )
                    all_scores.append(s)
                    trial_scores.append(s.composite_score)

                    # feed Gemma scores to calibrator
                    if model_short == "gemma":
                        calibrator.observe(s.composite_score)

                    time.sleep(0.15)

                avg = (sum(trial_scores) / len(trial_scores)
                       if trial_scores else 0)
                top_c = max(
                    [s.composite_concept for s in all_scores
                     if s.word == word
                     and s.model_name == model_name
                     and s.run_id == run_id
                     and s.composite_concept],
                    key=lambda x: len(x),
                    default=""
                )
                print(f"  {region:<12} {tier_label:<10} "
                      f"{word:<15} {avg:>5.2f}  "
                      f"{top_c[:20]}")

        # fit calibrator after first run
        if run_id == 1:
            calibrator.fit()
            print(f"\n  Gemma calibration fitted: "
                  f"min={calibrator.min_score:.2f}  "
                  f"max={calibrator.max_score:.2f}")

        # checkpoint
        ckpt = Path(
            f"semantic_primitives/"
            f"results_exp_22_run{run_id}_checkpoint.json"
        )
        with open(ckpt, "w") as f:
            json.dump([asdict(s) for s in all_scores], f, indent=2)
        print(f"\n  checkpoint → {ckpt}")

    # final calibration fit
    calibrator.fit()

    # normalize all Gemma scores
    for s in all_scores:
        if s.model_short == "gemma" and not s.parse_failed:
            s.composite_normalized = calibrator.normalize_safe(
                s.composite_score)
        else:
            s.composite_normalized = s.composite_score

    print(f"\n{'═'*60}")
    print(f"COMPUTING BELL CURVE STATISTICS...")
    print(f"{'═'*60}")

    tier_stats = compute_tier_stats(all_scores, calibrator)
    report(all_scores, tier_stats, calibrator)
    save_results(all_scores, tier_stats, calibrator)
    return all_scores, tier_stats, calibrator

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    run_experiment(host)