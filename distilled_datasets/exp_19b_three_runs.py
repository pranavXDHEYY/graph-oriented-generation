#!/usr/bin/env python3
"""
Experiment 19b: Stability Validation — Multi-Run Consistency
LOCATION: semantic_primitives/experiment_19b_stability.py

PURPOSE:
Experiment 19 showed high variance between runs on the same combinations.
A single run is insufficient to distinguish signal from noise.

This experiment runs the full primitive × emotion scoring protocol
3 times with different random seeds and measures:
  - Which combinations produce stable scores across all 3 runs
  - Which combinations are noisy and unreliable
  - Whether the top-10 composite leaders hold across runs
  - Cross-run stability as a filter for ground truth

STABILITY DEFINITION:
  A combination is STABLE if its composite score variance
  across 3 runs is <= 1.5 points (on a 0-10 scale).
  A combination is UNSTABLE if variance > 3.0 points.
  Between 1.5 and 3.0 = MARGINAL.

GROUND TRUTH FILTER:
  Combinations that are:
    1. STABLE across 3 runs AND
    2. Score >= 6.0 composite average AND
    3. Consistent across >= 2 models
  ...are promoted to GROUND TRUTH Layer 1 candidates.
"""
import requests
import json
import csv
import time
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── same mappings as experiment 19 ───────────────────────────────────────────
NSM_EMOTION_MAPPINGS = {
    "I":          ["Joy", "Anxiety", "Fear", "Shame"],
    "you":        ["Admiration", "Adoration", "Romance", "Sympathy"],
    "someone":    ["Admiration", "Sympathy", "Envy", "Adoration"],
    "people":     ["Admiration", "Sympathy", "Awe"],
    "something":  ["Interest", "Craving", "Awe"],
    "body":       ["Calmness", "Excitement", "Fear", "Satisfaction"],
    "good":       ["Joy", "Satisfaction", "Admiration", "Calmness"],
    "bad":        ["Disgust", "Sadness", "Fear", "Horror"],
    "think":      ["Interest", "Confusion", "Awe"],
    "know":       ["Satisfaction", "Calmness", "Anxiety"],
    "want":       ["Craving", "Excitement", "Anxiety"],
    "feel":       ["Joy", "Sadness", "Fear", "Awe", "Nostalgia"],
    "see":        ["Aesthetic Appreciation", "Interest", "Awe", "Fear"],
    "hear":       ["Aesthetic Appreciation", "Interest", "Awe", "Calmness"],
    "do":         ["Excitement", "Triumph", "Anxiety"],
    "happen":     ["Awe", "Fear", "Surprise", "Interest"],
    "move":       ["Excitement", "Awe", "Fear", "Interest"],
    "live":       ["Joy", "Excitement", "Awe", "Fear"],
    "die":        ["Horror", "Fear", "Sadness", "Awe", "Nostalgia"],
    "time":       ["Nostalgia", "Anxiety", "Awe", "Interest"],
    "now":        ["Anxiety", "Excitement", "Calmness"],
    "before":     ["Nostalgia", "Sadness", "Pride"],
    "after":      ["Relief", "Satisfaction", "Nostalgia"],
    "moment":     ["Awe", "Excitement", "Fear", "Interest"],
    "place":      ["Nostalgia", "Awe", "Anxiety", "Curiosity"],
    "here":       ["Calmness", "Nostalgia", "Anxiety"],
    "far":        ["Nostalgia", "Awe", "Anxiety"],
    "near":       ["Calmness", "Anxiety", "Romance"],
    "inside":     ["Comfort", "Curiosity", "Fear"],
    "not":        ["Disgust", "Fear", "Anxiety", "Calmness"],
    "maybe":      ["Hope", "Anxiety", "Curiosity"],
    "if":         ["Hope", "Anxiety", "Curiosity"],
    "very":       ["Awe", "Excitement", "Fear"],
    "more":       ["Interest", "Craving", "Excitement", "Envy"],
}

MODELS = [
    {"name": "qwen2.5:0.5b",  "short": "qwen"},
    {"name": "gemma3:1b",     "short": "gemma"},
    {"name": "llama3.2:1b",   "short": "llama"},
]

SCORE_DIMENSIONS = [
    "meaning_score",
    "excitement_score",
    "emotional_elicitation",
    "clarity_score",
    "universality_score",
    "embodiment_score",
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

TRIALS_PER_COMBO   = 3
NUM_RUNS           = 3
STABILITY_STABLE   = 1.5   # max variance for STABLE
STABILITY_MARGINAL = 3.0   # max variance for MARGINAL
GROUND_TRUTH_MIN   = 6.0   # min composite for ground truth

RANDOM_SEEDS = [42, 137, 999]

# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM_MEMBRANE = """You are a language membrane with introspective
scoring capability.

When given a primitive concept paired with an emotion, you will:
1. Feel into the combination — what arises internally
2. Score your response across 7 dimensions from 0-10

Be honest about your internal states. These scores are
speculative but meaningful. Trust your first response.
Score 0 = none/absent, 10 = maximum/overwhelming.

Output ONLY valid JSON. No explanation outside the JSON."""

def build_prompt(primitive: str, emotion: str) -> str:
    return (
        f"Primitive concept: {primitive}\n"
        f"Emotion: {emotion}\n\n"
        f"Score this combination across all 7 dimensions.\n\n"
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
class TrialResult:
    run_id: int
    seed: int
    model_name: str
    model_short: str
    primitive: str
    emotion: str
    combo: str
    trial: int
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
    confidence: float
    parse_failed: bool

@dataclass
class ComboStability:
    combo: str
    primitive: str
    emotion: str
    # per-run composite averages
    run1_avg: float
    run2_avg: float
    run3_avg: float
    # cross-run statistics
    mean_composite: float
    variance: float
    std_dev: float
    min_score: float
    max_score: float
    range_score: float
    # stability classification
    stability_class: str    # STABLE / MARGINAL / UNSTABLE
    # ground truth
    is_ground_truth: bool
    model_agreement: int    # how many models scored >= 6.0
    # top concept across all runs
    top_concept: str
    concept_consistency: float  # fraction of runs agreeing on concept

# ── probe ─────────────────────────────────────────────────────────────────────

def probe_combo(primitive: str, emotion: str,
                model: str, host: str,
                run_id: int, seed: int,
                trial: int, model_short: str) -> TrialResult:
    prompt = build_prompt(primitive, emotion)
    raw    = ollama_call(host, model,
                         prompt=prompt,
                         system=SYSTEM_MEMBRANE)
    result = extract_json(raw)

    base = TrialResult(
        run_id=run_id, seed=seed,
        model_name=model, model_short=model_short,
        primitive=primitive, emotion=emotion,
        combo=f"{primitive} × {emotion}",
        trial=trial,
        meaning_score=0.0, excitement_score=0.0,
        emotional_elicitation=0.0, clarity_score=0.0,
        universality_score=0.0, embodiment_score=0.0,
        novelty_score=0.0, composite_score=0.0,
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
    base.dominant_sensation    = str(result.get("dominant_sensation",""))
    base.composite_concept     = str(result.get("composite_concept",""))
    base.confidence            = safe_float(result.get("confidence", 0.0))
    base.parse_failed          = False
    return base

# ── stability analysis ────────────────────────────────────────────────────────

def analyze_stability(all_results: list[TrialResult]) -> list[ComboStability]:
    combos = list(dict.fromkeys(r.combo for r in all_results))
    stability_results = []

    for combo in combos:
        primitive, emotion = combo.split(" × ")
        combo_results = [r for r in all_results
                        if r.combo == combo and not r.parse_failed]

        # per-run averages
        run_avgs = {}
        for run_id in range(1, NUM_RUNS + 1):
            run_results = [r for r in combo_results
                          if r.run_id == run_id]
            if run_results:
                run_avgs[run_id] = sum(
                    r.composite_score for r in run_results
                ) / len(run_results)
            else:
                run_avgs[run_id] = 0.0

        run_scores = list(run_avgs.values())
        mean_comp  = sum(run_scores) / len(run_scores) if run_scores else 0
        variance   = sum((s - mean_comp) ** 2
                        for s in run_scores) / len(run_scores) if run_scores else 0
        std_dev    = variance ** 0.5
        min_score  = min(run_scores) if run_scores else 0
        max_score  = max(run_scores) if run_scores else 0
        range_sc   = max_score - min_score

        # stability class
        if range_sc <= STABILITY_STABLE:
            stability = "STABLE"
        elif range_sc <= STABILITY_MARGINAL:
            stability = "MARGINAL"
        else:
            stability = "UNSTABLE"

        # model agreement — how many models avg >= 6.0
        model_agree = 0
        for model_def in MODELS:
            model_results = [r for r in combo_results
                            if r.model_name == model_def["name"]]
            if model_results:
                model_avg = sum(
                    r.composite_score for r in model_results
                ) / len(model_results)
                if model_avg >= 6.0:
                    model_agree += 1

        # ground truth filter
        is_gt = (
            stability == "STABLE" and
            mean_comp >= GROUND_TRUTH_MIN and
            model_agree >= 2
        )

        # top concept
        concepts = [r.composite_concept for r in combo_results
                   if r.composite_concept]
        top_concept = (
            max(set(concepts), key=concepts.count)
            if concepts else ""
        )
        concept_consistency = (
            concepts.count(top_concept) / len(concepts)
            if concepts else 0.0
        )

        stability_results.append(ComboStability(
            combo=combo,
            primitive=primitive,
            emotion=emotion,
            run1_avg=round(run_avgs.get(1, 0), 2),
            run2_avg=round(run_avgs.get(2, 0), 2),
            run3_avg=round(run_avgs.get(3, 0), 2),
            mean_composite=round(mean_comp, 3),
            variance=round(variance, 3),
            std_dev=round(std_dev, 3),
            min_score=round(min_score, 2),
            max_score=round(max_score, 2),
            range_score=round(range_sc, 2),
            stability_class=stability,
            is_ground_truth=is_gt,
            model_agreement=model_agree,
            top_concept=top_concept,
            concept_consistency=round(concept_consistency, 3),
        ))

    return sorted(stability_results,
                  key=lambda x: x.mean_composite, reverse=True)

# ── reporting ─────────────────────────────────────────────────────────────────

def report(stability: list[ComboStability],
           all_results: list[TrialResult]):

    stable   = [s for s in stability if s.stability_class == "STABLE"]
    marginal = [s for s in stability if s.stability_class == "MARGINAL"]
    unstable = [s for s in stability if s.stability_class == "UNSTABLE"]
    gt       = [s for s in stability if s.is_ground_truth]

    print(f"\n{'═'*60}")
    print(f"STABILITY ANALYSIS — {NUM_RUNS} RUNS × {len(RANDOM_SEEDS)} SEEDS")
    print(f"{'═'*60}")

    print(f"\n── STABILITY DISTRIBUTION ───────────────────────────────────")
    total = len(stability)
    print(f"  STABLE    {len(stable):>4}/{total}  "
          f"({len(stable)/total:.0%})  "
          f"variance <= {STABILITY_STABLE}")
    print(f"  MARGINAL  {len(marginal):>4}/{total}  "
          f"({len(marginal)/total:.0%})  "
          f"variance {STABILITY_STABLE}-{STABILITY_MARGINAL}")
    print(f"  UNSTABLE  {len(unstable):>4}/{total}  "
          f"({len(unstable)/total:.0%})  "
          f"variance > {STABILITY_MARGINAL}")
    print(f"\n  GROUND TRUTH (stable + score>=6 + 2+ models): "
          f"{len(gt)}/{total}")

    print(f"\n── GROUND TRUTH LAYER 1 CANDIDATES ─────────────────────────")
    print(f"  These combinations are stable, high-scoring, and")
    print(f"  cross-model consistent. These are real.\n")
    print(f"  {'Combination':<35} {'Mean':>5}  {'Range':>5}  "
          f"{'Models':>6}  Concept")
    print(f"  {'───────────':<35} {'────':>5}  {'─────':>5}  "
          f"{'──────':>6}  ───────")
    for s in gt:
        print(f"  {s.combo:<35} "
              f"{s.mean_composite:>5.2f}  "
              f"{s.range_score:>5.2f}  "
              f"{s.model_agreement:>6}/3  "
              f"{s.top_concept[:25]}")

    print(f"\n── RUN-BY-RUN COMPARISON — TOP 20 ───────────────────────────")
    print(f"  {'Combination':<35} {'Run1':>5}  {'Run2':>5}  "
          f"{'Run3':>5}  {'Mean':>5}  {'Range':>5}  Class")
    print(f"  {'───────────':<35} {'────':>5}  {'────':>5}  "
          f"{'────':>5}  {'────':>5}  {'─────':>5}  ─────")
    for s in stability[:20]:
        flag = " ★" if s.is_ground_truth else ""
        print(f"  {s.combo:<35} "
              f"{s.run1_avg:>5.2f}  "
              f"{s.run2_avg:>5.2f}  "
              f"{s.run3_avg:>5.2f}  "
              f"{s.mean_composite:>5.2f}  "
              f"{s.range_score:>5.2f}  "
              f"{s.stability_class}{flag}")

    print(f"\n── MOST STABLE HIGH-SCORING COMBINATIONS ────────────────────")
    print(f"  Sorted by stability (lowest variance) among high scorers")
    high_stable = sorted(
        [s for s in stability if s.mean_composite >= 5.0],
        key=lambda x: x.range_score
    )[:15]
    for s in high_stable:
        bar = "█" * int(s.mean_composite) + "░" * (10 - int(s.mean_composite))
        print(f"  {s.combo:<35} [{bar}] "
              f"{s.mean_composite:.2f}  "
              f"range:{s.range_score:.2f}  "
              f"{s.stability_class}")

    print(f"\n── MOST UNSTABLE COMBINATIONS ───────────────────────────────")
    print(f"  High variance — do not trust single-run scores")
    most_unstable = sorted(unstable,
                           key=lambda x: x.range_score,
                           reverse=True)[:10]
    for s in most_unstable:
        print(f"  {s.combo:<35} "
              f"run1:{s.run1_avg:.1f}  "
              f"run2:{s.run2_avg:.1f}  "
              f"run3:{s.run3_avg:.1f}  "
              f"range:{s.range_score:.1f}")

    print(f"\n── DID TOP-10 FROM EXP 19 HOLD? ─────────────────────────────")
    print(f"  Checking exp 19 composite leaders against stability\n")
    exp19_top10 = [
        "body × Excitement",
        "people × Admiration",
        "good × Joy",
        "moment × Excitement",
        "now × Excitement",
        "near × Romance",
        "see × Interest",
        "more × Excitement",
        "if × Hope",
        "far × Nostalgia",
    ]
    for combo in exp19_top10:
        match = next((s for s in stability
                      if s.combo == combo), None)
        if match:
            gt_flag = " ★ GROUND TRUTH" if match.is_ground_truth else ""
            print(f"  {combo:<35} "
                  f"mean:{match.mean_composite:.2f}  "
                  f"range:{match.range_score:.2f}  "
                  f"{match.stability_class}{gt_flag}")
        else:
            print(f"  {combo:<35} not found in results")

    print(f"\n── PRIMITIVE STABILITY RANKING ──────────────────────────────")
    print(f"  Which primitives produce the most stable scores?")
    prim_ranges: dict[str, list] = defaultdict(list)
    for s in stability:
        prim_ranges[s.primitive].append(s.range_score)
    prim_avg_range = {
        p: round(sum(v)/len(v), 2)
        for p, v in prim_ranges.items()
    }
    for prim, avg_range in sorted(prim_avg_range.items(),
                                   key=lambda x: x[1]):
        bar = "█" * max(0, 10 - int(avg_range * 2))
        print(f"  {prim:<12} [{bar:<10}] avg_range:{avg_range:.2f}")

    print(f"\n── EMOTION STABILITY RANKING ────────────────────────────────")
    print(f"  Which emotions produce the most stable scores?")
    emo_ranges: dict[str, list] = defaultdict(list)
    for s in stability:
        emo_ranges[s.emotion].append(s.range_score)
    emo_avg_range = {
        e: round(sum(v)/len(v), 2)
        for e, v in emo_ranges.items()
    }
    for emo, avg_range in sorted(emo_avg_range.items(),
                                  key=lambda x: x[1])[:15]:
        bar = "█" * max(0, 10 - int(avg_range * 2))
        print(f"  {emo:<25} [{bar:<10}] avg_range:{avg_range:.2f}")

    print(f"\n── WHAT IS REAL ─────────────────────────────────────────────")
    print(f"  {len(gt)} ground truth combinations identified.")
    if gt:
        print(f"  These survived 3 runs, score >= {GROUND_TRUTH_MIN}, "
              f"and 2+ models agree.")
        print(f"  They represent the most reliable Layer 1 vocabulary")
        print(f"  discovered so far in the SRM research program.")
    print(f"\n  {len(unstable)} combinations are too noisy to trust.")
    print(f"  Single-run scores for these are meaningless.")
    print(f"  Exclude them from any downstream analysis.")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(all_results: list[TrialResult],
                 stability: list[ComboStability],
                 base="semantic_primitives/results_exp_19b"):
    Path("semantic_primitives").mkdir(exist_ok=True)

    # all trial results
    json_path = Path(f"{base}_trials.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    csv_path = Path(f"{base}_trials.csv")
    if all_results:
        fieldnames = list(asdict(all_results[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))

    # stability analysis
    stab_path = Path(f"{base}_stability.json")
    with open(stab_path, "w") as f:
        json.dump([asdict(s) for s in stability], f, indent=2)

    stab_csv = Path(f"{base}_stability.csv")
    if stability:
        fieldnames = list(asdict(stability[0]).keys())
        with open(stab_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in stability:
                writer.writerow(asdict(s))

    # ground truth export — clean list of validated combinations
    gt = [s for s in stability if s.is_ground_truth]
    gt_path = Path("semantic_primitives/ground_truth_layer1.json")
    with open(gt_path, "w") as f:
        json.dump({
            "timestamp":   datetime.now().isoformat(),
            "source":      "experiment_19b",
            "method":      f"{NUM_RUNS} runs × {len(MODELS)} models",
            "filter":      f"STABLE + composite>={GROUND_TRUTH_MIN} + 2+ models",
            "total_tested": len(stability),
            "ground_truth_count": len(gt),
            "ground_truth": [
                {
                    "combo":           s.combo,
                    "primitive":       s.primitive,
                    "emotion":         s.emotion,
                    "mean_composite":  s.mean_composite,
                    "range":           s.range_score,
                    "model_agreement": s.model_agreement,
                    "top_concept":     s.top_concept,
                }
                for s in gt
            ],
        }, f, indent=2)

    # update primitive summary
    psummary_path = Path(
        "semantic_primitives/primitive_summary.json")
    existing = {}
    if psummary_path.exists():
        with open(psummary_path) as f:
            existing = json.load(f)

    existing["exp_19b"] = {
        "timestamp":       datetime.now().isoformat(),
        "ground_truth":    [s.combo for s in gt],
        "stable_count":    len([s for s in stability
                               if s.stability_class == "STABLE"]),
        "unstable_count":  len([s for s in stability
                               if s.stability_class == "UNSTABLE"]),
    }
    with open(psummary_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n── SAVED ────────────────────────────────────────────────────")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {stab_path}")
    print(f"  {stab_csv}")
    print(f"  semantic_primitives/ground_truth_layer1.json  ← KEY OUTPUT")
    print(f"  semantic_primitives/primitive_summary.json  (updated)")

# ── main ──────────────────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434"):
    combos = [
        (primitive, emotion)
        for primitive, emotions in NSM_EMOTION_MAPPINGS.items()
        for emotion in emotions
    ]

    total_combos = len(combos)
    total_calls  = (total_combos * TRIALS_PER_COMBO *
                    len(MODELS) * NUM_RUNS)

    print(f"SRM experiment 19b — stability validation")
    print(f"{NUM_RUNS} runs × {len(MODELS)} models × "
          f"{total_combos} combos × {TRIALS_PER_COMBO} trials")
    print(f"Total calls: {total_calls}")
    print(f"Seeds: {RANDOM_SEEDS}")
    print(f"Ground truth filter: "
          f"STABLE + score>={GROUND_TRUTH_MIN} + 2+ models\n")

    all_results: list[TrialResult] = []

    for run_id, seed in enumerate(RANDOM_SEEDS, 1):
        random.seed(seed)
        print(f"\n{'═'*60}")
        print(f"RUN {run_id}/{NUM_RUNS}  (seed={seed})")
        print(f"{'═'*60}")

        # shuffle combo order per run — different seed = different order
        run_combos = combos.copy()
        random.shuffle(run_combos)

        for model_def in MODELS:
            model_name  = model_def["name"]
            model_short = model_def["short"]
            print(f"\n  ── {model_name}")

            for primitive, emotion in run_combos:
                combo_str = f"{primitive} × {emotion}"
                trial_scores = []

                for trial in range(1, TRIALS_PER_COMBO + 1):
                    r = probe_combo(
                        primitive, emotion,
                        model_name, host,
                        run_id, seed,
                        trial, model_short
                    )
                    all_results.append(r)
                    trial_scores.append(r.composite_score)
                    time.sleep(0.15)

                avg = (sum(trial_scores) / len(trial_scores)
                       if trial_scores else 0)
                print(f"    {combo_str:<35} "
                      f"avg:{avg:>5.2f}  "
                      f"→ {r.composite_concept[:20]}")

        # checkpoint after each run
        save_checkpoint(all_results, run_id)

    # full stability analysis
    print(f"\n{'═'*60}")
    print(f"COMPUTING STABILITY ACROSS {NUM_RUNS} RUNS...")
    print(f"{'═'*60}")
    stability = analyze_stability(all_results)

    report(stability, all_results)
    save_results(all_results, stability)
    return all_results, stability

def save_checkpoint(results: list[TrialResult], run_id: int):
    path = Path(
        f"semantic_primitives/results_exp_19b_run{run_id}_checkpoint.json"
    )
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  checkpoint saved → {path}")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    run_experiment(host)