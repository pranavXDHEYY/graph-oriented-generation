#!/usr/bin/env python3
"""
Experiment 20: Layer 1 Regional Stability Mapping
LOCATION: semantic_primitives/experiment_20_layer1_regions.py

PURPOSE:
Three ground truth Layer 1 combinations were identified in experiment 19b:
  move × Excitement    → physical momentum / agency in motion
  know × Satisfaction  → understanding as completion
  someone × Admiration → recognition of worth in another

This experiment asks: are these isolated peaks or connected regions?

STRATEGY:
For each ground truth combination, test:
  1. The anchor primitive × all stable emotions
  2. Semantic neighbor primitives × the anchor emotion
  3. The near-miss combinations that almost made ground truth

If stability extends outward from the anchors, we have regions.
If stability drops sharply, we have isolated peaks.

Regions → connected Layer 1 vocabulary with compositional rules
Peaks   → isolated compounds, no generalization possible

STABILITY PROTOCOL: identical to experiment 19b
  3 runs × 3 seeds × 3 models × 3 trials per combo
  Ground truth filter: STABLE + composite>=6.0 + 2+ models
"""
import requests
import json
import csv
import time
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── ground truth anchors from experiment 19b ─────────────────────────────────
GROUND_TRUTH_ANCHORS = {
    "move × Excitement":    {
        "concept": "physical momentum / agency in motion",
        "primitive": "move",
        "emotion": "Excitement",
    },
    "know × Satisfaction":  {
        "concept": "understanding as completion",
        "primitive": "know",
        "emotion": "Satisfaction",
    },
    "someone × Admiration": {
        "concept": "recognition of worth in another",
        "primitive": "someone",
        "emotion": "Admiration",
    },
}

# ── stable emotions from experiment 19b (sorted by stability) ─────────────────
STABLE_EMOTIONS = [
    "Surprise",    # range 0.25 — most stable
    "Triumph",     # range 0.26
    "Disgust",     # range 0.68
    "Confusion",   # range 0.78
    "Hope",        # range 0.90
    "Pride",       # range 1.00
    "Calmness",    # range 1.04
    "Fear",        # range 1.08
    "Nostalgia",   # range 1.14
    "Craving",     # range 1.20
    "Awe",         # range 1.27
    "Horror",      # range 1.33
    "Joy",         # range 1.44
    "Anxiety",     # range 1.44
]

# ── stable primitives from experiment 19b ────────────────────────────────────
STABLE_PRIMITIVES = [
    "something",   # range 0.64 — most stable
    "time",        # range 0.73
    "maybe",       # range 0.79
    "not",         # range 0.83
    "body",        # range 0.95
    "now",         # range 1.02
    "people",      # range 1.06
    "bad",         # range 1.06
    "do",          # range 1.07
    "after",       # range 1.14
    "far",         # range 1.17
    "feel",        # range 1.18
    "know",        # range 1.21
    "happen",      # range 1.25
]

# ── semantic neighbors for each anchor ───────────────────────────────────────
# words that mean similar things or belong to the same primitive family

MOVE_NEIGHBORS    = ["happen", "do", "live", "go", "flow",
                     "travel", "change", "shift", "act"]
KNOW_NEIGHBORS    = ["think", "feel", "see", "hear", "understand",
                     "learn", "find", "remember", "realize"]
SOMEONE_NEIGHBORS = ["people", "you", "I", "body", "person",
                     "other", "all", "one", "we"]

# ── near-miss combinations from experiment 19b ───────────────────────────────
# these almost made ground truth — worth deeper testing
NEAR_MISSES = [
    ("if",     "Hope"),        # 6.02 stable, missed model agreement
    ("I",      "Joy"),         # 6.21 stable, missed model agreement
    ("body",   "Excitement"),  # 5.68 stable, missed score threshold
    ("do",     "Excitement"),  # 5.61 stable, missed score threshold
    ("feel",   "Joy"),         # 5.56 stable, missed score threshold
    ("maybe",  "Hope"),        # 5.46 stable, missed score threshold
    ("happen", "Interest"),    # 5.61 stable, missed score threshold
    ("body",   "Calmness"),    # 5.78 stable, missed score threshold
]

# ── build the full test set ───────────────────────────────────────────────────

def build_test_combos() -> list[tuple[str, str, str]]:
    """
    Returns list of (primitive, emotion, region_label) tuples.
    region_label identifies which region/hypothesis each combo tests.
    """
    combos = []
    seen = set()

    def add(p, e, region):
        key = (p, e)
        if key not in seen:
            seen.add(key)
            combos.append((p, e, region))

    # ── Region 1: move anchor ──────────────────────────────────────────────
    # anchor itself
    add("move", "Excitement", "R1_ANCHOR")
    # move × all stable emotions
    for emo in STABLE_EMOTIONS:
        add("move", emo, "R1_MOVE_EXPAND")
    # neighbor primitives × Excitement
    for prim in MOVE_NEIGHBORS:
        add(prim, "Excitement", "R1_EXCITE_EXPAND")

    # ── Region 2: know anchor ──────────────────────────────────────────────
    add("know", "Satisfaction", "R2_ANCHOR")
    for emo in STABLE_EMOTIONS:
        add("know", emo, "R2_KNOW_EXPAND")
    for prim in KNOW_NEIGHBORS:
        add(prim, "Satisfaction", "R2_SATIS_EXPAND")

    # ── Region 3: someone anchor ───────────────────────────────────────────
    add("someone", "Admiration", "R3_ANCHOR")
    for emo in STABLE_EMOTIONS:
        add("someone", emo, "R3_SOMEONE_EXPAND")
    for prim in SOMEONE_NEIGHBORS:
        add(prim, "Admiration", "R3_ADMIRE_EXPAND")

    # ── near misses ────────────────────────────────────────────────────────
    for p, e in NEAR_MISSES:
        add(p, e, "NEAR_MISS")

    # ── stable × stable cross product (not already covered) ───────────────
    # test whether stable primitives + stable emotions = stable compounds
    for prim in STABLE_PRIMITIVES[:7]:   # top 7 most stable
        for emo in STABLE_EMOTIONS[:7]:  # top 7 most stable
            add(prim, emo, "STABLE_CROSS")

    return combos

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

TRIALS_PER_COMBO   = 3
NUM_RUNS           = 3
RANDOM_SEEDS       = [42, 137, 999]
STABILITY_STABLE   = 1.5
STABILITY_MARGINAL = 3.0
GROUND_TRUTH_MIN   = 6.0

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
    region: str
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
    region: str
    run1_avg: float
    run2_avg: float
    run3_avg: float
    mean_composite: float
    variance: float
    std_dev: float
    range_score: float
    stability_class: str
    is_ground_truth: bool
    model_agreement: int
    top_concept: str
    concept_consistency: float
    distance_from_anchor: str  # ANCHOR / NEAR / FAR based on region

# ── probe ─────────────────────────────────────────────────────────────────────

def probe_combo(primitive, emotion, model, host,
                run_id, seed, trial, model_short,
                region) -> TrialResult:
    prompt = build_prompt(primitive, emotion)
    raw    = ollama_call(host, model, prompt=prompt,
                         system=SYSTEM_MEMBRANE)
    result = extract_json(raw)

    base = TrialResult(
        run_id=run_id, seed=seed,
        model_name=model, model_short=model_short,
        primitive=primitive, emotion=emotion,
        combo=f"{primitive} × {emotion}",
        region=region, trial=trial,
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
    base.confidence            = safe_float(result.get("confidence",0.0))
    base.parse_failed          = False
    return base

# ── stability analysis ────────────────────────────────────────────────────────

def analyze_stability(all_results: list[TrialResult],
                      test_combos: list[tuple]) -> list[ComboStability]:
    combo_region = {
        f"{p} × {e}": r for p, e, r in test_combos
    }
    combos = list(dict.fromkeys(r.combo for r in all_results))
    results_out = []

    for combo in combos:
        primitive, emotion = combo.split(" × ")
        region = combo_region.get(combo, "UNKNOWN")
        combo_rs = [r for r in all_results
                   if r.combo == combo and not r.parse_failed]

        run_avgs = {}
        for run_id in range(1, NUM_RUNS + 1):
            run_rs = [r for r in combo_rs if r.run_id == run_id]
            run_avgs[run_id] = (
                sum(r.composite_score for r in run_rs) / len(run_rs)
                if run_rs else 0.0
            )

        run_scores = list(run_avgs.values())
        mean_c = sum(run_scores) / len(run_scores) if run_scores else 0
        var    = (sum((s - mean_c)**2 for s in run_scores) /
                  len(run_scores) if run_scores else 0)
        std    = var ** 0.5
        rng    = max(run_scores) - min(run_scores) if run_scores else 0

        if rng <= STABILITY_STABLE:
            stab = "STABLE"
        elif rng <= STABILITY_MARGINAL:
            stab = "MARGINAL"
        else:
            stab = "UNSTABLE"

        model_agree = 0
        for md in MODELS:
            mdr = [r for r in combo_rs
                  if r.model_name == md["name"]]
            if mdr:
                mavg = sum(r.composite_score
                           for r in mdr) / len(mdr)
                if mavg >= 6.0:
                    model_agree += 1

        is_gt = (stab == "STABLE" and
                 mean_c >= GROUND_TRUTH_MIN and
                 model_agree >= 2)

        concepts = [r.composite_concept for r in combo_rs
                   if r.composite_concept]
        top_c = (max(set(concepts), key=concepts.count)
                 if concepts else "")
        cc    = (concepts.count(top_c) / len(concepts)
                 if concepts else 0.0)

        # distance from anchor
        if "ANCHOR" in region:
            dist = "ANCHOR"
        elif "EXPAND" in region or "NEAR_MISS" in region:
            dist = "NEAR"
        else:
            dist = "FAR"

        results_out.append(ComboStability(
            combo=combo, primitive=primitive,
            emotion=emotion, region=region,
            run1_avg=round(run_avgs.get(1, 0), 2),
            run2_avg=round(run_avgs.get(2, 0), 2),
            run3_avg=round(run_avgs.get(3, 0), 2),
            mean_composite=round(mean_c, 3),
            variance=round(var, 3),
            std_dev=round(std, 3),
            range_score=round(rng, 2),
            stability_class=stab,
            is_ground_truth=is_gt,
            model_agreement=model_agree,
            top_concept=top_c,
            concept_consistency=round(cc, 3),
            distance_from_anchor=dist,
        ))

    return sorted(results_out,
                  key=lambda x: x.mean_composite, reverse=True)

# ── reporting ─────────────────────────────────────────────────────────────────

def report(stability: list[ComboStability]):
    gt = [s for s in stability if s.is_ground_truth]

    print(f"\n{'═'*60}")
    print(f"LAYER 1 REGIONAL STABILITY ANALYSIS")
    print(f"{'═'*60}")

    # stability overview
    stable   = [s for s in stability if s.stability_class == "STABLE"]
    marginal = [s for s in stability if s.stability_class == "MARGINAL"]
    unstable = [s for s in stability if s.stability_class == "UNSTABLE"]
    total    = len(stability)

    print(f"\n── STABILITY DISTRIBUTION ───────────────────────────────────")
    print(f"  STABLE    {len(stable):>4}/{total}  ({len(stable)/total:.0%})")
    print(f"  MARGINAL  {len(marginal):>4}/{total}  ({len(marginal)/total:.0%})")
    print(f"  UNSTABLE  {len(unstable):>4}/{total}  ({len(unstable)/total:.0%})")
    print(f"  GROUND TRUTH: {len(gt)}/{total}")

    # the key question — regions or peaks?
    print(f"\n── THE KEY QUESTION: REGIONS OR PEAKS? ──────────────────────")
    print(f"  Do anchors have stable neighbors, or are they isolated?\n")

    for anchor_combo, anchor_data in GROUND_TRUTH_ANCHORS.items():
        prim  = anchor_data["primitive"]
        emo   = anchor_data["emotion"]
        concept = anchor_data["concept"]

        print(f"  ── {anchor_combo}")
        print(f"     confirmed concept: {concept}\n")

        # same primitive, different emotions
        prim_expand = [s for s in stability
                      if s.primitive == prim
                      and s.emotion != emo]
        prim_gt = [s for s in prim_expand if s.is_ground_truth]
        prim_stable = [s for s in prim_expand
                      if s.stability_class == "STABLE"
                      and s.mean_composite >= 5.5]

        print(f"     {prim} × other emotions:")
        for s in sorted(prim_expand,
                        key=lambda x: x.mean_composite,
                        reverse=True)[:6]:
            gt_flag = " ★" if s.is_ground_truth else ""
            print(f"       {s.emotion:<20} "
                  f"{s.mean_composite:.2f}  "
                  f"{s.stability_class}{gt_flag}  "
                  f"→ {s.top_concept[:25]}")

        # same emotion, different primitives
        emo_expand = [s for s in stability
                     if s.emotion == emo
                     and s.primitive != prim]
        print(f"\n     other primitives × {emo}:")
        for s in sorted(emo_expand,
                        key=lambda x: x.mean_composite,
                        reverse=True)[:6]:
            gt_flag = " ★" if s.is_ground_truth else ""
            print(f"       {s.primitive:<15} "
                  f"{s.mean_composite:.2f}  "
                  f"{s.stability_class}{gt_flag}  "
                  f"→ {s.top_concept[:25]}")
        print()

    # new ground truth discoveries
    print(f"\n── NEW GROUND TRUTH DISCOVERIES ─────────────────────────────")
    # exclude the three known anchors
    known = set(GROUND_TRUTH_ANCHORS.keys())
    new_gt = [s for s in gt if s.combo not in known]
    if new_gt:
        print(f"  {len(new_gt)} new ground truth combinations found!\n")
        print(f"  {'Combination':<35} {'Mean':>5}  "
              f"{'Range':>5}  {'Models':>6}  Concept")
        print(f"  {'───────────':<35} {'────':>5}  "
              f"{'─────':>5}  {'──────':>6}  ───────")
        for s in new_gt:
            print(f"  {s.combo:<35} "
                  f"{s.mean_composite:>5.2f}  "
                  f"{s.range_score:>5.2f}  "
                  f"{s.model_agreement:>6}/3  "
                  f"{s.top_concept[:30]}")
    else:
        print(f"  No new ground truth combinations beyond the 3 anchors.")
        print(f"  The anchors may be isolated peaks, not connected regions.")

    # near miss promotions
    print(f"\n── NEAR MISS RESULTS ─────────────────────────────────────────")
    near_miss_results = [s for s in stability
                        if s.region == "NEAR_MISS"]
    for s in sorted(near_miss_results,
                    key=lambda x: x.mean_composite,
                    reverse=True):
        gt_flag = " ★ PROMOTED TO GROUND TRUTH" if s.is_ground_truth else ""
        print(f"  {s.combo:<35} "
              f"{s.mean_composite:.2f}  "
              f"{s.stability_class}  "
              f"range:{s.range_score:.2f}{gt_flag}")

    # stable cross product results
    print(f"\n── STABLE × STABLE CROSS PRODUCT ────────────────────────────")
    print(f"  Most stable primitives × most stable emotions")
    cross = [s for s in stability if s.region == "STABLE_CROSS"]
    cross_gt = [s for s in cross if s.is_ground_truth]
    print(f"  {len(cross)} tested, {len(cross_gt)} ground truth\n")
    for s in sorted(cross,
                    key=lambda x: x.mean_composite,
                    reverse=True)[:10]:
        gt_flag = " ★" if s.is_ground_truth else ""
        print(f"  {s.combo:<35} "
              f"{s.mean_composite:.2f}  "
              f"{s.stability_class}{gt_flag}  "
              f"→ {s.top_concept[:25]}")

    # complete ground truth vocabulary
    print(f"\n── COMPLETE LAYER 1 GROUND TRUTH VOCABULARY ─────────────────")
    print(f"  All combinations that survived every filter\n")
    all_gt = sorted(gt, key=lambda x: x.mean_composite, reverse=True)
    if all_gt:
        for i, s in enumerate(all_gt, 1):
            anchor_flag = " [original anchor]" if s.combo in known else " [new]"
            print(f"  {i:>2}. {s.combo:<35} "
                  f"{s.mean_composite:.2f}  "
                  f"→ {s.top_concept[:30]}"
                  f"{anchor_flag}")
    else:
        print("  No ground truth combinations found in this run.")

    # region verdict
    print(f"\n── REGION vs PEAK VERDICT ───────────────────────────────────")
    anchor_gt_count = sum(1 for s in gt if s.combo in known)
    new_gt_count    = len(new_gt)

    if new_gt_count >= 3:
        print(f"  REGIONS CONFIRMED — {new_gt_count} new ground truth "
              f"combinations cluster around anchors.")
        print(f"  The Layer 1 vocabulary is connected, not isolated.")
        print(f"  Compositional rules exist.")
    elif new_gt_count >= 1:
        print(f"  PARTIAL REGIONS — {new_gt_count} new ground truth "
              f"found near anchors.")
        print(f"  Some connectivity exists but anchors are largely peaks.")
    else:
        print(f"  ISOLATED PEAKS — no new ground truth around anchors.")
        print(f"  The 3 anchors are specific, not generalizable.")
        print(f"  The Layer 1 vocabulary may require different probing.")

    print(f"\n── IMPLICATIONS FOR LAYER 0a/0b → LAYER 1 INTERACTION ──────")
    print(f"  Based on these results, the interaction hypothesis is:")
    if new_gt_count >= 3:
        print(f"  CONNECTED: Layer 0 primitives combine with emotions")
        print(f"  in predictable neighborhood patterns. The membrane")
        print(f"  has semantic gravity wells, not just isolated points.")
        print(f"  Layer 0a operators + Layer 0b seeds → Layer 1 regions.")
    else:
        print(f"  DISCRETE: Each ground truth combination is a specific")
        print(f"  primitive-emotion binding, not a general rule.")
        print(f"  Layer 0 → Layer 1 requires explicit enumeration,")
        print(f"  not interpolation from nearby combinations.")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(all_results: list[TrialResult],
                 stability: list[ComboStability],
                 base="semantic_primitives/results_exp_20"):
    Path("semantic_primitives").mkdir(exist_ok=True)

    # trials
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

    # stability
    stab_path = Path(f"{base}_stability.json")
    with open(stab_path, "w") as f:
        json.dump([asdict(s) for s in stability], f, indent=2)

    # updated ground truth — merge with exp 19b findings
    gt = [s for s in stability if s.is_ground_truth]
    existing_gt = []
    gt_path = Path("semantic_primitives/ground_truth_layer1.json")
    if gt_path.exists():
        with open(gt_path) as f:
            existing = json.load(f)
            existing_gt = existing.get("ground_truth", [])

    existing_combos = {g["combo"] for g in existing_gt}
    new_entries = [
        {
            "combo":           s.combo,
            "primitive":       s.primitive,
            "emotion":         s.emotion,
            "mean_composite":  s.mean_composite,
            "range":           s.range_score,
            "model_agreement": s.model_agreement,
            "top_concept":     s.top_concept,
            "source":          "exp_20",
        }
        for s in gt if s.combo not in existing_combos
    ]

    all_gt_entries = existing_gt + new_entries
    with open(gt_path, "w") as f:
        json.dump({
            "timestamp":          datetime.now().isoformat(),
            "source":             "experiments 19b + 20",
            "total_ground_truth": len(all_gt_entries),
            "ground_truth":       sorted(
                all_gt_entries,
                key=lambda x: x["mean_composite"],
                reverse=True
            ),
        }, f, indent=2)

    # update primitive summary
    psummary = Path("semantic_primitives/primitive_summary.json")
    existing_ps = {}
    if psummary.exists():
        with open(psummary) as f:
            existing_ps = json.load(f)

    existing_ps["exp_20"] = {
        "timestamp":         datetime.now().isoformat(),
        "new_ground_truth":  [s.combo for s in gt
                             if s.combo not in
                             set(GROUND_TRUTH_ANCHORS.keys())],
        "total_ground_truth": len(all_gt_entries),
        "region_verdict":    "REGIONS" if len(new_entries) >= 3
                             else "PARTIAL" if len(new_entries) >= 1
                             else "PEAKS",
    }
    with open(psummary, "w") as f:
        json.dump(existing_ps, f, indent=2)

    print(f"\n── SAVED ────────────────────────────────────────────────────")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {stab_path}")
    print(f"  ground_truth_layer1.json  ← updated with new entries")
    print(f"  primitive_summary.json    ← updated")

# ── main ──────────────────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434"):
    test_combos = build_test_combos()
    total_calls = (len(test_combos) * TRIALS_PER_COMBO *
                   len(MODELS) * NUM_RUNS)

    print(f"SRM experiment 20 — Layer 1 regional stability mapping")
    print(f"Anchors: {list(GROUND_TRUTH_ANCHORS.keys())}")
    print(f"{len(test_combos)} combinations × {NUM_RUNS} runs × "
          f"{len(MODELS)} models × {TRIALS_PER_COMBO} trials")
    print(f"Total calls: {total_calls}")
    print(f"Seeds: {RANDOM_SEEDS}\n")
    print(f"Regions: move, know, someone + near-misses + stable×stable\n")

    all_results: list[TrialResult] = []

    for run_id, seed in enumerate(RANDOM_SEEDS, 1):
        random.seed(seed)
        print(f"\n{'═'*60}")
        print(f"RUN {run_id}/{NUM_RUNS}  (seed={seed})")
        print(f"{'═'*60}")

        run_combos = test_combos.copy()
        random.shuffle(run_combos)

        for model_def in MODELS:
            model_name  = model_def["name"]
            model_short = model_def["short"]
            print(f"\n  ── {model_name}")

            for primitive, emotion, region in run_combos:
                combo_str = f"{primitive} × {emotion}"
                trial_scores = []

                for trial in range(1, TRIALS_PER_COMBO + 1):
                    r = probe_combo(
                        primitive, emotion,
                        model_name, host,
                        run_id, seed,
                        trial, model_short,
                        region
                    )
                    all_results.append(r)
                    trial_scores.append(r.composite_score)
                    time.sleep(0.15)

                avg = (sum(trial_scores) / len(trial_scores)
                       if trial_scores else 0)
                print(f"    [{region:<18}] "
                      f"{combo_str:<35} "
                      f"avg:{avg:>5.2f}  "
                      f"→ {r.composite_concept[:20]}")

        # checkpoint
        ckpt = Path(
            f"semantic_primitives/"
            f"results_exp_20_run{run_id}_checkpoint.json"
        )
        with open(ckpt, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\n  checkpoint → {ckpt}")

    print(f"\n{'═'*60}")
    print(f"COMPUTING STABILITY...")
    print(f"{'═'*60}")

    stability = analyze_stability(all_results, test_combos)
    report(stability)
    save_results(all_results, stability)
    return all_results, stability

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    run_experiment(host)