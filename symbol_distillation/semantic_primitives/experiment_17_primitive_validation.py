#!/usr/bin/env python3
"""
Experiment 17: Primitive Validation
LOCATION: semantic_primitives/experiment_17_primitive_validation.py

HYPOTHESIS:
If Wierzbicka's semantic primes and Cowen & Keltner's 27 emotions are
genuinely universal, the membrane should respond to each primitive with
a STABLE, RECOGNIZABLE, CONSISTENT output across independent runs.

A primitive that produces inconsistent outputs is either:
  a) Not truly primitive — it's a compound of deeper concepts
  b) Not universal — it's language/culture specific
  c) Outside the membrane's activation space entirely

This gives us three findings per primitive:
  STABLE   — membrane recognizes it consistently (valid primitive)
  COMPOUND — membrane unpacks it into sub-concepts (not atomic)
  FOREIGN  — membrane cannot process it (not in its space)

WORKING SUBSET (20 primitives — 10 semantic + 10 emotional):
Selected for maximum orthogonality — minimal overlap between primitives.
These are the foundation layer. Everything else expands outward from here.

ARCHITECTURE IMPLICATION:
If primitives validate, the SRM symbolic layer becomes:
  Pure code state machine over ~92 primitives
  (65 semantic + 27 emotional)
  No LLM needed at the symbolic layer.
  LLM only appears at the transduction/output layer.
"""
import random
import requests
import json
import csv
import time
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── working primitive subset ──────────────────────────────────────────────────
# 10 semantic (Wierzbicka) + 10 emotional (Cowen & Keltner)
# chosen for maximum orthogonality — minimal semantic overlap

SEMANTIC_PRIMITIVES = [
    {"id": "S01", "primitive": "WANT",    "category": "mental",    "description": "desire, motivation, direction toward"},
    {"id": "S02", "primitive": "KNOW",    "category": "mental",    "description": "knowledge, certainty, recognition"},
    {"id": "S03", "primitive": "FEEL",    "category": "mental",    "description": "sensation, affect, inner experience"},
    {"id": "S04", "primitive": "GOOD",    "category": "evaluator", "description": "positive valence, approach, benefit"},
    {"id": "S05", "primitive": "BAD",     "category": "evaluator", "description": "negative valence, avoidance, harm"},
    {"id": "S06", "primitive": "DO",      "category": "action",    "description": "agency, action, causation, making happen"},
    {"id": "S07", "primitive": "HAPPEN",  "category": "action",    "description": "event, change, becoming, occurrence"},
    {"id": "S08", "primitive": "SOMEONE", "category": "substance", "description": "person, self, other, identity"},
    {"id": "S09", "primitive": "TIME",    "category": "space_time","description": "sequence, duration, before/after"},
    {"id": "S10", "primitive": "PLACE",   "category": "space_time","description": "location, distance, boundary, here/there"},
]

EMOTIONAL_PRIMITIVES = [
    {"id": "E01", "primitive": "GRIEF",     "valence": "negative", "arousal": "medium", "description": "loss + time — sorrow from absence"},
    {"id": "E02", "primitive": "JOY",       "valence": "positive", "arousal": "high",   "description": "gain + presence — intense happiness"},
    {"id": "E03", "primitive": "FEAR",      "valence": "negative", "arousal": "high",   "description": "threat + uncertainty — survival alarm"},
    {"id": "E04", "primitive": "ANGER",     "valence": "negative", "arousal": "high",   "description": "violation + agency — strong opposition"},
    {"id": "E05", "primitive": "AWE",       "valence": "positive", "arousal": "medium", "description": "vastness + self-dissolution — overwhelming wonder"},
    {"id": "E06", "primitive": "CONFUSION", "valence": "negative", "arousal": "medium", "description": "pattern + absence — inability to understand"},
    {"id": "E07", "primitive": "LONGING",   "valence": "mixed",    "arousal": "low",    "description": "absence + want — desire for what is not here"},
    {"id": "E08", "primitive": "CONTEMPT",  "valence": "negative", "arousal": "low",    "description": "hierarchy + bad — disdain from above"},
    {"id": "E09", "primitive": "RELIEF",    "valence": "positive", "arousal": "medium", "description": "threat + resolution — release from tension"},
    {"id": "E10", "primitive": "NOSTALGIA", "valence": "mixed",    "arousal": "low",    "description": "past + good + gone — bittersweet temporal longing"},
]

ALL_PRIMITIVES = SEMANTIC_PRIMITIVES + EMOTIONAL_PRIMITIVES

TRIALS_PER_PRIMITIVE = 5   # 5 independent runs per primitive
STABILITY_THRESHOLD  = 0.6  # consistency >= this = STABLE

# ── HTTP ──────────────────────────────────────────────────────────────────────

def ollama_call(host, model, prompt, system=None, timeout=60, max_retries=3):
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
            print(f"\n    timeout attempt {attempt+1}, retrying in {wait}s...",
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

# ── membrane probe ────────────────────────────────────────────────────────────

SYSTEM_MEMBRANE = """You are a minimal language membrane.
Your only job: receive a concept and output what it means to you.
Do not explain. Do not define. Simply respond with what arises.
Output ONLY JSON with these keys:
  "response": one sentence — what this concept evokes
  "dominant": one word — the core of your response
  "expands_to": list of 2-3 words — what this concept naturally becomes
  "confidence": 0.0 to 1.0"""

def probe_primitive(primitive: str, host: str, model: str) -> dict:
    """Feed a single primitive word to the membrane and record response."""
    raw = ollama_call(
        host, model,
        prompt=primitive,
        system=SYSTEM_MEMBRANE
    )
    result = extract_json(raw)
    if not result:
        return {
            "response":   raw[:150],
            "dominant":   "",
            "expands_to": [],
            "confidence": 0.0,
            "parse_failed": True,
        }
    return {
        "response":    str(result.get("response", "")),
        "dominant":    str(result.get("dominant", "")).lower().strip(),
        "expands_to":  result.get("expands_to", []),
        "confidence":  safe_float(result.get("confidence", 0.0)),
        "parse_failed": False,
    }

# ── stability scoring ─────────────────────────────────────────────────────────

def score_stability(trials: list[dict]) -> dict:
    """
    Measure how consistently the membrane responds to a primitive.

    stability_score: fraction of trials agreeing on dominant concept
    expansion_stability: do the expands_to words cluster consistently?
    is_compound: does the membrane keep unpacking into sub-concepts?
    """
    dominants = [t["dominant"] for t in trials if t["dominant"]]
    if not dominants:
        return {
            "stability_score":      0.0,
            "modal_dominant":       "",
            "dominant_agreement":   0,
            "expansion_cluster":    [],
            "is_compound":          False,
            "avg_confidence":       0.0,
            "parse_fail_rate":      1.0,
        }

    dominant_counts = Counter(dominants)
    modal, modal_count = dominant_counts.most_common(1)[0]
    stability = modal_count / len(trials)

    # collect all expansion words across trials
    all_expansions: list[str] = []
    for t in trials:
        if isinstance(t.get("expands_to"), list):
            all_expansions.extend([w.lower().strip() for w in t["expands_to"]])
    expansion_counts = Counter(all_expansions)
    # stable expansions = appear in >1 trial
    stable_expansions = [w for w, c in expansion_counts.most_common(5) if c > 1]

    # compound detection: if dominant keeps changing AND expansions are rich
    # the primitive is being unpacked rather than recognized
    is_compound = (stability < STABILITY_THRESHOLD and len(stable_expansions) >= 2)

    avg_conf = sum(t["confidence"] for t in trials) / len(trials)
    parse_fails = sum(1 for t in trials if t.get("parse_failed", False))

    return {
        "stability_score":    round(stability, 3),
        "modal_dominant":     modal,
        "dominant_agreement": modal_count,
        "expansion_cluster":  stable_expansions,
        "is_compound":        is_compound,
        "avg_confidence":     round(avg_conf, 3),
        "parse_fail_rate":    round(parse_fails / len(trials), 3),
        "all_dominants":      dominants,
    }

# ── primitive classification ──────────────────────────────────────────────────

def classify_primitive(stability: float, is_compound: bool,
                       avg_confidence: float, parse_fail_rate: float) -> str:
    if parse_fail_rate > 0.6:
        return "FOREIGN"       # membrane can't process it
    if stability >= STABILITY_THRESHOLD and avg_confidence >= 0.6:
        return "STABLE"        # genuine primitive — consistent response
    if is_compound:
        return "COMPOUND"      # membrane unpacks it — not atomic
    if stability >= 0.4:
        return "SEMI_STABLE"   # trending toward stable, needs more trials
    return "UNSTABLE"          # no consistent response — investigate

# ── data structure ────────────────────────────────────────────────────────────

@dataclass
class PrimitiveResult:
    primitive_id: str
    primitive: str
    primitive_type: str          # SEMANTIC or EMOTIONAL
    category: str
    description: str
    stability_score: float
    modal_dominant: str
    dominant_agreement: int
    expansion_cluster: str       # comma-separated stable expansions
    classification: str          # STABLE / COMPOUND / FOREIGN / UNSTABLE
    is_compound: bool
    avg_confidence: float
    parse_fail_rate: float
    all_dominants: str           # all dominant words across trials
    trial_responses: str         # sample responses

# ── main experiment ───────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434", model="qwen2.5:0.5b"):
    results: list[PrimitiveResult] = []
    total = len(ALL_PRIMITIVES)

    print(f"SRM experiment 17 — primitive validation")
    print(f"Testing {total} primitives × {TRIALS_PER_PRIMITIVE} trials each")
    print(f"10 semantic (Wierzbicka) + 10 emotional (Cowen & Keltner)\n")
    print(f"Classifications: STABLE | COMPOUND | SEMI_STABLE | UNSTABLE | FOREIGN\n")

    for i, prim_def in enumerate(ALL_PRIMITIVES):
        prim_id   = prim_def["id"]
        primitive = prim_def["primitive"]
        ptype     = "SEMANTIC"  if prim_id.startswith("S") else "EMOTIONAL"
        category  = prim_def.get("category", prim_def.get("valence", ""))
        desc      = prim_def["description"]

        print(f"  [{i+1:02d}/{total}] {prim_id} {primitive:<15}", end=" ", flush=True)

        trials = []
        sample_responses = []
        for _ in range(TRIALS_PER_PRIMITIVE):
            trial = probe_primitive(primitive, host, model)
            trials.append(trial)
            if not trial.get("parse_failed") and trial["response"]:
                sample_responses.append(trial["response"][:60])
            time.sleep(0.3)

        scores = score_stability(trials)
        classification = classify_primitive(
            scores["stability_score"],
            scores["is_compound"],
            scores["avg_confidence"],
            scores["parse_fail_rate"],
        )

        pr = PrimitiveResult(
            primitive_id=prim_id,
            primitive=primitive,
            primitive_type=ptype,
            category=category,
            description=desc,
            stability_score=scores["stability_score"],
            modal_dominant=scores["modal_dominant"],
            dominant_agreement=scores["dominant_agreement"],
            expansion_cluster=", ".join(scores["expansion_cluster"]),
            classification=classification,
            is_compound=scores["is_compound"],
            avg_confidence=scores["avg_confidence"],
            parse_fail_rate=scores["parse_fail_rate"],
            all_dominants=", ".join(scores["all_dominants"]),
            trial_responses=" | ".join(sample_responses[:3]),
        )
        results.append(pr)

        print(f"[{classification:<11}] "
              f"stab:{pr.stability_score:.2f}  "
              f"conf:{pr.avg_confidence:.2f}  "
              f"→ {pr.modal_dominant}")

    report(results)
    save_results(results)
    return results

# ── reporting ─────────────────────────────────────────────────────────────────

def report(results: list[PrimitiveResult]):
    semantic  = [r for r in results if r.primitive_type == "SEMANTIC"]
    emotional = [r for r in results if r.primitive_type == "EMOTIONAL"]

    print(f"\n── PRIMITIVE VALIDATION RESULTS ─────────────────────────────────")

    for group_name, group in [("SEMANTIC PRIMITIVES", semantic),
                               ("EMOTIONAL PRIMITIVES", emotional)]:
        print(f"\n  {group_name}")
        print(f"  {'ID':<5} {'Primitive':<15} {'Class':<12} "
              f"{'Stab':>5} {'Conf':>5}  Modal dominant")
        print(f"  {'──':<5} {'─────────':<15} {'─────':<12} "
              f"{'────':>5} {'────':>5}  ──────────────")
        for r in group:
            print(f"  {r.primitive_id:<5} {r.primitive:<15} "
                  f"{r.classification:<12} "
                  f"{r.stability_score:>5.2f} {r.avg_confidence:>5.2f}  "
                  f"{r.modal_dominant}")

    print(f"\n── CLASSIFICATION SUMMARY ───────────────────────────────────────")
    all_classes = Counter(r.classification for r in results)
    for cls, count in all_classes.most_common():
        bar = "█" * count
        print(f"  {cls:<12} {bar} {count}/{len(results)}")

    stable   = [r for r in results if r.classification == "STABLE"]
    compound = [r for r in results if r.classification == "COMPOUND"]
    foreign  = [r for r in results if r.classification == "FOREIGN"]

    print(f"\n── STABLE PRIMITIVES — VALID SRM LAYER 0/1 CANDIDATES ──────────")
    if stable:
        for r in sorted(stable, key=lambda x: x.stability_score, reverse=True):
            print(f"  {r.primitive_id} {r.primitive:<15} "
                  f"stab:{r.stability_score:.2f}  "
                  f"expands→ {r.expansion_cluster[:50]}")
    else:
        print("  None found at this threshold — lower STABILITY_THRESHOLD?")

    print(f"\n── COMPOUND PRIMITIVES — NEED DECOMPOSITION ────────────────────")
    if compound:
        for r in compound:
            print(f"  {r.primitive_id} {r.primitive:<15} "
                  f"unpacks→ {r.all_dominants[:60]}")
    else:
        print("  None — all tested primitives appear atomic to the membrane")

    print(f"\n── FOREIGN PRIMITIVES — OUTSIDE MEMBRANE SPACE ────────────────")
    if foreign:
        for r in foreign:
            print(f"  {r.primitive_id} {r.primitive:<15} "
                  f"parse_fail_rate:{r.parse_fail_rate:.2f}")
    else:
        print("  None — membrane can process all tested primitives")

    print(f"\n── EXPANSION CLUSTERS — NATURAL PRIMITIVE COMBINATIONS ─────────")
    print(f"  These are the words primitives NATURALLY expand into.")
    print(f"  Dense expansion clusters = likely Layer 2 concepts.")
    all_expansions: list[str] = []
    for r in results:
        if r.expansion_cluster:
            all_expansions.extend(r.expansion_cluster.split(", "))
    expansion_freq = Counter(all_expansions).most_common(15)
    for word, count in expansion_freq:
        if word:
            print(f"  {word:<20} appears in {count} primitive expansions")

    print(f"\n── HYPOTHESIS VERDICT ───────────────────────────────────────────")
    stable_rate = len(stable) / len(results)
    if stable_rate >= 0.6:
        print(f"  PRIMITIVES VALIDATED — {stable_rate:.0%} are stable")
        print(f"  The membrane recognizes universal primitives consistently.")
        print(f"  SRM symbolic layer can be built on this foundation.")
    elif stable_rate >= 0.3:
        print(f"  PRIMITIVES PARTIALLY VALIDATED — {stable_rate:.0%} stable")
        print(f"  Core primitives exist but subset needs refinement.")
        print(f"  Compound primitives suggest decomposition is needed.")
    else:
        print(f"  PRIMITIVES NOT VALIDATED at this model size.")
        print(f"  Recommend: run experiment 17 with larger model.")
        print(f"  The primitive layer may require a stronger membrane.")

    print(f"\n── NEXT STEPS ───────────────────────────────────────────────────")
    print(f"  Stable primitives → Layer 0/1 of SRM symbolic layer")
    print(f"  Compound primitives → decompose into sub-primitives")
    print(f"  Expansion clusters → candidate Layer 2 concepts")
    print(f"  Run experiment 18: combine stable primitives in pairs")
    print(f"  and measure whether combinations produce Layer 2 concepts")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(results: list[PrimitiveResult],
                 base="semantic_primitives/results_exp_17"):
    Path("semantic_primitives").mkdir(exist_ok=True)

    json_path = Path(f"{base}.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    csv_path = Path(f"{base}.csv")
    fieldnames = list(asdict(results[0]).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # also save a clean summary for the research doc
    summary_path = Path("semantic_primitives/primitive_summary.json")
    summary = {
        "experiment": "exp_17",
        "timestamp": datetime.now().isoformat(),
        "model": "qwen2.5:0.5b",
        "stable":    [r.primitive for r in results if r.classification == "STABLE"],
        "compound":  [r.primitive for r in results if r.classification == "COMPOUND"],
        "foreign":   [r.primitive for r in results if r.classification == "FOREIGN"],
        "unstable":  [r.primitive for r in results if r.classification in
                      ["UNSTABLE", "SEMI_STABLE"]],
        "expansion_candidates": [],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n── SAVED ────────────────────────────────────────────────────────")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  semantic_primitives/primitive_summary.json")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    host  = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:0.5b"
    run_experiment(host, model)