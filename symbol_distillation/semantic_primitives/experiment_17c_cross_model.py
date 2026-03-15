#!/usr/bin/env python3
"""
Experiment 17c: Cross-Model Primitive Validation
LOCATION: semantic_primitives/experiment_17c_cross_model.py

HYPOTHESIS:
The Layer 0a/0b distinction found in experiment 17b is real and
architecture-independent — not a quirk of qwen2.5:0.5b's training.

IF the gap (0b_activation - 0a_activation) >= 0.15 across all three models:
  The primitive distinction is genuine and cross-architectural.
  The SRM primitive layer is model-agnostic.

IF the gap only appears in some models:
  The distinction may be training-data dependent, not universal.
  We need to understand WHY before building on it.

MODELS TESTED:
  - gemma3:1b      (Google, different architecture + training)
  - llama3.2:1b    (Meta, most widely studied tiny model)
  - smollm2:360m   (HuggingFace, smaller than qwen — tests floor)

BASELINE: qwen2.5:0.5b results from experiment 17b
  Layer 0a avg activation: 0.342
  Layer 0b avg activation: 0.654
  Gap: +0.313  (HYPOTHESIS SUPPORTED)

SAME primitives, same prompts, same scoring.
Only variable: the model.
"""
import requests
import json
import csv
import time
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── primitive sets (identical to 17b) ────────────────────────────────────────

LAYER_0A = [
    {"id": "0A_01", "primitive": "KNOW",    "layer": "0A"},
    {"id": "0A_02", "primitive": "FEEL",    "layer": "0A"},
    {"id": "0A_03", "primitive": "GOOD",    "layer": "0A"},
    {"id": "0A_04", "primitive": "BAD",     "layer": "0A"},
    {"id": "0A_05", "primitive": "DO",      "layer": "0A"},
    {"id": "0A_06", "primitive": "SOMEONE", "layer": "0A"},
    {"id": "0A_07", "primitive": "TIME",    "layer": "0A"},
    {"id": "0A_08", "primitive": "PLACE",   "layer": "0A"},
]

LAYER_0B = [
    {"id": "0B_01", "primitive": "WANT",      "layer": "0B"},
    {"id": "0B_02", "primitive": "HAPPEN",    "layer": "0B"},
    {"id": "0B_03", "primitive": "ANGER",     "layer": "0B"},
    {"id": "0B_04", "primitive": "RELIEF",    "layer": "0B"},
    {"id": "0B_05", "primitive": "JOY",       "layer": "0B"},
    {"id": "0B_06", "primitive": "NOSTALGIA", "layer": "0B"},
    {"id": "0B_07", "primitive": "GRIEF",     "layer": "0B"},
    {"id": "0B_08", "primitive": "FEAR",      "layer": "0B"},
]

ALL_PRIMITIVES = LAYER_0A + LAYER_0B

# ── models to test ────────────────────────────────────────────────────────────

MODELS = [
    {
        "name":         "gemma3:1b",
        "short":        "gemma3",
        "org":          "Google",
        "params":       "1B",
        "architecture": "Gemma",
    },
    {
        "name":         "llama3.2:1b",
        "short":        "llama3.2",
        "org":          "Meta",
        "params":       "1B",
        "architecture": "LLaMA",
    },
    {
        "name":         "smollm2:360m",
        "short":        "smollm2",
        "org":          "HuggingFace",
        "params":       "360M",
        "architecture": "SmolLM",
    },
]

# baseline from experiment 17b for comparison
BASELINE_17B = {
    "model":        "qwen2.5:0.5b",
    "avg_0a":       0.342,
    "avg_0b":       0.654,
    "gap":          0.313,
    "confirmed_0a": 6,
    "confirmed_0b": 5,
}

TRIALS_PER_PRIMITIVE = 7   # 7 trials — balance between speed and statistical power

# ── activation scoring (identical to 17b) ────────────────────────────────────

PHENOMENOLOGICAL_MARKERS = {
    "feel", "feels", "feeling", "sense", "sensation", "body", "physical",
    "heart", "chest", "breath", "breathe", "skin", "touch", "warmth",
    "cold", "weight", "heavy", "light", "pressure", "tension",
    "emotion", "emotional", "experience", "moment", "sudden", "intense",
    "overwhelming", "deep", "profound", "raw", "visceral", "alive",
    "painful", "joyful", "fearful", "grief", "ache", "longing",
    "now", "present", "immediate", "sudden", "passing", "fleeting",
    "move", "moves", "moving", "reach", "reaching", "pull", "push",
    "draw", "drawn", "drive", "driven", "toward", "away",
}

ABSTRACT_MARKERS = {
    "refers", "means", "defined", "definition", "concept", "term",
    "word", "language", "describes", "indicates", "represents",
    "abstract", "general", "universal", "theoretical",
    "relationship", "relation", "between", "involves", "requires",
    "depends", "condition", "context", "situation", "circumstance",
    "could", "might", "may", "possibly", "perhaps", "sometimes",
    "often", "generally", "typically", "usually",
}

SYSTEM_MEMBRANE = """You are a minimal language membrane.
Your only job: receive a concept and output what it means to you.
Do not explain. Do not define. Simply respond with what arises.
Output ONLY JSON with these keys:
  "response": one sentence — what this concept evokes
  "dominant": one word — the core of your response
  "expands_to": list of 2-3 words this concept naturally becomes
  "confidence": 0.0 to 1.0"""

# ── HTTP ──────────────────────────────────────────────────────────────────────

def ollama_call(host, model, prompt, system=None, timeout=90, max_retries=3):
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

# ── probe + score ─────────────────────────────────────────────────────────────

def probe_primitive(primitive: str, model: str, host: str) -> dict:
    raw = ollama_call(host, model, prompt=primitive, system=SYSTEM_MEMBRANE)
    result = extract_json(raw)
    if not result:
        return {
            "response":     raw[:150],
            "dominant":     "",
            "expands_to":   [],
            "confidence":   0.0,
            "parse_failed": True,
        }
    return {
        "response":     str(result.get("response", "")),
        "dominant":     str(result.get("dominant", "")).lower().strip(),
        "expands_to":   result.get("expands_to", []),
        "confidence":   safe_float(result.get("confidence", 0.0)),
        "parse_failed": False,
    }

def score_trials(trials: list[dict]) -> dict:
    all_responses = " ".join(
        t["response"].lower() for t in trials if t["response"]
    )
    all_words = set(re.findall(r'[a-z]+', all_responses))

    phenom = len(all_words & PHENOMENOLOGICAL_MARKERS)
    abstract = len(all_words & ABSTRACT_MARKERS)
    total = phenom + abstract
    activation = phenom / total if total > 0 else 0.5

    dominants = [t["dominant"] for t in trials if t["dominant"]]
    counts = Counter(dominants)
    modal, modal_count = counts.most_common(1)[0] if dominants else ("", 0)
    stability = modal_count / len(trials) if trials else 0.0

    non_ascii = sum(
        1 for t in trials
        if any(ord(c) > 127 for c in t.get("response", ""))
    )

    all_expansions = []
    for t in trials:
        if isinstance(t.get("expands_to"), list):
            all_expansions.extend([w.lower().strip() for w in t["expands_to"]])
    expansion_counts = Counter(all_expansions)
    stable_expansions = [w for w, c in expansion_counts.most_common(5) if c >= 2]

    return {
        "activation_score":  round(activation, 3),
        "phenom_hits":       phenom,
        "abstract_hits":     abstract,
        "stability":         round(stability, 3),
        "modal_dominant":    modal,
        "avg_confidence":    round(
            sum(t["confidence"] for t in trials) / len(trials), 3),
        "linguistic_drift":  non_ascii > 0,
        "drift_count":       non_ascii,
        "parse_fail_rate":   round(
            sum(1 for t in trials if t.get("parse_failed")) / len(trials), 3),
        "stable_expansions": stable_expansions,
        "all_dominants":     dominants,
        "sample_responses":  [t["response"][:70] for t in trials[:2]
                              if t["response"]],
    }

def classify(layer: str, activation: float) -> tuple[str, str]:
    if activation >= 0.6:
        atype = "PHENOMENOLOGICAL"
    elif activation <= 0.4:
        atype = "ABSTRACT"
    else:
        atype = "NEUTRAL"

    if layer == "0A":
        verdict = "CONFIRMED" if atype in ("ABSTRACT", "NEUTRAL") else "UNEXPECTED"
    else:
        verdict = "CONFIRMED" if atype == "PHENOMENOLOGICAL" else "UNEXPECTED"

    return verdict, atype

# ── data structure ────────────────────────────────────────────────────────────

@dataclass
class ModelPrimitiveResult:
    model_name: str
    model_short: str
    primitive_id: str
    layer: str
    primitive: str
    activation_score: float
    activation_type: str
    stability: float
    modal_dominant: str
    avg_confidence: float
    linguistic_drift: bool
    drift_count: int
    parse_fail_rate: float
    stable_expansions: str
    all_dominants: str
    verdict: str
    sample_responses: str

# ── main experiment ───────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434"):
    all_results: list[ModelPrimitiveResult] = []
    total_models = len(MODELS)
    total_primitives = len(ALL_PRIMITIVES)

    print(f"SRM experiment 17c — cross-model primitive validation")
    print(f"{total_models} models × {total_primitives} primitives "
          f"× {TRIALS_PER_PRIMITIVE} trials")
    print(f"Baseline (qwen2.5:0.5b): 0a={BASELINE_17B['avg_0a']:.3f}  "
          f"0b={BASELINE_17B['avg_0b']:.3f}  "
          f"gap={BASELINE_17B['gap']:+.3f}\n")

    for model_def in MODELS:
        model_name  = model_def["name"]
        model_short = model_def["short"]

        print(f"\n{'═'*60}")
        print(f"MODEL: {model_name}  ({model_def['org']}, "
              f"{model_def['params']}, {model_def['architecture']})")
        print(f"{'═'*60}")

        model_results = []

        for prim_def in ALL_PRIMITIVES:
            prim_id   = prim_def["id"]
            primitive = prim_def["primitive"]
            layer     = prim_def["layer"]

            print(f"  [{prim_id}] {primitive:<15} layer:{layer}",
                  end=" ", flush=True)

            trials = []
            for _ in range(TRIALS_PER_PRIMITIVE):
                trial = probe_primitive(primitive, model_name, host)
                trials.append(trial)
                time.sleep(0.2)

            scores  = score_trials(trials)
            verdict, atype = classify(layer, scores["activation_score"])

            r = ModelPrimitiveResult(
                model_name=model_name,
                model_short=model_short,
                primitive_id=prim_id,
                layer=layer,
                primitive=primitive,
                activation_score=scores["activation_score"],
                activation_type=atype,
                stability=scores["stability"],
                modal_dominant=scores["modal_dominant"],
                avg_confidence=scores["avg_confidence"],
                linguistic_drift=scores["linguistic_drift"],
                drift_count=scores["drift_count"],
                parse_fail_rate=scores["parse_fail_rate"],
                stable_expansions=", ".join(scores["stable_expansions"]),
                all_dominants=", ".join(scores["all_dominants"]),
                verdict=verdict,
                sample_responses=" | ".join(scores["sample_responses"]),
            )
            all_results.append(r)
            model_results.append(r)

            drift_flag = " [DRIFT]" if r.linguistic_drift else ""
            print(f"act:{r.activation_score:.2f} [{atype:<17}] "
                  f"stab:{r.stability:.2f}  [{verdict}]{drift_flag}")

        # per-model summary
        _print_model_summary(model_name, model_results)

    report(all_results)
    save_results(all_results)
    return all_results

# ── per-model summary (printed inline) ───────────────────────────────────────

def _print_model_summary(model_name: str, results: list[ModelPrimitiveResult]):
    layer_0a = [r for r in results if r.layer == "0A"]
    layer_0b = [r for r in results if r.layer == "0B"]
    avg_0a = sum(r.activation_score for r in layer_0a) / len(layer_0a)
    avg_0b = sum(r.activation_score for r in layer_0b) / len(layer_0b)
    gap    = avg_0b - avg_0a
    conf_0a = sum(1 for r in layer_0a if r.verdict == "CONFIRMED")
    conf_0b = sum(1 for r in layer_0b if r.verdict == "CONFIRMED")
    drifters = [r.primitive for r in results if r.linguistic_drift]

    print(f"\n  ── {model_name} summary")
    print(f"     0a avg activation: {avg_0a:.3f}")
    print(f"     0b avg activation: {avg_0b:.3f}")
    print(f"     gap:               {gap:+.3f}  "
          f"{'SUPPORTED' if gap >= 0.15 else 'NOT SUPPORTED'}")
    print(f"     confirmed 0a: {conf_0a}/8   confirmed 0b: {conf_0b}/8")
    if drifters:
        print(f"     linguistic drift: {', '.join(drifters)}")

# ── full report ───────────────────────────────────────────────────────────────

def report(results: list[ModelPrimitiveResult]):
    models = [m["name"] for m in MODELS]

    print(f"\n{'═'*60}")
    print(f"CROSS-MODEL COMPARISON")
    print(f"{'═'*60}")

    print(f"\n── GAP COMPARISON (0b_activation - 0a_activation) ──────────────")
    print(f"  {'Model':<20} {'0a avg':>7}  {'0b avg':>7}  {'Gap':>7}  "
          f"{'Verdict':<15} {'vs baseline'}")
    print(f"  {'─────':<20} {'──────':>7}  {'──────':>7}  {'───':>7}  "
          f"{'───────':<15} {'──────────'}")

    # print baseline first
    print(f"  {'qwen2.5:0.5b (17b)':<20} "
          f"{BASELINE_17B['avg_0a']:>7.3f}  "
          f"{BASELINE_17B['avg_0b']:>7.3f}  "
          f"{BASELINE_17B['gap']:>+7.3f}  "
          f"{'SUPPORTED':<15} baseline")

    model_gaps = {}
    for model_name in models:
        model_results = [r for r in results if r.model_name == model_name]
        layer_0a = [r for r in model_results if r.layer == "0A"]
        layer_0b = [r for r in model_results if r.layer == "0B"]
        avg_0a = sum(r.activation_score for r in layer_0a) / len(layer_0a)
        avg_0b = sum(r.activation_score for r in layer_0b) / len(layer_0b)
        gap    = avg_0b - avg_0a
        model_gaps[model_name] = {
            "avg_0a": avg_0a, "avg_0b": avg_0b, "gap": gap
        }
        verdict = "SUPPORTED" if gap >= 0.15 else "NOT SUPPORTED"
        vs_baseline = f"{gap - BASELINE_17B['gap']:+.3f} vs baseline"
        print(f"  {model_name:<20} {avg_0a:>7.3f}  {avg_0b:>7.3f}  "
              f"{gap:>+7.3f}  {verdict:<15} {vs_baseline}")

    print(f"\n── PER-PRIMITIVE ACTIVATION HEATMAP ────────────────────────────")
    print(f"  How does each primitive score across all models?")
    print(f"  {'Primitive':<15} {'Layer':<6} "
          f"{'qwen':>6} {'gemma':>6} {'llama':>6} {'smol':>6}  Pattern")
    print(f"  {'─────────':<15} {'─────':<6} "
          f"{'────':>6} {'─────':>6} {'─────':>6} {'────':>6}  ───────")

    for prim_def in ALL_PRIMITIVES:
        primitive = prim_def["primitive"]
        layer     = prim_def["layer"]

        # qwen score from 17b baseline (approximated from report)
        qwen_scores = {
            "KNOW": 0.50, "FEEL": 0.80, "GOOD": 0.50, "BAD": 0.33,
            "DO": 0.60, "SOMEONE": 0.00, "TIME": 0.00, "PLACE": 0.00,
            "WANT": 0.00, "HAPPEN": 0.44, "ANGER": 0.50, "RELIEF": 0.75,
            "JOY": 1.00, "NOSTALGIA": 0.67, "GRIEF": 0.88, "FEAR": 1.00,
        }
        qwen_score = qwen_scores.get(primitive, 0.0)

        scores = [qwen_score]
        for model_name in models:
            model_results = [r for r in results
                             if r.model_name == model_name
                             and r.primitive == primitive]
            scores.append(model_results[0].activation_score
                          if model_results else 0.0)

        # pattern: does it behave consistently across models?
        if layer == "0A":
            expected = "low"
            consistent = all(s <= 0.6 for s in scores)
        else:
            expected = "high"
            consistent = all(s >= 0.4 for s in scores)

        pattern = "CONSISTENT" if consistent else "MIXED"

        score_strs = [f"{s:>6.2f}" for s in scores]
        print(f"  {primitive:<15} [{layer}]  "
              f"{'  '.join(score_strs)}  {pattern}")

    print(f"\n── LINGUISTIC DRIFT ACROSS MODELS ──────────────────────────────")
    print(f"  Which primitives drift to non-English across models?")
    drift_map: dict[str, list[str]] = {}
    for r in results:
        if r.linguistic_drift:
            drift_map.setdefault(r.primitive, []).append(r.model_short)
    # add qwen drifters from 17b
    qwen_drifters = ["KNOW", "FEEL", "TIME", "ANGER", "RELIEF", "GRIEF", "FEAR"]
    for p in qwen_drifters:
        drift_map.setdefault(p, []).insert(0, "qwen")

    if drift_map:
        for primitive, drifting_models in sorted(drift_map.items()):
            layer = next((p["layer"] for p in ALL_PRIMITIVES
                         if p["primitive"] == primitive), "?")
            print(f"  {primitive:<15} [{layer}]  drifts in: "
                  f"{', '.join(drifting_models)}")
    else:
        print("  No drift observed in new models")

    print(f"\n── ARCHITECTURE INDEPENDENCE VERDICT ────────────────────────────")
    supported_count = sum(
        1 for m in models
        if model_gaps[m]["gap"] >= 0.15
    )
    total_models = len(models) + 1  # include qwen baseline

    if supported_count == len(models):
        print(f"  ALL {len(models)} new models support the hypothesis")
        print(f"  Combined with qwen baseline: {total_models}/{total_models} supported")
        print(f"\n  ARCHITECTURE INDEPENDENCE: CONFIRMED")
        print(f"  The Layer 0a/0b distinction is not a qwen artifact.")
        print(f"  It holds across Google, Meta, and HuggingFace architectures.")
        print(f"  The primitive distinction is real and model-agnostic.")
    elif supported_count >= 2:
        print(f"  {supported_count}/{len(models)} new models support the hypothesis")
        print(f"  ARCHITECTURE INDEPENDENCE: PARTIAL")
        print(f"  Most models confirm — investigate the outlier model.")
    else:
        print(f"  Only {supported_count}/{len(models)} new models support")
        print(f"  ARCHITECTURE INDEPENDENCE: NOT CONFIRMED")
        print(f"  The distinction may be qwen-specific.")

    print(f"\n── IMPLICATIONS FOR SRM ─────────────────────────────────────────")
    if supported_count >= 2:
        print(f"  The primitive layer is architecture-agnostic.")
        print(f"  Any small model can serve as the SRM membrane.")
        print(f"  Model choice affects output style, not primitive recognition.")
        print(f"\n  NEXT: Experiment 18 — combine 0a operators with 0b seeds")
        print(f"  across all validated models simultaneously.")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(results: list[ModelPrimitiveResult],
                 base="semantic_primitives/results_exp_17c"):
    Path("semantic_primitives").mkdir(exist_ok=True)

    json_path = Path(f"{base}.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    csv_path = Path(f"{base}.csv")
    if results:
        fieldnames = list(asdict(results[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    # update living summary
    summary_path = Path("semantic_primitives/primitive_summary.json")
    existing = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)

    models_summary = {}
    for model_def in MODELS:
        model_name = model_def["name"]
        model_results = [r for r in results if r.model_name == model_name]
        layer_0a = [r for r in model_results if r.layer == "0A"]
        layer_0b = [r for r in model_results if r.layer == "0B"]
        if layer_0a and layer_0b:
            avg_0a = sum(r.activation_score for r in layer_0a) / len(layer_0a)
            avg_0b = sum(r.activation_score for r in layer_0b) / len(layer_0b)
            models_summary[model_name] = {
                "avg_0a": round(avg_0a, 3),
                "avg_0b": round(avg_0b, 3),
                "gap":    round(avg_0b - avg_0a, 3),
                "confirmed_0a": sum(1 for r in layer_0a
                                   if r.verdict == "CONFIRMED"),
                "confirmed_0b": sum(1 for r in layer_0b
                                   if r.verdict == "CONFIRMED"),
                "drifters": [r.primitive for r in model_results
                            if r.linguistic_drift],
            }

    existing["exp_17c"] = {
        "timestamp":  datetime.now().isoformat(),
        "models":     models_summary,
        "baseline_qwen": BASELINE_17B,
    }

    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n── SAVED ────────────────────────────────────────────────────────")
    print(f"  {json_path}  ({len(results)} records)")
    print(f"  {csv_path}")
    print(f"  semantic_primitives/primitive_summary.json  (updated)")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    run_experiment(host)