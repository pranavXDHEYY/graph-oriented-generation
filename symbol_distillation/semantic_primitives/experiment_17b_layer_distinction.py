#!/usr/bin/env python3
"""
Experiment 17b: Layer 0a vs Layer 0b Primitive Distinction
LOCATION: semantic_primitives/experiment_17b_layer_distinction.py

HYPOTHESIS:
Wierzbicka's primitives split into two functionally distinct sublayers:

  Layer 0a — SCAFFOLDING primitives
    Organize and relate meaning but don't carry it in isolation.
    Expected membrane behavior: weak activation, abstract/meta responses,
    low confidence, inconsistent dominants, may describe the word itself.
    Examples: KNOW, FEEL, GOOD, BAD, TIME, PLACE, DO, SOMEONE

  Layer 0b — CONTENT primitives  
    Carry phenomenological meaning that the membrane activates on directly.
    Expected membrane behavior: strong activation, specific responses,
    high confidence, consistent dominants, emotional or sensory texture.
    Examples: WANT, HAPPEN, RELIEF, ANGER, JOY, NOSTALGIA

IF THE HYPOTHESIS IS CORRECT:
  0a primitives → low activation score (abstract, scaffolding)
  0b primitives → high activation score (content, phenomenological)
  The gap between them validates the two-layer model.

IF THE HYPOTHESIS IS WRONG:
  No consistent difference between 0a and 0b activation scores.
  We need to rethink the primitive taxonomy entirely.

METHODOLOGY:
  10 trials per primitive (up from 5 — more statistical power)
  Two scoring dimensions:
    1. Consistency (same as exp 17)
    2. Activation strength — is the response phenomenological or abstract?
  Activation strength distinguishes the two sublayers.
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

# ── Layer 0a: scaffolding primitives ─────────────────────────────────────────
# Hypothesis: organize meaning, weak membrane activation in isolation
# These are the GRAMMAR of the symbolic layer

LAYER_0A = [
    {"id": "0A_01", "primitive": "KNOW",    "why_scaffolding": "relational — always KNOW something"},
    {"id": "0A_02", "primitive": "FEEL",    "why_scaffolding": "relational — always FEEL something"},
    {"id": "0A_03", "primitive": "GOOD",    "why_scaffolding": "evaluative — needs referent to evaluate"},
    {"id": "0A_04", "primitive": "BAD",     "why_scaffolding": "evaluative — needs referent to evaluate"},
    {"id": "0A_05", "primitive": "DO",      "why_scaffolding": "agentive — always DO something"},
    {"id": "0A_06", "primitive": "SOMEONE", "why_scaffolding": "referential — points to entity, not content"},
    {"id": "0A_07", "primitive": "TIME",    "why_scaffolding": "structural — organizes sequence, no texture"},
    {"id": "0A_08", "primitive": "PLACE",   "why_scaffolding": "structural — organizes location, no texture"},
]

# ── Layer 0b: content primitives ─────────────────────────────────────────────
# Hypothesis: carry phenomenological content, strong membrane activation
# These are the VOCABULARY of the symbolic layer

LAYER_0B = [
    {"id": "0B_01", "primitive": "WANT",      "why_content": "directional drive — has phenomenological texture"},
    {"id": "0B_02", "primitive": "HAPPEN",    "why_content": "event-ness — the quality of occurrence itself"},
    {"id": "0B_03", "primitive": "ANGER",     "why_content": "full emotional phenomenology, no referent needed"},
    {"id": "0B_04", "primitive": "RELIEF",    "why_content": "directional release — tension → resolution"},
    {"id": "0B_05", "primitive": "JOY",       "why_content": "full emotional phenomenology, self-contained"},
    {"id": "0B_06", "primitive": "NOSTALGIA", "why_content": "complex temporal-emotional texture"},
    {"id": "0B_07", "primitive": "GRIEF",     "why_content": "full emotional phenomenology — loss + time"},
    {"id": "0B_08", "primitive": "FEAR",      "why_content": "survival phenomenology — threat texture"},
]

TRIALS_PER_PRIMITIVE = 10  # up from 5 — more statistical power
STABILITY_THRESHOLD  = 0.5  # adjusted for n=10

# ── activation strength indicators ───────────────────────────────────────────
# These word lists detect phenomenological vs abstract responses

PHENOMENOLOGICAL_MARKERS = {
    # sensory / bodily
    "feel", "feels", "feeling", "sense", "sensation", "body", "physical",
    "heart", "chest", "breath", "breathe", "skin", "touch", "warmth",
    "cold", "weight", "heavy", "light", "pressure", "tension",
    # experiential / emotional
    "emotion", "emotional", "experience", "moment", "sudden", "intense",
    "overwhelming", "deep", "profound", "raw", "visceral", "alive",
    "painful", "joyful", "fearful", "grief", "ache", "longing",
    # temporal presence
    "now", "present", "immediate", "sudden", "passing", "fleeting",
    # action / movement
    "move", "moves", "moving", "reach", "reaching", "pull", "push",
    "draw", "drawn", "drive", "driven", "toward", "away",
}

ABSTRACT_MARKERS = {
    # definitional / meta
    "refers", "means", "defined", "definition", "concept", "term",
    "word", "language", "describes", "indicates", "represents",
    "abstract", "general", "universal", "theoretical",
    # relational / logical
    "relationship", "relation", "between", "involves", "requires",
    "depends", "condition", "context", "situation", "circumstance",
    # hedging
    "could", "might", "may", "possibly", "perhaps", "sometimes",
    "often", "generally", "typically", "usually",
}

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
  "expands_to": list of 2-3 words this concept naturally becomes
  "confidence": 0.0 to 1.0"""

def probe_primitive(primitive: str, host: str, model: str) -> dict:
    raw = ollama_call(
        host, model,
        prompt=primitive,
        system=SYSTEM_MEMBRANE
    )
    result = extract_json(raw)
    if not result:
        return {
            "response":     raw[:150],
            "dominant":     "",
            "expands_to":   [],
            "confidence":   0.0,
            "parse_failed": True,
            "raw":          raw[:200],
        }
    return {
        "response":     str(result.get("response", "")),
        "dominant":     str(result.get("dominant", "")).lower().strip(),
        "expands_to":   result.get("expands_to", []),
        "confidence":   safe_float(result.get("confidence", 0.0)),
        "parse_failed": False,
        "raw":          raw[:200],
    }

# ── activation strength scoring ───────────────────────────────────────────────

def score_activation_strength(trials: list[dict]) -> dict:
    """
    Measures whether the membrane response is phenomenological (content)
    or abstract (scaffolding).

    activation_score: 0.0 = purely abstract, 1.0 = purely phenomenological
    This is the key metric that distinguishes 0a from 0b.
    """
    all_responses = " ".join(
        t["response"].lower() for t in trials if t["response"]
    )
    all_words = set(re.findall(r'[a-z]+', all_responses))

    phenom_hits = len(all_words & PHENOMENOLOGICAL_MARKERS)
    abstract_hits = len(all_words & ABSTRACT_MARKERS)
    total_hits = phenom_hits + abstract_hits

    if total_hits == 0:
        activation_score = 0.5  # neutral — can't determine
    else:
        activation_score = phenom_hits / total_hits

    # also check for linguistic drift (Mandarin etc)
    non_ascii = sum(
        1 for t in trials
        if any(ord(c) > 127 for c in t.get("response", ""))
    )
    linguistic_drift = non_ascii > 0

    # dominant stability
    dominants = [t["dominant"] for t in trials if t["dominant"]]
    dominant_counts = Counter(dominants)
    modal, modal_count = dominant_counts.most_common(1)[0] if dominants else ("", 0)
    stability = modal_count / len(trials) if trials else 0.0

    # average confidence
    avg_conf = sum(t["confidence"] for t in trials) / len(trials)

    # parse failure rate
    parse_fails = sum(1 for t in trials if t.get("parse_failed", False))

    # collect all expansion words
    all_expansions = []
    for t in trials:
        if isinstance(t.get("expands_to"), list):
            all_expansions.extend([w.lower().strip() for w in t["expands_to"]])
    expansion_counts = Counter(all_expansions)
    stable_expansions = [w for w, c in expansion_counts.most_common(5) if c >= 2]

    return {
        "activation_score":   round(activation_score, 3),
        "phenom_hits":        phenom_hits,
        "abstract_hits":      abstract_hits,
        "stability":          round(stability, 3),
        "modal_dominant":     modal,
        "modal_count":        modal_count,
        "avg_confidence":     round(avg_conf, 3),
        "linguistic_drift":   linguistic_drift,
        "parse_fail_rate":    round(parse_fails / len(trials), 3),
        "stable_expansions":  stable_expansions,
        "all_dominants":      dominants,
        "sample_responses":   [t["response"][:80] for t in trials[:3]
                               if t["response"]],
    }

# ── classification ────────────────────────────────────────────────────────────

def classify_primitive(layer: str, activation: float,
                       stability: float, avg_conf: float) -> str:
    """
    Classifies each primitive and whether it behaves as expected for its layer.
    """
    # activation score interpretation
    if activation >= 0.6:
        activation_type = "PHENOMENOLOGICAL"
    elif activation <= 0.4:
        activation_type = "ABSTRACT"
    else:
        activation_type = "NEUTRAL"

    # stability classification
    if stability >= 0.6:
        stability_class = "STABLE"
    elif stability >= 0.4:
        stability_class = "SEMI_STABLE"
    else:
        stability_class = "UNSTABLE"

    # hypothesis check
    if layer == "0A":
        confirmed = activation_type in ("ABSTRACT", "NEUTRAL")
        verdict = "CONFIRMED_0A" if confirmed else "UNEXPECTED_0A"
    else:
        confirmed = activation_type == "PHENOMENOLOGICAL"
        verdict = "CONFIRMED_0B" if confirmed else "UNEXPECTED_0B"

    return verdict, activation_type, stability_class

# ── data structure ────────────────────────────────────────────────────────────

@dataclass
class PrimitiveResult17b:
    primitive_id: str
    layer: str                   # "0A" or "0B"
    primitive: str
    hypothesis: str              # why we expect it in this layer
    # activation scoring
    activation_score: float      # KEY: phenomological vs abstract
    phenom_hits: int
    abstract_hits: int
    activation_type: str         # PHENOMENOLOGICAL / ABSTRACT / NEUTRAL
    # stability scoring
    stability: float
    stability_class: str
    modal_dominant: str
    modal_count: int
    avg_confidence: float
    # additional signals
    linguistic_drift: bool
    parse_fail_rate: float
    stable_expansions: str
    all_dominants: str
    # verdict
    verdict: str                 # CONFIRMED_0A/0B or UNEXPECTED_0A/0B
    sample_responses: str

# ── main experiment ───────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434", model="qwen2.5:0.5b"):
    results: list[PrimitiveResult17b] = []
    all_primitives = (
        [(p, "0A") for p in LAYER_0A] +
        [(p, "0B") for p in LAYER_0B]
    )
    total = len(all_primitives)

    print(f"SRM experiment 17b — Layer 0a vs 0b distinction")
    print(f"Testing {total} primitives × {TRIALS_PER_PRIMITIVE} trials")
    print(f"8 scaffolding (0a) + 8 content (0b)\n")
    print(f"Key metric: activation_score")
    print(f"  0.0 = purely abstract (expected for 0a)")
    print(f"  1.0 = purely phenomenological (expected for 0b)\n")

    for prim_def, layer in all_primitives:
        prim_id   = prim_def["id"]
        primitive = prim_def["primitive"]
        hypothesis = prim_def.get("why_scaffolding",
                                   prim_def.get("why_content", ""))

        print(f"  [{prim_id}] {primitive:<15} layer:{layer}",
              end=" ", flush=True)

        trials = []
        for _ in range(TRIALS_PER_PRIMITIVE):
            trial = probe_primitive(primitive, host, model)
            trials.append(trial)
            time.sleep(0.3)

        scores = score_activation_strength(trials)
        verdict, activation_type, stability_class = classify_primitive(
            layer,
            scores["activation_score"],
            scores["stability"],
            scores["avg_confidence"],
        )

        pr = PrimitiveResult17b(
            primitive_id=prim_id,
            layer=layer,
            primitive=primitive,
            hypothesis=hypothesis,
            activation_score=scores["activation_score"],
            phenom_hits=scores["phenom_hits"],
            abstract_hits=scores["abstract_hits"],
            activation_type=activation_type,
            stability=scores["stability"],
            stability_class=stability_class,
            modal_dominant=scores["modal_dominant"],
            modal_count=scores["modal_count"],
            avg_confidence=scores["avg_confidence"],
            linguistic_drift=scores["linguistic_drift"],
            parse_fail_rate=scores["parse_fail_rate"],
            stable_expansions=", ".join(scores["stable_expansions"]),
            all_dominants=", ".join(scores["all_dominants"]),
            verdict=verdict,
            sample_responses=" | ".join(scores["sample_responses"][:2]),
        )
        results.append(pr)

        drift_flag = " [DRIFT]" if pr.linguistic_drift else ""
        print(f"act:{pr.activation_score:.2f} [{activation_type:<17}] "
              f"stab:{pr.stability:.2f}  [{verdict}]{drift_flag}")

    report(results)
    save_results(results)
    return results

# ── reporting ─────────────────────────────────────────────────────────────────

def report(results: list[PrimitiveResult17b]):
    layer_0a = [r for r in results if r.layer == "0A"]
    layer_0b = [r for r in results if r.layer == "0B"]

    print(f"\n── LAYER 0A: SCAFFOLDING PRIMITIVES ────────────────────────────")
    print(f"  Expected: LOW activation score (abstract responses)")
    print(f"  {'Primitive':<15} {'Act':>5}  {'Type':<20} {'Stab':>5}  Verdict")
    print(f"  {'─────────':<15} {'───':>5}  {'────':<20} {'────':>5}  ───────")
    for r in layer_0a:
        drift = " D" if r.linguistic_drift else "  "
        print(f"  {r.primitive:<15} {r.activation_score:>5.2f}  "
              f"{r.activation_type:<20} {r.stability:>5.2f}  "
              f"{r.verdict}{drift}")

    avg_0a_activation = sum(r.activation_score for r in layer_0a) / len(layer_0a)
    confirmed_0a = sum(1 for r in layer_0a if r.verdict == "CONFIRMED_0A")
    print(f"\n  Avg activation: {avg_0a_activation:.3f}  "
          f"Confirmed: {confirmed_0a}/{len(layer_0a)}")

    print(f"\n── LAYER 0B: CONTENT PRIMITIVES ────────────────────────────────")
    print(f"  Expected: HIGH activation score (phenomenological responses)")
    print(f"  {'Primitive':<15} {'Act':>5}  {'Type':<20} {'Stab':>5}  Verdict")
    print(f"  {'─────────':<15} {'───':>5}  {'────':<20} {'────':>5}  ───────")
    for r in layer_0b:
        drift = " D" if r.linguistic_drift else "  "
        print(f"  {r.primitive:<15} {r.activation_score:>5.2f}  "
              f"{r.activation_type:<20} {r.stability:>5.2f}  "
              f"{r.verdict}{drift}")

    avg_0b_activation = sum(r.activation_score for r in layer_0b) / len(layer_0b)
    confirmed_0b = sum(1 for r in layer_0b if r.verdict == "CONFIRMED_0B")
    print(f"\n  Avg activation: {avg_0b_activation:.3f}  "
          f"Confirmed: {confirmed_0b}/{len(layer_0b)}")

    print(f"\n── THE KEY COMPARISON ───────────────────────────────────────────")
    print(f"  Layer 0a avg activation: {avg_0a_activation:.3f}  "
          f"(expected: LOW)")
    print(f"  Layer 0b avg activation: {avg_0b_activation:.3f}  "
          f"(expected: HIGH)")
    gap = avg_0b_activation - avg_0a_activation
    print(f"  Gap (0b - 0a):           {gap:+.3f}")

    print(f"\n── LINGUISTIC DRIFT REPORT ──────────────────────────────────────")
    drifters = [r for r in results if r.linguistic_drift]
    if drifters:
        for r in drifters:
            print(f"  {r.primitive:<15} [{r.layer}] — drifted to non-English")
            print(f"    dominants: {r.all_dominants[:60]}")
    else:
        print(f"  No linguistic drift observed")

    print(f"\n── UNEXPECTED RESULTS ───────────────────────────────────────────")
    unexpected = [r for r in results if "UNEXPECTED" in r.verdict]
    if unexpected:
        for r in unexpected:
            print(f"  {r.primitive:<15} [{r.layer}] — {r.verdict}")
            print(f"    act:{r.activation_score:.2f}  "
                  f"type:{r.activation_type}")
            print(f"    sample: {r.sample_responses[:100]}")
    else:
        print(f"  All primitives behaved as expected for their layer")

    print(f"\n── EXPANSION PATTERNS ───────────────────────────────────────────")
    print(f"  What do primitives naturally expand into?")
    for r in results:
        if r.stable_expansions:
            print(f"  {r.primitive:<15} → {r.stable_expansions}")

    print(f"\n── HYPOTHESIS VERDICT ───────────────────────────────────────────")
    if gap >= 0.15:
        print(f"  HYPOTHESIS SUPPORTED — clear activation gap between layers")
        print(f"  Layer 0a (scaffolding): abstract, relational responses")
        print(f"  Layer 0b (content): phenomenological, textured responses")
        print(f"  The two-sublayer model is real and measurable.")
        print(f"\n  IMPLICATION FOR SRM:")
        print(f"  Layer 0a primitives → combinatorial operators (pure code)")
        print(f"  Layer 0b primitives → content seeds (membrane-activatable)")
        print(f"  Experiment 18: combine 0a operators with 0b seeds")
        print(f"  and measure whether this produces Layer 1 concepts.")
    elif gap >= 0.05:
        print(f"  HYPOTHESIS WEAKLY SUPPORTED — small but present gap")
        print(f"  Distinction exists but needs larger model or more trials.")
    else:
        print(f"  HYPOTHESIS NOT SUPPORTED — no clear activation gap")
        print(f"  Rethink the 0a/0b taxonomy or scoring method.")

    print(f"\n── WHAT THIS MEANS FOR THE PAPER ────────────────────────────────")
    print(f"  Confirmed 0a primitives: {confirmed_0a}/8 behave as scaffolding")
    print(f"  Confirmed 0b primitives: {confirmed_0b}/8 behave as content")
    total_confirmed = confirmed_0a + confirmed_0b
    total = len(results)
    print(f"  Overall confirmation rate: {total_confirmed}/{total} "
          f"({total_confirmed/total:.0%})")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(results: list[PrimitiveResult17b],
                 base="semantic_primitives/results_exp_17b"):
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

    # update the living summary document
    summary_path = Path("semantic_primitives/primitive_summary.json")
    existing = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)

    layer_0a_confirmed = [r.primitive for r in results
                          if r.verdict == "CONFIRMED_0A"]
    layer_0b_confirmed = [r.primitive for r in results
                          if r.verdict == "CONFIRMED_0B"]
    unexpected = [r.primitive for r in results
                  if "UNEXPECTED" in r.verdict]

    existing["exp_17b"] = {
        "timestamp":          datetime.now().isoformat(),
        "layer_0a_confirmed": layer_0a_confirmed,
        "layer_0b_confirmed": layer_0b_confirmed,
        "unexpected":         unexpected,
        "avg_0a_activation":  round(
            sum(r.activation_score for r in results if r.layer == "0A") / 8, 3),
        "avg_0b_activation":  round(
            sum(r.activation_score for r in results if r.layer == "0B") / 8, 3),
        "linguistic_drifters": [r.primitive for r in results
                                if r.linguistic_drift],
    }

    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n── SAVED ────────────────────────────────────────────────────────")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  semantic_primitives/primitive_summary.json  (updated)")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    host  = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:0.5b"
    run_experiment(host, model)