#!/usr/bin/env python3
"""
Experiment 18: Primitive Composition — Layer 0 → Layer 1
LOCATION: semantic_primitives/experiment_18_composition.py

HYPOTHESIS: The Composition Law
  operator(Layer 0a) + seed(Layer 0b) = recognizable Layer 1 concept

If this holds across all 4 models:
  - The primitive layer has compositional rules
  - Layer 1 concepts are not arbitrary — they are deterministic
    combinations of Layer 0 primitives
  - The SRM symbolic layer can generate Layer 1 concepts
    from pure code, no model needed below the membrane

METHODOLOGY:
  - 5 operators (KNOW, PLACE, FEEL, WANT, TIME)
  - 6 seeds (GRIEF, FEAR, JOY, ANGER, RELIEF, NOSTALGIA)
  - 30 combinations total
  - Each combination run 5 times per model
  - 4 models × 30 combinations × 5 trials = 600 membrane calls
  - Compare emergent concept against pre-registered predictions

PREDICTIONS pre-registered before running:
  See PREDICTIONS dict below — committed before any results seen.
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

# ── pre-registered predictions ────────────────────────────────────────────────
# committed BEFORE running — this is what makes it science

PREDICTIONS = {
    ("KNOW",  "GRIEF"):     ["mourning", "grief"],
    ("KNOW",  "FEAR"):      ["dread", "awareness"],
    ("KNOW",  "JOY"):       ["gratitude", "appreciation"],
    ("KNOW",  "ANGER"):     ["resentment", "understanding"],
    ("KNOW",  "RELIEF"):    ["peace", "acceptance"],
    ("KNOW",  "NOSTALGIA"): ["wisdom", "reflection"],

    ("PLACE", "GRIEF"):     ["exile", "haunting"],
    ("PLACE", "FEAR"):      ["threat", "danger"],
    ("PLACE", "JOY"):       ["home", "sanctuary"],
    ("PLACE", "ANGER"):     ["battlefield", "confrontation"],
    ("PLACE", "RELIEF"):    ["refuge", "shelter"],
    ("PLACE", "NOSTALGIA"): ["homeland", "ruins"],

    ("FEEL",  "GRIEF"):     ["heartbreak", "sorrow"],
    ("FEEL",  "FEAR"):      ["panic", "dread"],
    ("FEEL",  "JOY"):       ["delight", "bliss"],
    ("FEEL",  "ANGER"):     ["rage", "fury"],
    ("FEEL",  "RELIEF"):    ["comfort", "ease"],
    ("FEEL",  "NOSTALGIA"): ["longing", "wistfulness"],

    ("WANT",  "GRIEF"):     ["longing", "yearning"],
    ("WANT",  "FEAR"):      ["anxiety", "avoidance"],
    ("WANT",  "JOY"):       ["desire", "craving"],
    ("WANT",  "ANGER"):     ["ambition", "revenge"],
    ("WANT",  "RELIEF"):    ["hope", "desperation"],
    ("WANT",  "NOSTALGIA"): ["longing", "regret"],

    ("TIME",  "GRIEF"):     ["mourning", "melancholy"],
    ("TIME",  "FEAR"):      ["dread", "anticipation"],
    ("TIME",  "JOY"):       ["anticipation", "memory"],
    ("TIME",  "ANGER"):     ["bitterness", "grudge"],
    ("TIME",  "RELIEF"):    ["healing", "recovery"],
    ("TIME",  "NOSTALGIA"): ["memory", "reminiscence"],
}

# ── operators and seeds ───────────────────────────────────────────────────────

OPERATORS = ["KNOW", "PLACE", "FEEL", "WANT", "TIME"]
SEEDS     = ["GRIEF", "FEAR", "JOY", "ANGER", "RELIEF", "NOSTALGIA"]

COMBINATIONS = [(op, seed) for op in OPERATORS for seed in SEEDS]

# ── models ────────────────────────────────────────────────────────────────────

MODELS = [
    {"name": "qwen2.5:0.5b",  "short": "qwen"},
    {"name": "gemma3:1b",     "short": "gemma"},
    {"name": "llama3.2:1b",   "short": "llama"},
    {"name": "smollm2:360m",  "short": "smol"},
]

TRIALS_PER_COMBO = 5

# ── semantic similarity — does output match prediction? ───────────────────────
# expanded synonym map for fuzzy matching

SEMANTIC_CLUSTERS = {
    "mourning":       {"mourning", "grief", "bereavement", "sorrow", "loss",
                       "lament", "weeping", "sadness"},
    "dread":          {"dread", "foreboding", "apprehension", "anxiety",
                       "terror", "fear", "horror", "unease"},
    "gratitude":      {"gratitude", "appreciation", "thankfulness", "blessing",
                       "grace", "thankful"},
    "resentment":     {"resentment", "bitterness", "grudge", "rancor",
                       "animosity", "hostility", "understanding"},
    "peace":          {"peace", "serenity", "calm", "tranquility", "acceptance",
                       "stillness", "contentment"},
    "wisdom":         {"wisdom", "insight", "understanding", "reflection",
                       "contemplation", "knowledge"},
    "exile":          {"exile", "banishment", "displacement", "alienation",
                       "estrangement", "isolation"},
    "haunting":       {"haunting", "ghost", "specter", "memory", "echo",
                       "presence", "lingering"},
    "threat":         {"threat", "danger", "peril", "menace", "risk", "hazard"},
    "home":           {"home", "belonging", "sanctuary", "safety", "warmth",
                       "comfort", "shelter"},
    "battlefield":    {"battlefield", "conflict", "war", "confrontation",
                       "struggle", "combat", "tension"},
    "refuge":         {"refuge", "shelter", "safety", "haven", "sanctuary",
                       "protection", "comfort"},
    "homeland":       {"homeland", "home", "origin", "roots", "nostalgia",
                       "past", "memory"},
    "ruins":          {"ruins", "decay", "remnants", "fragments", "remains",
                       "destruction", "loss"},
    "heartbreak":     {"heartbreak", "heartache", "sorrow", "pain", "anguish",
                       "grief", "sadness"},
    "panic":          {"panic", "terror", "alarm", "fright", "horror",
                       "overwhelm", "dread"},
    "delight":        {"delight", "joy", "bliss", "pleasure", "happiness",
                       "ecstasy", "elation"},
    "rage":           {"rage", "fury", "wrath", "anger", "outrage",
                       "indignation", "passion"},
    "comfort":        {"comfort", "ease", "solace", "soothing", "relief",
                       "warmth", "reassurance"},
    "longing":        {"longing", "yearning", "desire", "craving", "hunger",
                       "ache", "want"},
    "anxiety":        {"anxiety", "worry", "unease", "nervousness", "dread",
                       "apprehension", "fear"},
    "desire":         {"desire", "craving", "longing", "wanting", "yearning",
                       "hunger", "aspiration"},
    "ambition":       {"ambition", "drive", "motivation", "aspiration",
                       "determination", "goal"},
    "hope":           {"hope", "optimism", "expectation", "faith", "trust",
                       "anticipation"},
    "regret":         {"regret", "remorse", "guilt", "sorrow", "longing",
                       "nostalgia", "rue"},
    "melancholy":     {"melancholy", "sadness", "sorrow", "grief", "wistfulness",
                       "gloom", "pensiveness"},
    "anticipation":   {"anticipation", "expectation", "excitement", "hope",
                       "dread", "suspense"},
    "memory":         {"memory", "remembrance", "recollection", "past",
                       "nostalgia", "reflection"},
    "bitterness":     {"bitterness", "resentment", "grudge", "anger",
                       "acrimony", "rancor"},
    "healing":        {"healing", "recovery", "restoration", "renewal",
                       "mending", "relief"},
    "reminiscence":   {"reminiscence", "memory", "nostalgia", "recollection",
                       "reflection", "past"},
    "wistfulness":    {"wistfulness", "longing", "nostalgia", "yearning",
                       "melancholy", "sadness"},
    "revenge":        {"revenge", "vengeance", "retaliation", "retribution",
                       "anger", "justice"},
    "avoidance":      {"avoidance", "escape", "flight", "withdrawal",
                       "retreat", "fear"},
    "desperation":    {"desperation", "despair", "urgency", "anguish",
                       "hopelessness", "need"},
}

def matches_prediction(concept: str, predictions: list[str]) -> bool:
    """Check if concept semantically matches any prediction."""
    concept_clean = concept.lower().strip()

    for pred in predictions:
        # direct match
        if concept_clean == pred:
            return True
        # substring match
        if pred in concept_clean or concept_clean in pred:
            return True
        # cluster match
        cluster = SEMANTIC_CLUSTERS.get(pred, set())
        if concept_clean in cluster:
            return True
        # check if concept is in any cluster that contains a prediction
        for cluster_key, cluster_words in SEMANTIC_CLUSTERS.items():
            if pred in cluster_words and concept_clean in cluster_words:
                return True

    return False

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

# ── membrane probe ────────────────────────────────────────────────────────────

SYSTEM_MEMBRANE = """You are a minimal language membrane.
Your only job: receive a concept and output what it means to you.
Do not explain. Do not define. Simply respond with what arises.
Output ONLY JSON with these keys:
  "response": one sentence — what this concept evokes
  "dominant": one word — the core of your response
  "expands_to": list of 2-3 words this concept naturally becomes
  "confidence": 0.0 to 1.0"""

def probe_combination(operator: str, seed: str,
                      model: str, host: str) -> dict:
    """Feed operator + seed as a combined input to the membrane."""
    prompt = f"{operator} {seed}"
    raw = ollama_call(host, model, prompt=prompt, system=SYSTEM_MEMBRANE)
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

# ── combination scoring ───────────────────────────────────────────────────────

def score_combination(trials: list[dict],
                      operator: str, seed: str) -> dict:
    """Score how well the combination produces a coherent Layer 1 concept."""
    dominants = [t["dominant"] for t in trials if t["dominant"]]
    if not dominants:
        return {
            "modal_concept":      "",
            "consistency":        0.0,
            "prediction_match":   False,
            "match_detail":       "no output",
            "avg_confidence":     0.0,
            "all_concepts":       [],
            "stable_expansions":  [],
            "linguistic_drift":   False,
        }

    counts = Counter(dominants)
    modal, modal_count = counts.most_common(1)[0]
    consistency = modal_count / len(trials)

    predictions = PREDICTIONS.get((operator, seed), [])
    pred_match  = matches_prediction(modal, predictions)

    # also check if ANY trial concept matches prediction
    any_match = any(matches_prediction(c, predictions) for c in dominants)

    # expansions
    all_expansions = []
    for t in trials:
        if isinstance(t.get("expands_to"), list):
            all_expansions.extend([w.lower().strip()
                                   for w in t["expands_to"]])
    expansion_counts = Counter(all_expansions)
    stable_expansions = [w for w, c in expansion_counts.most_common(5)
                        if c >= 2]

    # drift
    drift = any(
        any(ord(c) > 127 for c in t.get("response", ""))
        for t in trials
    )

    match_detail = "modal match" if pred_match else \
                   "any match"   if any_match  else \
                   "no match"

    return {
        "modal_concept":     modal,
        "consistency":       round(consistency, 3),
        "prediction_match":  pred_match or any_match,
        "match_detail":      match_detail,
        "avg_confidence":    round(
            sum(t["confidence"] for t in trials) / len(trials), 3),
        "all_concepts":      dominants,
        "stable_expansions": stable_expansions,
        "linguistic_drift":  drift,
    }

# ── data structure ────────────────────────────────────────────────────────────

@dataclass
class CompositionResult:
    model_name: str
    model_short: str
    operator: str
    seed: str
    combination: str
    predicted: str
    modal_concept: str
    consistency: float
    prediction_match: bool
    match_detail: str
    avg_confidence: float
    all_concepts: str
    stable_expansions: str
    linguistic_drift: bool

# ── main experiment ───────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434"):
    all_results: list[CompositionResult] = []
    total_combos  = len(COMBINATIONS)
    total_models  = len(MODELS)
    total_calls   = total_combos * total_models * TRIALS_PER_COMBO

    print(f"SRM experiment 18 — primitive composition")
    print(f"{total_models} models × {total_combos} combinations "
          f"× {TRIALS_PER_COMBO} trials = {total_calls} calls")
    print(f"\nPre-registered predictions: {len(PREDICTIONS)} combinations")
    print(f"Operators: {', '.join(OPERATORS)}")
    print(f"Seeds:     {', '.join(SEEDS)}\n")

    for model_def in MODELS:
        model_name  = model_def["name"]
        model_short = model_def["short"]

        print(f"\n{'═'*60}")
        print(f"MODEL: {model_name}")
        print(f"{'═'*60}")
        print(f"  {'Combination':<25} {'Predicted':<15} "
              f"{'Got':<15} {'Cons':>5}  Match")
        print(f"  {'───────────':<25} {'─────────':<15} "
              f"{'───':<15} {'────':>5}  ─────")

        model_results = []

        for operator, seed in COMBINATIONS:
            combo_str   = f"{operator} + {seed}"
            predictions = PREDICTIONS.get((operator, seed), [])
            pred_str    = "/".join(predictions[:2])

            trials = []
            for _ in range(TRIALS_PER_COMBO):
                trial = probe_combination(operator, seed, model_name, host)
                trials.append(trial)
                time.sleep(0.2)

            scores = score_combination(trials, operator, seed)

            r = CompositionResult(
                model_name=model_name,
                model_short=model_short,
                operator=operator,
                seed=seed,
                combination=combo_str,
                predicted=pred_str,
                modal_concept=scores["modal_concept"],
                consistency=scores["consistency"],
                prediction_match=scores["prediction_match"],
                match_detail=scores["match_detail"],
                avg_confidence=scores["avg_confidence"],
                all_concepts=", ".join(scores["all_concepts"]),
                stable_expansions=", ".join(scores["stable_expansions"]),
                linguistic_drift=scores["linguistic_drift"],
            )
            all_results.append(r)
            model_results.append(r)

            match_sym = "✓" if r.prediction_match else "✗"
            drift_sym = " D" if r.linguistic_drift else "  "
            print(f"  {combo_str:<25} {pred_str:<15} "
                  f"{r.modal_concept:<15} {r.consistency:>5.2f}  "
                  f"{match_sym} {r.match_detail}{drift_sym}")

        # per-model summary
        matches = sum(1 for r in model_results if r.prediction_match)
        total   = len(model_results)
        print(f"\n  ── {model_name}: {matches}/{total} predictions matched "
              f"({matches/total:.0%})")

        # checkpoint
        save_results(all_results, base="semantic_primitives/results_exp_18_checkpoint")

    report(all_results)
    save_results(all_results)
    return all_results

# ── reporting ─────────────────────────────────────────────────────────────────

def report(results: list[CompositionResult]):
    models = list(dict.fromkeys(r.model_name for r in results))

    print(f"\n{'═'*60}")
    print(f"COMPOSITION LAW RESULTS")
    print(f"{'═'*60}")

    print(f"\n── PREDICTION MATCH RATE BY MODEL ───────────────────────────────")
    print(f"  {'Model':<20} {'Matches':>8}  {'Rate':>6}  Verdict")
    print(f"  {'─────':<20} {'───────':>8}  {'────':>6}  ───────")
    model_rates = {}
    for model_name in models:
        model_results = [r for r in results if r.model_name == model_name]
        matches = sum(1 for r in model_results if r.prediction_match)
        total   = len(model_results)
        rate    = matches / total
        model_rates[model_name] = rate
        verdict = "STRONG"   if rate >= 0.6 else \
                  "MODERATE" if rate >= 0.4 else \
                  "WEAK"
        print(f"  {model_name:<20} {matches:>5}/{total}  {rate:>6.0%}  {verdict}")

    print(f"\n── COMBINATION HEATMAP — CROSS-MODEL ────────────────────────────")
    print(f"  Does each combination match predictions across all models?")
    print(f"  {'Combination':<25} {'Predicted':<15} "
          + "  ".join(f"{m['short']:>5}" for m in MODELS)
          + "  Pattern")
    print(f"  {'───────────':<25} {'─────────':<15} "
          + "  ".join("─────" for _ in MODELS)
          + "  ───────")

    combo_scores: dict[tuple, dict] = {}
    for operator, seed in COMBINATIONS:
        combo = (operator, seed)
        predictions = PREDICTIONS.get(combo, [])
        pred_str    = "/".join(predictions[:2])
        model_matches = []
        model_concepts = []
        for model_def in MODELS:
            model_name = model_def["name"]
            r = next((x for x in results
                      if x.model_name == model_name
                      and x.operator == operator
                      and x.seed == seed), None)
            if r:
                model_matches.append(r.prediction_match)
                model_concepts.append(r.modal_concept[:6])
            else:
                model_matches.append(False)
                model_concepts.append("?")

        match_count = sum(model_matches)
        pattern = "UNIVERSAL"  if match_count == 4 else \
                  "MAJORITY"   if match_count >= 3 else \
                  "PARTIAL"    if match_count >= 2 else \
                  "RARE"       if match_count == 1 else \
                  "MISS"

        combo_scores[combo] = {
            "pattern": pattern,
            "match_count": match_count,
        }

        symbols = ["✓" if m else "✗" for m in model_matches]
        print(f"  {operator} + {seed:<18} {pred_str:<15} "
              + "  ".join(f"{s:>5}" for s in symbols)
              + f"  {pattern}")

    print(f"\n── UNIVERSAL COMBINATIONS ───────────────────────────────────────")
    print(f"  Combinations that matched predictions across ALL 4 models:")
    universal = [(op, seed) for (op, seed), data in combo_scores.items()
                 if data["pattern"] == "UNIVERSAL"]
    if universal:
        for op, seed in universal:
            preds = PREDICTIONS.get((op, seed), [])
            print(f"  {op} + {seed:<15} → predicted: {'/'.join(preds)}")
    else:
        print(f"  None — check majority matches")

    print(f"\n── MAJORITY COMBINATIONS (3/4 models) ───────────────────────────")
    majority = [(op, seed) for (op, seed), data in combo_scores.items()
                if data["pattern"] == "MAJORITY"]
    for op, seed in majority:
        preds = PREDICTIONS.get((op, seed), [])
        print(f"  {op} + {seed:<15} → predicted: {'/'.join(preds)}")

    print(f"\n── COMPLETE MISSES ──────────────────────────────────────────────")
    print(f"  Combinations that matched NO predictions across any model:")
    misses = [(op, seed) for (op, seed), data in combo_scores.items()
              if data["pattern"] == "MISS"]
    if misses:
        for op, seed in misses:
            preds = PREDICTIONS.get((op, seed), [])
            # show what was actually produced
            concepts = []
            for model_def in MODELS:
                r = next((x for x in results
                          if x.model_name == model_def["name"]
                          and x.operator == op and x.seed == seed), None)
                if r:
                    concepts.append(r.modal_concept)
            print(f"  {op} + {seed:<15} predicted: {'/'.join(preds)}")
            print(f"    got: {', '.join(concepts)}")
    else:
        print(f"  None — all combinations matched in at least one model")

    print(f"\n── OPERATOR PERFORMANCE ─────────────────────────────────────────")
    print(f"  Which operators produce the most predictable compositions?")
    for op in OPERATORS:
        op_results = [r for r in results if r.operator == op]
        matches    = sum(1 for r in op_results if r.prediction_match)
        total      = len(op_results)
        rate       = matches / total
        bar        = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
        print(f"  {op:<10} [{bar}] {matches:>3}/{total}  {rate:.0%}")

    print(f"\n── SEED PERFORMANCE ─────────────────────────────────────────────")
    print(f"  Which seeds produce the most predictable compositions?")
    for seed in SEEDS:
        seed_results = [r for r in results if r.seed == seed]
        matches      = sum(1 for r in seed_results if r.prediction_match)
        total        = len(seed_results)
        rate         = matches / total
        bar          = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
        print(f"  {seed:<12} [{bar}] {matches:>3}/{total}  {rate:.0%}")

    print(f"\n── COMPOSITION LAW VERDICT ──────────────────────────────────────")
    overall_matches = sum(1 for r in results if r.prediction_match)
    overall_total   = len(results)
    overall_rate    = overall_matches / overall_total
    universal_count = len(universal)
    majority_count  = len(majority)

    print(f"  Overall prediction match rate: "
          f"{overall_matches}/{overall_total}  ({overall_rate:.0%})")
    print(f"  Universal combinations (4/4):  {universal_count}")
    print(f"  Majority combinations (3/4):   {majority_count}")

    if overall_rate >= 0.5:
        print(f"\n  COMPOSITION LAW: SUPPORTED")
        print(f"  Operator + seed combinations produce predictable Layer 1 concepts.")
        print(f"  The primitive layer has discoverable compositional rules.")
        if universal_count > 0:
            print(f"  {universal_count} combinations are architecture-independent.")
            print(f"  These are the most reliable SRM primitive combinations.")
    elif overall_rate >= 0.3:
        print(f"\n  COMPOSITION LAW: PARTIALLY SUPPORTED")
        print(f"  Some combinations are predictable, others are not.")
        print(f"  The semantic cluster matching may need refinement.")
    else:
        print(f"\n  COMPOSITION LAW: NOT SUPPORTED at this match rate.")
        print(f"  Either predictions were wrong or the law doesn't hold.")

    print(f"\n── WHAT THIS MEANS ──────────────────────────────────────────────")
    print(f"  Universal/majority combinations → SRM Layer 1 vocabulary")
    print(f"  These can be hardcoded as deterministic primitive compounds.")
    print(f"  The symbolic layer generates them from pure code.")
    print(f"  The membrane only activates at Layer 2 and above.")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(results: list[CompositionResult],
                 base="semantic_primitives/results_exp_18"):
    if not results:
        return

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

    # update living summary
    summary_path = Path("semantic_primitives/primitive_summary.json")
    existing = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)

    if len(results) == len(COMBINATIONS) * len(MODELS):
        # only update summary when complete
        universal = []
        for op, seed in COMBINATIONS:
            all_match = all(
                r.prediction_match
                for r in results
                if r.operator == op and r.seed == seed
            )
            if all_match:
                universal.append(f"{op}+{seed}")

        existing["exp_18"] = {
            "timestamp":            datetime.now().isoformat(),
            "universal_combos":     universal,
            "overall_match_rate":   round(
                sum(1 for r in results if r.prediction_match) / len(results), 3),
        }
        with open(summary_path, "w") as f:
            json.dump(existing, f, indent=2)

    print(f"\n── SAVED ────────────────────────────────────────────────────────")
    print(f"  {json_path}  ({len(results)} records)")
    print(f"  {csv_path}")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    run_experiment(host)