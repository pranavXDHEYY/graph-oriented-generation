#!/usr/bin/env python3
"""
Experiment 16: Semantic Density Mapping — Two-Word Combinations
PATH C: Discover the geometry of meaningful combinations.

HYPOTHESIS:
Some two-word combinations are semantically DENSE —
the membrane responds to them with high consistency across runs.
Others are SPARSE — the membrane produces different concepts each time.

Dense pairs = the atomic vocabulary of SRM compression.
Sparse pairs = noise, not useful for symbolic communication.

METHOD:
- Generate two-word combinations from curated seed pool
- Run each combination 3 times independently
- Measure consistency: do all 3 runs produce the same dominant concept?
- Score: 1.0 = identical concepts, 0.0 = all different
- Build a ranked map of dense vs sparse pairs

SEED WORDS:
Curated from high-scoring outputs across experiments 1-15.
Every word here has produced at least one strong membrane response.
"""
import random
import requests
import json
import csv
import time
import re
import itertools
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── seed pool — curated from high-scoring experiment outputs ──────────────────
# every word here produced a strong membrane response in experiments 1-15
# sourced from: SYNTHESIS hits, high-confidence runs, notable concepts

SEED_WORDS = [
    # from exp 1-3 phoneme runs that produced strong outputs
    "morning", "fishing", "feminine", "magnetic", "solstice",

    # from exp 5 HIGH_NEGATIVE confirmed synthesis triggers
    "grief", "dread", "despair", "doom", "loss",
    "hollow", "bleak", "forsaken", "wretched", "torment",

    # from exp 6-7 verb runs that broke synthesis cleanly
    "devour", "shatter", "forsake", "dissolve", "haunt",
    "wither", "cleave", "sunder", "smolder", "languish",

    # from exp 11 VOICE_PROJECTION runs — strongest membrane activation
    "melancholy", "longing", "tension", "anticipation", "sorrow",
    "yearning", "anguish", "warmth", "radiance", "stillness",

    # from exp 14-15 high-coherence concepts
    "conflict", "decay", "emergence", "fracture", "threshold",
    "silence", "descent", "void", "recursion", "convergence",

    # structural anchors from exp 9 PURE_STRUCTURE runs
    "pattern", "rhythm", "structure", "boundary", "flow",

    # wild cards — single-word outputs that appeared repeatedly
    "shadow", "flame", "echo", "storm", "bloom",
    "blood", "breath", "stone", "tide", "ash",
]

# ── English vocabulary for concept validation (from exp 14) ──────────────────
ENGLISH_VOCABULARY = {
    "abandon", "absence", "abyss", "aftermath", "agony", "anger", "anguish",
    "anticipation", "anxiety", "ash", "beauty", "betrayal", "bliss", "bloom",
    "blood", "boundary", "breath", "calm", "chaos", "clarity", "cleave",
    "cold", "collapse", "conflict", "connection", "contrast", "convergence",
    "corruption", "cycle", "dark", "dawn", "decay", "descent", "despair",
    "destruction", "devour", "dissolution", "doom", "dread", "drift", "dusk",
    "echo", "emergence", "emptiness", "entropy", "exile", "fall", "fear",
    "feminine", "fishing", "flame", "flow", "fog", "forsake", "forsaken",
    "fracture", "fragment", "grief", "growth", "harmony", "haunt", "hollow",
    "hope", "horror", "hunger", "isolation", "journey", "languish", "longing",
    "loss", "magnetic", "melancholy", "memory", "morning", "mourning",
    "mystery", "oblivion", "pattern", "peace", "radiance", "recursion",
    "repetition", "rhythm", "ruin", "rupture", "sadness", "separation",
    "shadow", "shatter", "silence", "smolder", "sorrow", "stillness", "stone",
    "storm", "structure", "sunder", "tension", "threshold", "tide", "torment",
    "transformation", "void", "warmth", "wither", "wonder", "wound", "yearning",
}

TRIALS_PER_PAIR = 3       # run each pair this many times
MAX_PAIRS = 150           # cap total pairs tested — manageable run time
DENSE_THRESHOLD = 0.7     # consistency score above this = dense pair

# ── HTTP ──────────────────────────────────────────────────────────────────────

def ollama_call(host, model, prompt, timeout=60, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout
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

def probe_pair(word_a: str, word_b: str, host: str, model: str) -> dict:
    """
    Run the pair through the membrane once.
    Returns dominant_concept, translation, confidence.
    """
    prompt = (
        "You are a minimal language membrane. "
        "Find whatever meaning emerges from these fragments. "
        "Output ONLY JSON with keys: "
        "\"dominant_concept\" (one English word — the core concept), "
        "\"translation\" (one sentence), "
        "\"confidence\" (0.0 to 1.0 as a number).\n\n"
        f"Fragments: {word_a} {word_b}"
    )
    raw = ollama_call(host, model, prompt)
    result = extract_json(raw)
    if not result:
        return {"dominant_concept": "", "translation": raw[:100], "confidence": 0.0}
    return {
        "dominant_concept": str(result.get("dominant_concept", "")).lower().strip(),
        "translation":      str(result.get("translation", "")),
        "confidence":       safe_float(result.get("confidence", 0.0)),
    }

# ── consistency scoring ───────────────────────────────────────────────────────

def compute_consistency(trials: list[dict]) -> dict:
    """
    Measure how consistently the membrane responds to a pair across trials.

    consistency_score: 1.0 = all trials produced same dominant concept
                       0.5 = two of three matched
                       0.0 = all different

    concept_real: fraction of trials where dominant_concept is a known word

    avg_confidence: mean confidence across trials
    """
    concepts = [t["dominant_concept"] for t in trials if t["dominant_concept"]]

    if not concepts:
        return {
            "consistency_score": 0.0,
            "dominant_concept":  "",
            "concept_real":      0.0,
            "avg_confidence":    0.0,
            "concept_variance":  [],
        }

    # count concept frequencies
    concept_counts = Counter(concepts)
    most_common, most_common_count = concept_counts.most_common(1)[0]

    # consistency = fraction of trials that agree on the most common concept
    consistency = most_common_count / len(trials)

    # concept_real = fraction of trials that produced a known English word
    real_count = sum(
        1 for c in concepts
        if re.sub(r'[^a-z]', '', c) in ENGLISH_VOCABULARY
    )
    concept_real = real_count / len(trials)

    avg_conf = sum(t["confidence"] for t in trials) / len(trials)

    return {
        "consistency_score": round(consistency, 3),
        "dominant_concept":  most_common,
        "concept_real":      round(concept_real, 3),
        "avg_confidence":    round(avg_conf, 3),
        "concept_variance":  concepts,  # all concepts produced
    }

# ── data structure ────────────────────────────────────────────────────────────

@dataclass
class PairResult:
    pair_id: int
    word_a: str
    word_b: str
    pair: str                    # "word_a + word_b"
    consistency_score: float     # KEY METRIC — how stable is the membrane response?
    dominant_concept: str        # most common concept across trials
    concept_real: float          # fraction of real English concept responses
    avg_confidence: float        # mean confidence across trials
    concept_variance: str        # all concepts produced (comma-separated)
    density_class: str           # DENSE / SPARSE / UNSTABLE
    trial_concepts: str          # raw concepts from each trial

# ── density classification ────────────────────────────────────────────────────

def classify_density(consistency: float, concept_real: float) -> str:
    if consistency >= DENSE_THRESHOLD and concept_real >= 0.67:
        return "DENSE"
    elif consistency >= DENSE_THRESHOLD and concept_real < 0.67:
        return "ANCHORED"  # consistent but not real English — echoes input
    elif consistency < 0.34:
        return "UNSTABLE"  # all different every time
    else:
        return "SPARSE"    # some consistency but not reliable

# ── pair generation ───────────────────────────────────────────────────────────

def generate_pairs(seed_words: list, max_pairs: int) -> list[tuple]:
    """
    Generate two-word combinations from seed pool.
    Uses all unique pairs up to max_pairs limit.
    Shuffled so we sample broadly rather than exhausting
    combinations from the first few words.
    """
    all_pairs = list(itertools.combinations(seed_words, 2))
    random.shuffle(all_pairs)
    return all_pairs[:max_pairs]

# ── main experiment ───────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434", model="qwen2.5:0.5b"):
    pairs = generate_pairs(SEED_WORDS, MAX_PAIRS)
    results: list[PairResult] = []
    total_calls = len(pairs) * TRIALS_PER_PAIR

    print(f"SRM experiment 16 — semantic density mapping")
    print(f"Two-word combinations from curated seed pool")
    print(f"{len(pairs)} pairs × {TRIALS_PER_PAIR} trials = {total_calls} membrane calls\n")
    print(f"Dense threshold: consistency >= {DENSE_THRESHOLD}")
    print(f"Seed pool: {len(SEED_WORDS)} words\n")

    for i, (word_a, word_b) in enumerate(pairs):
        pair_str = f"{word_a} + {word_b}"
        print(f"  [{i+1:03d}/{len(pairs)}] {pair_str:<35}", end=" ", flush=True)

        # run TRIALS_PER_PAIR independent membrane probes
        trials = []
        for trial_num in range(TRIALS_PER_PAIR):
            result = probe_pair(word_a, word_b, host, model)
            trials.append(result)
            time.sleep(0.2)  # gentle pacing

        # score consistency
        scores = compute_consistency(trials)
        density = classify_density(
            scores["consistency_score"],
            scores["concept_real"]
        )

        pr = PairResult(
            pair_id=i + 1,
            word_a=word_a,
            word_b=word_b,
            pair=pair_str,
            consistency_score=scores["consistency_score"],
            dominant_concept=scores["dominant_concept"],
            concept_real=scores["concept_real"],
            avg_confidence=scores["avg_confidence"],
            concept_variance=str(scores["concept_variance"]),
            density_class=density,
            trial_concepts=", ".join(
                t["dominant_concept"] for t in trials
            ),
        )
        results.append(pr)

        print(f"[{density:<8}] cons:{pr.consistency_score:.2f}  "
              f"conf:{pr.avg_confidence:.2f}  "
              f"→ {pr.dominant_concept[:20]}")

        # checkpoint every 25 pairs
        if (i + 1) % 25 == 0:
            save_results(results, base="results_exp_16_checkpoint")

    report(results)
    save_results(results)
    return results

# ── reporting ─────────────────────────────────────────────────────────────────

def report(results: list[PairResult]):
    total = len(results)
    dense    = [r for r in results if r.density_class == "DENSE"]
    anchored = [r for r in results if r.density_class == "ANCHORED"]
    sparse   = [r for r in results if r.density_class == "SPARSE"]
    unstable = [r for r in results if r.density_class == "UNSTABLE"]

    print(f"\n── DENSITY DISTRIBUTION ─────────────────────────────────────────")
    print(f"  {'Class':<12} {'Count':>6}  {'Rate':>6}  Description")
    print(f"  {'─────':<12} {'─────':>6}  {'────':>6}  ───────────")
    print(f"  {'DENSE':<12} {len(dense):>6}  "
          f"{len(dense)/total:>5.0%}  consistent + real English concept")
    print(f"  {'ANCHORED':<12} {len(anchored):>6}  "
          f"{len(anchored)/total:>5.0%}  consistent but echoes input word")
    print(f"  {'SPARSE':<12} {len(sparse):>6}  "
          f"{len(sparse)/total:>5.0%}  some consistency, not reliable")
    print(f"  {'UNSTABLE':<12} {len(unstable):>6}  "
          f"{len(unstable)/total:>5.0%}  different concept every run")

    print(f"\n── TOP 20 DENSE PAIRS ───────────────────────────────────────────")
    print(f"  These are your atomic SRM vocabulary candidates.")
    print(f"  {'Pair':<35} {'Concept':<20} {'Cons':>5}  {'Conf':>5}")
    print(f"  {'────':<35} {'───────':<20} {'────':>5}  {'────':>5}")
    top_dense = sorted(dense, key=lambda r: (
        r.consistency_score, r.avg_confidence
    ), reverse=True)[:20]
    for r in top_dense:
        print(f"  {r.pair:<35} {r.dominant_concept:<20} "
              f"{r.consistency_score:>5.2f}  {r.avg_confidence:>5.2f}")

    print(f"\n── TOP 10 UNSTABLE PAIRS ────────────────────────────────────────")
    print(f"  These pairs resist compression — too ambiguous for SRM use.")
    print(f"  {'Pair':<35} {'Concepts produced'}")
    print(f"  {'────':<35} {'─────────────────'}")
    top_unstable = sorted(unstable,
                          key=lambda r: r.consistency_score)[:10]
    for r in top_unstable:
        print(f"  {r.pair:<35} {r.trial_concepts}")

    print(f"\n── SEMANTIC GEOMETRY OBSERVATIONS ──────────────────────────────")

    # which seed words appear most in dense pairs?
    dense_words: list[str] = []
    for r in dense:
        dense_words.extend([r.word_a, r.word_b])
    word_density = Counter(dense_words).most_common(10)
    print(f"  Words that form the most dense pairs:")
    for word, count in word_density:
        print(f"    {word:<20} appears in {count} dense pairs")

    # what concepts do dense pairs converge on?
    dense_concepts = Counter(r.dominant_concept for r in dense).most_common(10)
    print(f"\n  Concepts dense pairs converge on:")
    for concept, count in dense_concepts:
        print(f"    {concept:<20} {count} pairs converge here")

    # avg consistency by word — which words anchor meaning?
    word_consistency: dict[str, list] = {}
    for r in results:
        word_consistency.setdefault(r.word_a, []).append(r.consistency_score)
        word_consistency.setdefault(r.word_b, []).append(r.consistency_score)
    word_avg = {
        w: sum(scores) / len(scores)
        for w, scores in word_consistency.items()
    }
    top_anchors = sorted(word_avg.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Words that anchor consistency across all pairs:")
    for word, avg in top_anchors:
        print(f"    {word:<20} avg consistency: {avg:.3f}")

    print(f"\n── HYPOTHESIS VERDICT ───────────────────────────────────────────")
    dense_rate = len(dense) / total
    if dense_rate >= 0.2:
        print(f"  SEMANTIC DENSITY EXISTS — {dense_rate:.0%} of pairs are dense")
        print(f"  The membrane has a stable vocabulary of atomic pairs.")
        print(f"  Path C validated: compression dictionary is buildable.")
    elif dense_rate >= 0.1:
        print(f"  SEMANTIC DENSITY PARTIAL — {dense_rate:.0%} of pairs are dense")
        print(f"  Some atomic pairs exist but the space is mostly sparse.")
    else:
        print(f"  SEMANTIC DENSITY WEAK — only {dense_rate:.0%} dense pairs found")
        print(f"  Two-word combinations may be too short for stable meaning.")
        print(f"  Recommendation: try three-word combinations next.")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(results: list[PairResult], base="results_exp_16"):
    if not results:
        return

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

    print(f"\n── SAVED ────────────────────────────────────────────────────────")
    print(f"  {json_path}  ({len(results)} pairs)")
    print(f"  {csv_path}")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    host  = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:0.5b"
    run_experiment(host, model)