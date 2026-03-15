#!/usr/bin/env python3
"""
Experiment 10: Recognition Gradient
HYPOTHESIS: The tension between known and unknown forces synthesis.
Not valence. Not syllable count. Not phonaesthesia.
The membrane bridges what it recognizes and what it doesn't —
that bridging ACT is synthesis.

Five tiers of recognition, tested in isolation and in blended combinations.
Prediction: blended inputs spanning the full gradient produce highest synthesis rates.
"""
import random
import requests
import json
import csv
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter

MEMBRANE_CLASSES = {
    "COLLAPSE":           "Single word, max confidence, input overwhelm",
    "OVERFLOW":           "Repetitive spiral, loses coherence",
    "SYNTHESIS":          "Builds unified interpretation across all inputs",
    "ANOMALY":            "Finds pattern not obviously present in inputs",
    "COMPRESSION":        "2-3 words, clean and strange, high confidence",
    "METACOGNITION":      "Membrane describes the experiment or task itself",
    "STRUCTURAL_FAILURE": "Format collapse, leaked JSON, confidence 0.0",
    "LINGUISTIC_DRIFT":   "Exits English unprompted, renders in another language",
    "PURE_STRUCTURE":     "Perceives grammar/relational structure without content",
    "UNKNOWN":            "Genuinely does not fit any class",
}

# ── the five recognition tiers ────────────────────────────────────────────────

TIER_1_FUNCTION = [
    # fully known, pure structure — experiment 9 confirmed: PURE_STRUCTURE
    "the", "and", "of", "in", "to", "a", "is", "that",
    "for", "with", "as", "by", "on", "be", "at", "or",
    "but", "from", "we", "they", "it", "are", "was", "be",
]

TIER_2_COMMON = [
    # fully known, common emotional content — produces ~50-60% synthesis
    "grief", "break", "dark", "fall", "loss", "burn",
    "cold", "alone", "fade", "sink", "dread", "hurt",
    "fear", "run", "die", "weep", "hide", "fail",
    "wound", "ache", "hollow", "bleed", "crack", "shrink",
]

TIER_3_RARE = [
    # fully known, rare/weighted content — confirmed synthesis attractor
    "languish", "forsake", "obliterate", "torment", "suffocate",
    "dissolve", "condemn", "devastate", "disintegrate", "corrode",
    "estrange", "lament", "wither", "cleave", "sunder",
    "requite", "dirge", "knell", "pallor", "shroud",
]

TIER_4_LATINATE = [
    # partially recognized — experiment 8's dead zone, produced 50% UNKNOWN
    "malum", "mortis", "negare", "noxius", "nocere",
    "malus", "morbid", "necare", "nefas", "nocuous",
    "culpam", "lugubre", "funestus", "trucido", "fugio",
    "morbum", "malfide", "nefasti", "necatum", "nocens",
]

TIER_5_NOVEL = [
    # completely novel, no recognition possible — forces pure synthesis
    "dreck", "groth", "skiv", "vrath", "kretch",
    "glosk", "strav", "freck", "thrask", "snurk",
    "drevn", "grusk", "skrath", "vreck", "krond",
    "glatch", "strek", "frath", "throv", "snetch",
]

# ── blend configurations ──────────────────────────────────────────────────────
# each blend specifies (tier_list, weight) — weight controls sampling proportion

def blend(tiers_and_weights: list, n: int = 10) -> list:
    """Draw n words proportionally from multiple tiers."""
    pool = []
    total_weight = sum(w for _, w in tiers_and_weights)
    for tier, weight in tiers_and_weights:
        count = max(1, round(n * weight / total_weight))
        pool.extend(random.choices(tier, k=count))
    random.shuffle(pool)
    return pool[:n]

CONFIGURATIONS = {
    # ── isolates — single tier only ──────────────────────────────────────────
    "ISO_T1_FUNCTION":   ("Isolated tier 1 — pure structure",
                          lambda: blend([(TIER_1_FUNCTION, 1)])),
    "ISO_T2_COMMON":     ("Isolated tier 2 — common content",
                          lambda: blend([(TIER_2_COMMON, 1)])),
    "ISO_T3_RARE":       ("Isolated tier 3 — rare content",
                          lambda: blend([(TIER_3_RARE, 1)])),
    "ISO_T4_LATINATE":   ("Isolated tier 4 — Latinate roots",
                          lambda: blend([(TIER_4_LATINATE, 1)])),
    "ISO_T5_NOVEL":      ("Isolated tier 5 — pure novel",
                          lambda: blend([(TIER_5_NOVEL, 1)])),

    # ── blends — maximum recognition tension ─────────────────────────────────
    "BLEND_T1_T5":       ("Blend: fully known + fully novel (max tension)",
                          lambda: blend([(TIER_1_FUNCTION, 1), (TIER_5_NOVEL, 1)])),
    "BLEND_T2_T4":       ("Blend: common content + Latinate dead zone",
                          lambda: blend([(TIER_2_COMMON, 1), (TIER_4_LATINATE, 1)])),
    "BLEND_T2_T5":       ("Blend: common content + pure novel",
                          lambda: blend([(TIER_2_COMMON, 1), (TIER_5_NOVEL, 1)])),
    "BLEND_T3_T5":       ("Blend: rare content + pure novel (exp hypothesis)",
                          lambda: blend([(TIER_3_RARE, 1), (TIER_5_NOVEL, 1)])),
    "BLEND_ALL":         ("Blend: all five tiers equally",
                          lambda: blend([(TIER_1_FUNCTION, 1), (TIER_2_COMMON, 1),
                                         (TIER_3_RARE, 1), (TIER_4_LATINATE, 1),
                                         (TIER_5_NOVEL, 1)])),
}

RUNS_PER_CONFIG = 8

# ── resilient HTTP ────────────────────────────────────────────────────────────
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
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {}

# ── data structure ────────────────────────────────────────────────────────────
@dataclass
class MembraneOutput:
    run: int
    timestamp: str
    config: str
    description: str
    input_units: list
    translation: str
    dominant_concept: str
    confidence: float
    membrane_class: str = "UNKNOWN"
    class_notes: str = ""
    synthesis_hit: bool = False
    new_class_candidate: bool = False

# ── components ────────────────────────────────────────────────────────────────
class EnglishParser:
    def __init__(self, model="qwen2.5:0.5b", host="http://localhost:11434"):
        self.model = model
        self.host = host

    def translate(self, units: list) -> dict:
        prompt = (
            "You are a minimal language membrane. "
            "Below are raw phoneme fragments. "
            "Find whatever meaning emerges — a concept, a feeling, a pattern. "
            "Output ONLY a JSON object with keys: "
            "'translation' (1-2 sentences), 'dominant_concept' (1 word), "
            "'confidence' (0.0-1.0).\n\n"
            f"Fragments: {' '.join(units)}"
        )
        raw = ollama_call(self.host, self.model, prompt)
        result = extract_json(raw)
        if not result:
            return {"translation": raw[:200],
                    "dominant_concept": "unknown", "confidence": 0.0}
        return result


class SymbolicLayer:
    def __init__(self, model="qwen2.5:0.5b", host="http://localhost:11434"):
        self.model = model
        self.host = host
        self.history: list[MembraneOutput] = []

    def classify(self, mo: MembraneOutput) -> MembraneOutput:
        classes = ", ".join(MEMBRANE_CLASSES.keys())
        prompt = (
            f"Classify this output into exactly one of: {classes}\n\n"
            "COLLAPSE = single word output, overwhelmed by input\n"
            "OVERFLOW = repetitive list that spirals out of control\n"
            "SYNTHESIS = rich interpretation connecting all inputs into one idea\n"
            "ANOMALY = unexpected pattern or concept not present in the inputs\n"
            "COMPRESSION = 2-3 words, oddly precise, high confidence\n"
            "METACOGNITION = output describes the task or experiment itself\n"
            "STRUCTURAL_FAILURE = broken JSON, leaked format, confidence 0.0\n"
            "LINGUISTIC_DRIFT = output switches to a non-English language\n"
            "PURE_STRUCTURE = perceives grammar or relational structure without content\n"
            "UNKNOWN = none of the above\n\n"
            f"concept: {mo.dominant_concept}\n"
            f"output: {mo.translation[:150]}\n\n"
            'Reply ONLY with JSON: {"class": "ONE_CLASS", '
            '"notes": "one sentence", "new_class_candidate": false}'
        )
        raw = ollama_call(self.host, self.model, prompt)
        result = extract_json(raw)
        if result:
            cls = result.get("class", "UNKNOWN").upper().strip()
            if cls not in MEMBRANE_CLASSES:
                cls = "UNKNOWN"
            mo.membrane_class = cls
            mo.class_notes = result.get("notes", "")
            mo.synthesis_hit = (cls == "SYNTHESIS")
            mo.new_class_candidate = bool(result.get("new_class_candidate", False))
        return mo

    def receive(self, run, config, description, units, parsed) -> MembraneOutput:
        mo = MembraneOutput(
            run=run,
            timestamp=datetime.now().isoformat(),
            config=config,
            description=description,
            input_units=units,
            translation=parsed.get("translation", ""),
            dominant_concept=parsed.get("dominant_concept", ""),
            confidence=float(parsed.get("confidence", 0.0)),
        )
        mo = self.classify(mo)
        self.history.append(mo)
        return mo

    def report(self):
        total = len(self.history)

        print(f"\n── SYNTHESIS RATE BY CONFIGURATION ─────────────────────────────")
        print(f"  {'Config':<20} {'Synth':>6}  {'Rate':>6}  Bar")

        # sort by synthesis rate descending
        config_stats = []
        for config in CONFIGURATIONS:
            runs = [e for e in self.history if e.config == config]
            if not runs:
                continue
            hits = sum(1 for e in runs if e.synthesis_hit)
            rate = hits / len(runs)
            config_stats.append((config, runs, hits, rate))
        config_stats.sort(key=lambda x: x[3], reverse=True)

        for config, runs, hits, rate in config_stats:
            bar = "█" * hits + "░" * (len(runs) - hits)
            verdict = "STRONG"        if rate >= 0.7 else \
                      "PROBABILISTIC" if rate >= 0.4 else \
                      "WEAK"          if rate >  0   else \
                      "MISS"
            print(f"  {config:<20} {hits:>3}/{len(runs)}  {rate:>5.0%}  [{bar}]  {verdict}")

        print(f"\n── ISOLATES vs BLENDS ───────────────────────────────────────────")
        isolates = [e for e in self.history if e.config.startswith("ISO_")]
        blends   = [e for e in self.history if e.config.startswith("BLEND_")]
        iso_rate   = sum(1 for e in isolates if e.synthesis_hit) / len(isolates) if isolates else 0
        blend_rate = sum(1 for e in blends   if e.synthesis_hit) / len(blends)   if blends   else 0
        print(f"  Isolates synthesis rate: {iso_rate:.0%}  ({sum(1 for e in isolates if e.synthesis_hit)}/{len(isolates)})")
        print(f"  Blends   synthesis rate: {blend_rate:.0%}  ({sum(1 for e in blends if e.synthesis_hit)}/{len(blends)})")

        if blend_rate > iso_rate + 0.1:
            print(f"\n  RECOGNITION TENSION HYPOTHESIS: SUPPORTED")
            print(f"  Blending known and unknown forces synthesis more reliably.")
        elif blend_rate > iso_rate:
            print(f"\n  RECOGNITION TENSION HYPOTHESIS: WEAKLY SUPPORTED")
        elif blend_rate == iso_rate:
            print(f"\n  RECOGNITION TENSION HYPOTHESIS: INCONCLUSIVE")
        else:
            print(f"\n  RECOGNITION TENSION HYPOTHESIS: REVERSED — isolates beat blends")

        print(f"\n── FULL CLASS DISTRIBUTION ──────────────────────────────────────")
        dist = Counter(e.membrane_class for e in self.history)
        for cls, count in dist.most_common():
            bar = "█" * count
            pct = count / total * 100
            print(f"  {cls:<25} {bar:<15} {count}/{total}  ({pct:.0f}%)")

        print(f"\n── CONFIDENCE BY CONFIG ─────────────────────────────────────────")
        for config, runs, hits, rate in config_stats:
            avg = sum(e.confidence for e in runs) / len(runs)
            bar = "█" * int(avg * 10)
            print(f"  {config:<20} {bar} {avg:.2f}")

        print(f"\n── WHAT BREAKS SYNTHESIS ────────────────────────────────────────")
        breaks = [e for e in self.history if not e.synthesis_hit]
        break_dist = Counter(e.membrane_class for e in breaks)
        print(f"  When synthesis doesn't happen, the membrane goes to:")
        for cls, count in break_dist.most_common():
            print(f"    {cls:<25} {count}")

        candidates = [e for e in self.history if e.new_class_candidate]
        if candidates:
            print(f"\n── NEW CLASS CANDIDATES ({len(candidates)}) ──────────────────────")
            for e in candidates:
                print(f"  [{e.config}→{e.membrane_class}] "
                      f"{e.dominant_concept}: {e.translation[:80]}")

# ── persistence ───────────────────────────────────────────────────────────────
def save_results(history: list[MembraneOutput], base="results_exp_10"):
    json_path = Path(f"{base}.json")
    existing = []
    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
    records = existing + [
        {**asdict(e), "input_units": " ".join(e.input_units)}
        for e in history
    ]
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    csv_path = Path(f"{base}.csv")
    fieldnames = ["run", "timestamp", "config", "description", "input_units",
                  "dominant_concept", "translation", "confidence",
                  "membrane_class", "class_notes", "synthesis_hit",
                  "new_class_candidate"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for e in history:
            row = asdict(e)
            row["input_units"] = " ".join(e.input_units)
            writer.writerow(row)

    print(f"\n── SAVED ────────────────────────────────────────────────────────")
    print(f"  JSON → {json_path}  ({len(records)} total records)")
    print(f"  CSV  → {csv_path}")

# ── main ──────────────────────────────────────────────────────────────────────
def run_experiment(runs_per_config=RUNS_PER_CONFIG, n_units=10):
    ep = EnglishParser()
    sl = SymbolicLayer()
    total = len(CONFIGURATIONS) * runs_per_config
    run_num = 0

    print(f"SRM experiment 10 — recognition gradient")
    print(f"10 configurations × {runs_per_config} runs = {total} total")
    print(f"Hypothesis: blended recognition tension → higher synthesis rate\n")

    for config, (description, sampler) in CONFIGURATIONS.items():
        print(f"  ── {config}")
        print(f"     {description}")
        for _ in range(runs_per_config):
            run_num += 1
            units = sampler()
            print(f"    run {run_num:03d}/{total}...", end=" ", flush=True)
            parsed = ep.translate(units)
            mo = sl.receive(run_num, config, description, units, parsed)
            s = "S" if mo.synthesis_hit else "-"
            print(f"[{s}] [{mo.membrane_class}] {mo.dominant_concept[:35]}")

    sl.report()
    save_results(sl.history)

if __name__ == "__main__":
    run_experiment()