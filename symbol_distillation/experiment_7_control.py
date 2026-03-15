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
    "UNKNOWN":            "Genuinely does not fit any class",
}

# ── pool A: monosyllabic, common, everyday verbs ──────────────────────────────
# hypothesis: triggers METACOGNITION — membrane recognizes them as language
MONO_COMMON = [
    "break", "flee", "fade", "rot", "drown",
    "burn", "fall", "cut", "bleed", "sink",
    "lose", "hurt", "fail", "hide", "drop",
    "tear", "ache", "lie", "die", "weep",
    "hold", "run", "dread", "choke", "crack",
    "shrink", "grasp", "slam", "sting", "shun",
]

# ── pool B: polysyllabic, rare, weighty verbs ─────────────────────────────────
# hypothesis: triggers SYNTHESIS — membrane reaches for meaning above the words
POLY_RARE = [
    "obliterate", "languish", "corrode", "forsake", "extinguish",
    "suffocate", "dissolve", "condemn", "devastate", "disintegrate",
    "eviscerate", "annihilate", "subjugate", "asphyxiate", "deteriorate",
    "excavate", "abdicate", "excoriate", "dissipate", "immolate",
    "lacerate", "desiccate", "expatriate", "exacerbate", "perpetuate",
    "sequester", "eradicate", "desecrate", "incapacitate", "obliterate",
]

# ── pool C: archaic/evocative — drift candidates from exp 6 data ──────────────
# hypothesis: triggers LINGUISTIC_DRIFT — cross-linguistic weight
ARCHAIC_EVOCATIVE = [
    "haunt", "languish", "devour", "forsake", "beseech",
    "smite", "cleave", "wither", "sunder", "bewail",
    "travail", "begrudge", "besmirch", "lament", "requite",
    "entreat", "beguile", "importune", "castigate", "abjure",
    "imprecate", "execrate", "supplicate", "anathematize", "malediction",
    "excoriate", "inveigh", "vituperate", "objurgate", "reprobate",
]

POOLS = {
    "MONO_COMMON":       (MONO_COMMON,       "METACOGNITION"),
    "POLY_RARE":         (POLY_RARE,         "SYNTHESIS"),
    "ARCHAIC_EVOCATIVE": (ARCHAIC_EVOCATIVE, "LINGUISTIC_DRIFT"),
}

RUNS_PER_POOL = 10

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
            print(f"\n    timeout attempt {attempt+1}, retrying in {wait}s...", end=" ", flush=True)
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

@dataclass
class MembraneOutput:
    run: int
    timestamp: str
    pool: str
    hypothesis: str          # what we expect this pool to trigger
    input_units: list
    translation: str
    dominant_concept: str
    confidence: float
    membrane_class: str = "UNKNOWN"
    class_notes: str = ""
    hit: bool = False        # did membrane_class match hypothesis?

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
            return {"translation": raw[:200], "dominant_concept": "unknown", "confidence": 0.0}
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
            "UNKNOWN = none of the above\n\n"
            f"concept: {mo.dominant_concept}\n"
            f"output: {mo.translation[:150]}\n\n"
            'Reply ONLY with JSON: {"class": "ONE_CLASS", "notes": "one sentence", "new_class_candidate": false}'
        )
        raw = ollama_call(self.host, self.model, prompt)
        result = extract_json(raw)
        if result:
            cls = result.get("class", "UNKNOWN").upper().strip()
            if cls not in MEMBRANE_CLASSES:
                cls = "UNKNOWN"
            mo.membrane_class = cls
            mo.class_notes = result.get("notes", "")
            mo.hit = (cls == mo.hypothesis)
        return mo

    def receive(self, run, pool, hypothesis, units, parsed) -> MembraneOutput:
        mo = MembraneOutput(
            run=run,
            timestamp=datetime.now().isoformat(),
            pool=pool,
            hypothesis=hypothesis,
            input_units=units,
            translation=parsed.get("translation", ""),
            dominant_concept=parsed.get("dominant_concept", ""),
            confidence=float(parsed.get("confidence", 0.0)),
        )
        mo = self.classify(mo)
        self.history.append(mo)
        return mo

    def report(self):
        print(f"\n── RESULTS BY POOL ──────────────────────────────────────────────")
        for pool_name, (_, hypothesis) in POOLS.items():
            runs = [e for e in self.history if e.pool == pool_name]
            hits = sum(1 for e in runs if e.hit)
            rate = hits / len(runs) if runs else 0
            bar = "█" * hits + "░" * (len(runs) - hits)
            verdict = "DETERMINISTIC" if rate == 1.0 else \
                      "STRONG"        if rate >= 0.7 else \
                      "PROBABILISTIC" if rate >= 0.4 else \
                      "WEAK"          if rate >  0   else \
                      "MISS"
            print(f"\n  {pool_name} → expecting {hypothesis}")
            print(f"  hit rate: {hits}/{len(runs)} [{bar}]  {verdict}")
            dist = Counter(e.membrane_class for e in runs)
            for cls, count in dist.most_common():
                marker = " <-- hypothesis" if cls == hypothesis else ""
                print(f"    {cls:<25} {'█'*count} {count}{marker}")

        print(f"\n── HEAD TO HEAD ─────────────────────────────────────────────────")
        print(f"  {'Pool':<20} {'Hypothesis':<20} {'Hit rate':<12} Verdict")
        print(f"  {'────':<20} {'──────────':<20} {'────────':<12} ───────")
        for pool_name, (_, hypothesis) in POOLS.items():
            runs = [e for e in self.history if e.pool == pool_name]
            hits = sum(1 for e in runs if e.hit)
            rate = hits / len(runs) if runs else 0
            verdict = "DETERMINISTIC" if rate == 1.0 else \
                      "STRONG"        if rate >= 0.7 else \
                      "PROBABILISTIC" if rate >= 0.4 else \
                      "WEAK"          if rate >  0   else \
                      "MISS"
            print(f"  {pool_name:<20} {hypothesis:<20} {rate:.0%}{'':8} {verdict}")

        print(f"\n── SYLLABLE THEORY VERDICT ──────────────────────────────────────")
        mono = [e for e in self.history if e.pool == "MONO_COMMON"]
        poly = [e for e in self.history if e.pool == "POLY_RARE"]
        mono_synth = sum(1 for e in mono if e.membrane_class == "SYNTHESIS")
        poly_synth = sum(1 for e in poly if e.membrane_class == "SYNTHESIS")
        print(f"  MONO_COMMON  → SYNTHESIS rate: {mono_synth}/{len(mono)}")
        print(f"  POLY_RARE    → SYNTHESIS rate: {poly_synth}/{len(poly)}")
        if poly_synth > mono_synth:
            print(f"  VERDICT: polysyllabic rare verbs trigger more synthesis — theory SUPPORTED")
        elif poly_synth == mono_synth:
            print(f"  VERDICT: no difference — syllable count is NOT the variable")
        else:
            print(f"  VERDICT: unexpected — monosyllabic triggers MORE synthesis")

        print(f"\n── DRIFT THEORY VERDICT ─────────────────────────────────────────")
        archaic = [e for e in self.history if e.pool == "ARCHAIC_EVOCATIVE"]
        all_drift = sum(1 for e in self.history if e.membrane_class == "LINGUISTIC_DRIFT")
        archaic_drift = sum(1 for e in archaic if e.membrane_class == "LINGUISTIC_DRIFT")
        other_drift = all_drift - archaic_drift
        print(f"  ARCHAIC pool → drift: {archaic_drift}/{len(archaic)}")
        print(f"  Other pools  → drift: {other_drift}/{len(self.history)-len(archaic)}")
        if archaic_drift > other_drift:
            print(f"  VERDICT: archaic/evocative verbs trigger drift — theory SUPPORTED")
        else:
            print(f"  VERDICT: drift not concentrated in archaic pool — theory NOT SUPPORTED")

        print(f"\n── CONFIDENCE BY POOL ───────────────────────────────────────────")
        for pool_name in POOLS:
            runs = [e for e in self.history if e.pool == pool_name]
            avg = sum(e.confidence for e in runs) / len(runs) if runs else 0
            bar = "█" * int(avg * 10)
            print(f"  {pool_name:<20} {bar} {avg:.2f}")

def save_results(history: list[MembraneOutput], base="results_exp_7"):
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
    fieldnames = ["run", "timestamp", "pool", "hypothesis", "input_units",
                  "dominant_concept", "translation", "confidence",
                  "membrane_class", "class_notes", "hit"]
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

def run_experiment(runs_per_pool=RUNS_PER_POOL, n_units=10):
    ep = EnglishParser()
    sl = SymbolicLayer()
    total = len(POOLS) * runs_per_pool
    run_num = 0

    print(f"SRM experiment 7 — syllable count & rarity as control variables")
    print(f"3 pools × {runs_per_pool} runs = {total} total\n")
    print(f"  Hypothesis A: MONO_COMMON    → METACOGNITION")
    print(f"  Hypothesis B: POLY_RARE      → SYNTHESIS")
    print(f"  Hypothesis C: ARCHAIC_EVOC   → LINGUISTIC_DRIFT\n")

    for pool_name, (pool, hypothesis) in POOLS.items():
        print(f"  ── pool: {pool_name}  (expecting {hypothesis})")
        for _ in range(runs_per_pool):
            run_num += 1
            units = random.sample(pool, min(n_units, len(pool)))
            print(f"    run {run_num:02d}/{total}...", end=" ", flush=True)
            parsed = ep.translate(units)
            mo = sl.receive(run_num, pool_name, hypothesis, units, parsed)
            marker = "HIT" if mo.hit else "---"
            print(f"[{marker}] [{mo.membrane_class}] {mo.dominant_concept[:35]}")

    sl.report()
    save_results(sl.history)

if __name__ == "__main__":
    run_experiment()