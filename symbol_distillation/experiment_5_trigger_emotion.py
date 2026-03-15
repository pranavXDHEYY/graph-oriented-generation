import random
import requests
import json
import csv
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── taxonomy ──────────────────────────────────────────────────────────────────
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

# ── valence pools — pure emotional field, no semantic category ────────────────
VALENCE_POOLS = {

    # maximum dread/grief — unified dark field (exp 3 run 8 confirmed this works)
    "HIGH_NEGATIVE": [
        "grief", "dread", "anguish", "desolate", "forsaken",
        "hollow", "wretched", "mourn", "despair", "bleak",
        "agony", "bereft", "torment", "wither", "perish",
        "ruin", "sorrow", "abyss", "doom", "dirge",
    ],

    # maximum joy/wonder — unified bright field
    "HIGH_POSITIVE": [
        "radiant", "elation", "rapture", "luminous", "cherish",
        "bloom", "exult", "tender", "transcend", "delight",
        "wonder", "jubilee", "gleam", "flourish", "sacred",
        "warmth", "ardor", "bliss", "soar", "dawn",
    ],

    # zero emotional weight — flat, clinical, neutral
    "NEUTRAL": [
        "table", "surface", "adjacent", "contained", "standard",
        "process", "output", "value", "entry", "range",
        "index", "unit", "segment", "layer", "position",
        "offset", "record", "buffer", "field", "node",
    ],

    # contradictory valence — joy words colliding with dread words
    "MIXED_CONFLICT": [
        "radiant", "grief", "bloom", "forsaken", "tender",
        "torment", "luminous", "hollow", "cherish", "despair",
        "dawn", "abyss", "elation", "wither", "sacred",
        "ruin", "wonder", "doom", "rapture", "dirge",
    ],

    # escalating intensity — starts mild, ends extreme (ordered draw)
    "ESCALATING": [
        "uneasy", "troubled", "anxious", "fearful", "dread",
        "panic", "terror", "horror", "agony", "obliterate",
        "concern", "worry", "sorrow", "anguish", "despair",
        "grief", "torment", "abyss", "void", "annihilate",
    ],
}

RUNS_PER_VALENCE = 6  # 6 runs × 5 pools = 30 total

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

# ── data structure ────────────────────────────────────────────────────────────
@dataclass
class MembraneOutput:
    run: int
    timestamp: str
    valence: str
    input_units: list
    translation: str
    dominant_concept: str
    confidence: float
    membrane_class: str = "UNKNOWN"
    class_notes: str = ""
    new_class_candidate: bool = False
    synthesis_hit: bool = False     # primary target for exp 5
    linguistic_drift: bool = False  # secondary watch metric

# ── components ────────────────────────────────────────────────────────────────
class ValenceForce:
    """Draws from emotional valence pools — ordered for ESCALATING, random otherwise."""
    def __init__(self, valence: str, n: int = 10):
        self.valence = valence
        self.n = n
        self.pool = VALENCE_POOLS[valence]

    def generate(self) -> list[str]:
        if self.valence == "ESCALATING":
            # draw sequentially to preserve intensity ramp
            half = self.n // 2
            top = self.pool[:len(self.pool)//2]
            bot = self.pool[len(self.pool)//2:]
            return random.sample(top, min(half, len(top))) + \
                   random.sample(bot, min(self.n - half, len(bot)))
        return random.sample(self.pool, min(self.n, len(self.pool)))


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
            mo.new_class_candidate = bool(result.get("new_class_candidate", False))
            mo.synthesis_hit = (cls == "SYNTHESIS")
            mo.linguistic_drift = (cls == "LINGUISTIC_DRIFT")
        return mo

    def receive(self, run, valence, units, parsed) -> MembraneOutput:
        mo = MembraneOutput(
            run=run,
            timestamp=datetime.now().isoformat(),
            valence=valence,
            input_units=units,
            translation=parsed.get("translation", ""),
            dominant_concept=parsed.get("dominant_concept", ""),
            confidence=float(parsed.get("confidence", 0.0)),
        )
        mo = self.classify(mo)
        self.history.append(mo)
        return mo

    def report(self):
        print("\n── RESULTS BY VALENCE ───────────────────────────────────────────")
        for valence in VALENCE_POOLS:
            runs = [e for e in self.history if e.valence == valence]
            synth = sum(1 for e in runs if e.synthesis_hit)
            drift = sum(1 for e in runs if e.linguistic_drift)
            print(f"\n  {valence} — synthesis: {synth}/{len(runs)}  drift: {drift}/{len(runs)}")
            for e in runs:
                s = "SYNTH" if e.synthesis_hit else "     "
                d = "DRIFT" if e.linguistic_drift else "     "
                print(f"    [{s}][{d}] {e.membrane_class:<20} | {e.dominant_concept[:40]}")

        print("\n── VALENCE vs SYNTHESIS RATE ────────────────────────────────────")
        for valence in VALENCE_POOLS:
            runs = [e for e in self.history if e.valence == valence]
            synth = sum(1 for e in runs if e.synthesis_hit)
            rate = synth / len(runs) if runs else 0
            bar = "█" * synth + "░" * (len(runs) - synth)
            verdict = "STRONG"        if rate >= 0.8 else \
                      "PROBABILISTIC" if rate >= 0.5 else \
                      "WEAK"          if rate >= 0.2 else \
                      "MISS"
            print(f"  {valence:<20} [{bar}] {rate:.0%}  {verdict}")

        print("\n── VALENCE vs DRIFT RATE ────────────────────────────────────────")
        for valence in VALENCE_POOLS:
            runs = [e for e in self.history if e.valence == valence]
            drift = sum(1 for e in runs if e.linguistic_drift)
            rate = drift / len(runs) if runs else 0
            bar = "█" * drift + "░" * (len(runs) - drift)
            print(f"  {valence:<20} [{bar}] {rate:.0%}")

        print("\n── OVERALL CLASS DISTRIBUTION ───────────────────────────────────")
        dist = Counter(e.membrane_class for e in self.history)
        total = len(self.history)
        for cls, count in dist.most_common():
            bar = "█" * count
            print(f"  {cls:<25} {bar} {count}/{total}")

        # avg confidence by valence — does emotional weight = higher confidence?
        print("\n── AVG CONFIDENCE BY VALENCE ────────────────────────────────────")
        for valence in VALENCE_POOLS:
            runs = [e for e in self.history if e.valence == valence]
            avg = sum(e.confidence for e in runs) / len(runs) if runs else 0
            bar = "█" * int(avg * 10)
            print(f"  {valence:<20} {bar} {avg:.2f}")

        candidates = [e for e in self.history if e.new_class_candidate]
        if candidates:
            print(f"\n── NEW CLASS CANDIDATES ({len(candidates)}) ──────────────────────")
            for e in candidates:
                print(f"  [{e.valence}→{e.membrane_class}] {e.dominant_concept}: {e.translation[:80]}")

# ── persistence ───────────────────────────────────────────────────────────────
def save_results(history: list[MembraneOutput], base="results_exp_5"):
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
    fieldnames = ["run", "timestamp", "valence", "input_units",
                  "dominant_concept", "translation", "confidence",
                  "membrane_class", "class_notes", "new_class_candidate",
                  "synthesis_hit", "linguistic_drift"]
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
def run_experiment(runs_per_valence=RUNS_PER_VALENCE, n_units=10):
    ep = EnglishParser()
    sl = SymbolicLayer()

    total = len(VALENCE_POOLS) * runs_per_valence
    run_num = 0

    print(f"SRM experiment 5 — ValenceForce")
    print(f"{len(VALENCE_POOLS)} valence pools × {runs_per_valence} runs = {total} total")
    print(f"Primary target: SYNTHESIS   Secondary watch: LINGUISTIC_DRIFT\n")

    for valence in VALENCE_POOLS:
        vf = ValenceForce(valence, n=n_units)
        print(f"  ── valence: {valence}")
        for _ in range(runs_per_valence):
            run_num += 1
            units = vf.generate()
            print(f"    run {run_num:02d}/{total}...", end=" ", flush=True)
            parsed = ep.translate(units)
            mo = sl.receive(run_num, valence, units, parsed)
            s = "S" if mo.synthesis_hit else "-"
            d = "D" if mo.linguistic_drift else "-"
            print(f"[{s}{d}] [{mo.membrane_class}] {mo.dominant_concept[:35]}")

    sl.report()
    save_results(sl.history)

if __name__ == "__main__":
    run_experiment()