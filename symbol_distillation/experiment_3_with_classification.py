import random
import requests
import json
import csv
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

MEMBRANE_CLASSES = {
    "COLLAPSE":    "One word, max confidence, input overwhelm",
    "OVERFLOW":    "Repetitive spiral, loses coherence",
    "SYNTHESIS":   "Builds unified world across all inputs",
    "ANOMALY":     "Finds pattern not obviously in input",
    "COMPRESSION": "2-3 words, clean and strange",
    "UNKNOWN":     "Genuinely does not fit any class",
}

ENGLISH_WORDS = [
    "abscond", "velvet", "theorem", "brine", "oscillate", "crevice", "mandate",
    "phosphor", "dwindle", "quorum", "larder", "vex", "solstice", "tether",
    "grimace", "plinth", "forfeit", "sunder", "mnemonic", "cavort", "dulcimer",
    "siphon", "wretched", "fallow", "torque", "sibilant", "gambit", "plover",
    "estrange", "burnish", "contrite", "nimbus", "falter", "requite", "sallow",
    "dossier", "verdant", "scoff", "lament", "perihelion", "cloister", "fetid",
    "mire", "zealot", "cornice", "oblique", "throttle", "vestige", "clamor",
    "impasse", "sardonic", "winnow", "penumbra", "bivouac", "querulous", "stave",
    "effluent", "mordant", "lissome", "chagrin", "rapine", "solace", "tundra",
    "reliquary", "smolder", "flint", "adroit", "moraine", "splinter", "cabal",
    "dearth", "reverie", "husk", "florid", "ingress", "parch", "sinew", "dirge",
    "canopy", "wither", "thrall", "cipher", "omen", "brood", "cleft", "marrow",
    "pallor", "silt", "ruse", "toil", "blight", "surge", "gaunt", "wend",
    "knell", "murk", "rasp", "prone", "cleave", "dusk", "gnarl", "shroud"
]

# ── resilient HTTP call with retry ────────────────────────────────────────────
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
            print(f"\n    timeout on attempt {attempt+1}, retrying in {wait}s...", end=" ", flush=True)
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
    input_units: list
    translation: str
    dominant_concept: str
    confidence: float
    membrane_class: str = "UNKNOWN"
    class_notes: str = ""
    new_class_candidate: bool = False

# ── components ────────────────────────────────────────────────────────────────
class RandomForce:
    def generate(self, pool=ENGLISH_WORDS):
        return random.choice(pool)

class OrderlyForce:
    def __init__(self, n=10):
        self.n = n
    def collect(self, generator_fn):
        return [generator_fn() for _ in range(self.n)]

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
            "'translation' (1-2 sentences), 'dominant_concept' (1 word), 'confidence' (0.0-1.0).\n\n"
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
        # simplified prompt — forced single-word choice the tiny model can handle
        classes = ", ".join(MEMBRANE_CLASSES.keys())
        prompt = (
            f"Classify this output into exactly one of: {classes}\n\n"
            f"COLLAPSE = single word output\n"
            f"OVERFLOW = repetitive list that spirals\n"
            f"SYNTHESIS = rich interpretation connecting all inputs\n"
            f"ANOMALY = unexpected pattern not in the inputs\n"
            f"COMPRESSION = 2-3 words, precise and strange\n"
            f"UNKNOWN = none of the above\n\n"
            f"concept: {mo.dominant_concept}\n"
            f"output: {mo.translation[:150]}\n\n"
            f'Reply ONLY with JSON: {{"class": "ONE_CLASS", "notes": "one sentence", "new_class_candidate": false}}'
        )
        raw = ollama_call(self.host, self.model, prompt)
        result = extract_json(raw)
        if result:
            cls = result.get("class", "UNKNOWN").upper().strip()
            # validate — reject hallucinated class names
            if cls not in MEMBRANE_CLASSES:
                cls = "UNKNOWN"
            mo.membrane_class = cls
            mo.class_notes = result.get("notes", "")
            mo.new_class_candidate = bool(result.get("new_class_candidate", False))
        return mo

    def receive(self, run, units, parsed) -> MembraneOutput:
        mo = MembraneOutput(
            run=run,
            timestamp=datetime.now().isoformat(),
            input_units=units,
            translation=parsed.get("translation", ""),
            dominant_concept=parsed.get("dominant_concept", ""),
            confidence=float(parsed.get("confidence", 0.0)),
        )
        mo = self.classify(mo)
        return mo

    def record(self, mo: MembraneOutput):
        self.history.append(mo)

    def report(self):
        print("\n── RESULTS ──────────────────────────────────────────────────────")
        for e in self.history:
            flag = " ***" if e.new_class_candidate else ""
            print(f"\nRun {e.run:02d} [{e.membrane_class}]{flag}")
            print(f"  Input:   {' '.join(e.input_units)}")
            print(f"  Concept: {e.dominant_concept}")
            print(f"  Meaning: {e.translation[:120]}")
            print(f"  Conf:    {e.confidence}")
            if e.class_notes:
                print(f"  Notes:   {e.class_notes}")

        print("\n── TAXONOMY DISTRIBUTION ────────────────────────────────────────")
        from collections import Counter
        dist = Counter(e.membrane_class for e in self.history)
        total = len(self.history)
        for cls, count in dist.most_common():
            bar = "█" * count
            print(f"  {cls:<15} {bar} {count}/{total}")

        candidates = [e for e in self.history if e.new_class_candidate]
        if candidates:
            print(f"\n── NEW CLASS CANDIDATES ({len(candidates)}) ──────────────────────")
            for e in candidates:
                print(f"  Run {e.run:02d}: [{e.dominant_concept}] — {e.translation[:100]}")

# ── persistence ───────────────────────────────────────────────────────────────
def save_results(history: list[MembraneOutput], base="results_exp_3"):
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
    fieldnames = ["run", "timestamp", "input_units", "dominant_concept",
                  "translation", "confidence", "membrane_class",
                  "class_notes", "new_class_candidate"]
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
def run_experiment(n_runs=20, n_units=10):
    rf = RandomForce()
    of = OrderlyForce(n=n_units)
    ep = EnglishParser()
    sl = SymbolicLayer()

    print(f"SRM emergence experiment 3 — {n_runs} runs × {n_units} words")
    print(f"Taxonomy: {len(MEMBRANE_CLASSES)} classes (open)\n")

    for i in range(1, n_runs + 1):
        print(f"  run {i:02d}/{n_runs}...", end=" ", flush=True)
        units = of.collect(rf.generate)
        parsed = ep.translate(units)
        mo = sl.receive(i, units, parsed)
        sl.record(mo)
        print(f"[{mo.membrane_class}] {mo.dominant_concept}")

    sl.report()
    save_results(sl.history)

if __name__ == "__main__":
    run_experiment()