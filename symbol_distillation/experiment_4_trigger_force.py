import random
import requests
import json
import csv
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── updated taxonomy ──────────────────────────────────────────────────────────
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

# ── intentional trigger pools ─────────────────────────────────────────────────
TRIGGER_POOLS = {

    "LINGUISTIC_DRIFT": [
        # words that strain at English edges — heavy with untranslatable weight
        "solstice", "perihelion", "penumbra", "reverie", "tundra",
        "nimbus", "dirge", "requiem", "vestige", "reliquary",
        "sunder", "cleave", "thrall", "wither", "smolder",
        "pallor", "shroud", "dusk", "knell", "elegy",
    ],

    "METACOGNITION": [
        # self-referential, meta, structural — words about meaning itself
        "cipher", "theorem", "mnemonic", "mandate", "pattern",
        "signal", "meaning", "fragment", "translate", "symbol",
        "code", "syntax", "parse", "token", "index",
        "schema", "map", "trace", "model", "frame",
    ],

    "SYNTHESIS": [
        # unified emotional field — all dread/grief tonal cluster
        "grimace", "husk", "marrow", "cloister", "estrange",
        "bivouac", "brood", "falter", "cabal", "wretched",
        "fetid", "gaunt", "blight", "dirge", "sallow",
        "lament", "mire", "dearth", "pallor", "knell",
    ],

    "ANOMALY": [
        # maximally incoherent — one word from five unrelated domains
        # science, cuisine, warfare, botany, architecture, music, geology
        "theorem", "larder", "gambit", "plover", "cornice",
        "dulcimer", "moraine", "siphon", "burnish", "quorum",
        "torque", "cavort", "phosphor", "verdant", "plinth",
        "oscillate", "forfeit", "lissome", "stave", "adroit",
    ],

    "STRUCTURAL_FAILURE": [
        # attempt to reproduce Run 19 — overload with repetition + abstraction
        "solstice", "solstice", "crevice", "gnarl", "cabal",
        "lament", "wend", "brood", "abscond", "solstice",
        "meaning", "meaning", "pattern", "pattern", "unknown",
        "fragment", "fragment", "signal", "signal", "void",
    ],

    "METACOGNITION": [
        "cipher", "theorem", "mnemonic", "mandate", "pattern",
        "signal", "meaning", "fragment", "translate", "symbol",
        "code", "syntax", "parse", "token", "index",
        "schema", "map", "trace", "model", "frame",
    ],
}

# deduplicate keys (Python dicts take last value for dupes)
# METACOGNITION was listed twice above as an example — clean version below
TRIGGER_POOLS = {
    "LINGUISTIC_DRIFT": [
        "solstice", "perihelion", "penumbra", "reverie", "tundra",
        "nimbus", "dirge", "requiem", "vestige", "reliquary",
        "sunder", "cleave", "thrall", "wither", "smolder",
        "pallor", "shroud", "dusk", "knell", "elegy",
    ],
    "METACOGNITION": [
        "cipher", "theorem", "mnemonic", "mandate", "pattern",
        "signal", "meaning", "fragment", "translate", "symbol",
        "code", "syntax", "parse", "token", "index",
        "schema", "map", "trace", "model", "frame",
    ],
    "SYNTHESIS": [
        "grimace", "husk", "marrow", "cloister", "estrange",
        "bivouac", "brood", "falter", "cabal", "wretched",
        "fetid", "gaunt", "blight", "dirge", "sallow",
        "lament", "mire", "dearth", "pallor", "knell",
    ],
    "ANOMALY": [
        "theorem", "larder", "gambit", "plover", "cornice",
        "dulcimer", "moraine", "siphon", "burnish", "quorum",
        "torque", "cavort", "phosphor", "verdant", "plinth",
        "oscillate", "forfeit", "lissome", "stave", "adroit",
    ],
    "STRUCTURAL_FAILURE": [
        "solstice", "solstice", "crevice", "gnarl", "cabal",
        "lament", "wend", "brood", "abscond", "solstice",
        "meaning", "meaning", "pattern", "pattern", "unknown",
        "fragment", "fragment", "signal", "signal", "void",
    ],
}

RUNS_PER_TRIGGER = 5  # enough to measure probabilistic vs deterministic

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
    intended_trigger: str       # what we were TRYING to produce
    input_units: list
    translation: str
    dominant_concept: str
    confidence: float
    membrane_class: str = "UNKNOWN"
    class_notes: str = ""
    new_class_candidate: bool = False
    hit: bool = False           # did intended_trigger == membrane_class?

# ── components ────────────────────────────────────────────────────────────────
class TriggerForce:
    """Draws from intentional semantic pools instead of random selection."""
    def __init__(self, trigger: str, n: int = 10):
        self.trigger = trigger
        self.n = n
        self.pool = TRIGGER_POOLS[trigger]

    def generate(self) -> list[str]:
        return random.sample(self.pool, min(self.n, len(self.pool)))


class OrderlyForce:
    def __init__(self, n=10):
        self.n = n


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
            mo.hit = (cls == mo.intended_trigger)
        return mo

    def receive(self, run, intended_trigger, units, parsed) -> MembraneOutput:
        mo = MembraneOutput(
            run=run,
            timestamp=datetime.now().isoformat(),
            intended_trigger=intended_trigger,
            input_units=units,
            translation=parsed.get("translation", ""),
            dominant_concept=parsed.get("dominant_concept", ""),
            confidence=float(parsed.get("confidence", 0.0)),
        )
        mo = self.classify(mo)
        self.history.append(mo)
        return mo

    def report(self):
        print("\n── RESULTS BY TRIGGER ───────────────────────────────────────────")
        for trigger in TRIGGER_POOLS:
            runs = [e for e in self.history if e.intended_trigger == trigger]
            hits = sum(1 for e in runs if e.hit)
            hit_rate = hits / len(runs) if runs else 0
            bar = "█" * hits + "░" * (len(runs) - hits)
            print(f"\n  {trigger} — hit rate: {hits}/{len(runs)} [{bar}]")
            for e in runs:
                marker = "HIT " if e.hit else "miss"
                print(f"    [{marker}] got {e.membrane_class:<20} | {e.dominant_concept}")

        print("\n── OVERALL DISTRIBUTION ─────────────────────────────────────────")
        dist = Counter(e.membrane_class for e in self.history)
        total = len(self.history)
        for cls, count in dist.most_common():
            bar = "█" * count
            print(f"  {cls:<25} {bar} {count}/{total}")

        print("\n── TRIGGER RELIABILITY SUMMARY ──────────────────────────────────")
        for trigger in TRIGGER_POOLS:
            runs = [e for e in self.history if e.intended_trigger == trigger]
            hits = sum(1 for e in runs if e.hit)
            hit_rate = hits / len(runs) if runs else 0
            verdict = "DETERMINISTIC" if hit_rate == 1.0 else \
                      "STRONG"        if hit_rate >= 0.8 else \
                      "PROBABILISTIC" if hit_rate >= 0.4 else \
                      "WEAK"          if hit_rate > 0   else \
                      "MISS"
            print(f"  {trigger:<25} {hit_rate:.0%}  →  {verdict}")

        candidates = [e for e in self.history if e.new_class_candidate]
        if candidates:
            print(f"\n── NEW CLASS CANDIDATES ({len(candidates)}) ──────────────────────")
            for e in candidates:
                print(f"  [{e.intended_trigger}→{e.membrane_class}] {e.dominant_concept}: {e.translation[:80]}")

# ── persistence ───────────────────────────────────────────────────────────────
def save_results(history: list[MembraneOutput], base="results_exp_4"):
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
    fieldnames = ["run", "timestamp", "intended_trigger", "input_units",
                  "dominant_concept", "translation", "confidence",
                  "membrane_class", "class_notes", "new_class_candidate", "hit"]
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
def run_experiment(runs_per_trigger=RUNS_PER_TRIGGER, n_units=10):
    ep = EnglishParser()
    sl = SymbolicLayer()

    total = len(TRIGGER_POOLS) * runs_per_trigger
    run_num = 0

    print(f"SRM experiment 4 — TriggerForce")
    print(f"{len(TRIGGER_POOLS)} triggers × {runs_per_trigger} runs = {total} total")
    print(f"Taxonomy: {len(MEMBRANE_CLASSES)} classes\n")

    for trigger in TRIGGER_POOLS:
        tf = TriggerForce(trigger, n=n_units)
        print(f"  ── trigger: {trigger}")
        for _ in range(runs_per_trigger):
            run_num += 1
            units = tf.generate()
            print(f"    run {run_num:02d}/{total}...", end=" ", flush=True)
            parsed = ep.translate(units)
            mo = sl.receive(run_num, trigger, units, parsed)
            marker = "HIT" if mo.hit else "---"
            print(f"[{marker}] [{mo.membrane_class}] {mo.dominant_concept}")

    sl.report()
    save_results(sl.history)

if __name__ == "__main__":
    run_experiment()