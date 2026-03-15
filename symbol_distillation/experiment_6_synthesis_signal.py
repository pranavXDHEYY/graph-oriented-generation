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

# ── negative action verbs — dark agency, causation, destruction ───────────────
NEGATIVE_VERBS = [
    # destruction / ending
    "shatter", "sever", "obliterate", "consume", "collapse",
    "fracture", "devour", "extinguish", "erode", "dissolve",
    # abandonment / separation
    "forsake", "abandon", "exile", "sunder", "expel",
    "betray", "isolate", "exclude", "discard", "flee",
    # decay / diminishment
    "wither", "dwindle", "corrode", "rot", "decay",
    "languish", "fester", "crumble", "fade", "hollow",
    # violence / force
    "crush", "rend", "pierce", "strike", "seize",
    "ravage", "wound", "scar", "bleed", "break",
    # psychological
    "torment", "haunt", "suffocate", "drown", "silence",
    "bury", "condemn", "betray", "suppress", "unmake",
]

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
    input_units: list
    translation: str
    dominant_concept: str
    confidence: float
    membrane_class: str = "UNKNOWN"
    class_notes: str = ""
    synthesis_hit: bool = False
    streak: int = 0

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
        self._streak = 0

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
            mo.synthesis_hit = (cls == "SYNTHESIS")
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
        if mo.synthesis_hit:
            self._streak += 1
        else:
            self._streak = 0
        mo.streak = self._streak
        self.history.append(mo)
        return mo

    def report(self):
        total = len(self.history)
        hits = sum(1 for e in self.history if e.synthesis_hit)
        rate = hits / total if total else 0
        max_streak = max((e.streak for e in self.history), default=0)
        first_miss = next((e.run for e in self.history if not e.synthesis_hit), None)

        print(f"\n── NEGATIVE VERB SYNTHESIS TEST ─────────────────────────────────")
        print(f"  Total runs:     {total}")
        print(f"  Synthesis hits: {hits}/{total}  ({rate:.0%})")
        print(f"  Max streak:     {max_streak}")
        print(f"  First miss:     run {first_miss}" if first_miss else "  First miss:     NONE")

        print(f"\n── RUN LOG ──────────────────────────────────────────────────────")
        for e in self.history:
            marker = "S" if e.synthesis_hit else "X"
            streak_str = f"streak:{e.streak}" if e.synthesis_hit else "BREAK "
            print(f"  [{marker}] run {e.run:03d} {streak_str:<12} "
                  f"conf:{e.confidence:.2f}  {e.membrane_class:<20} | {e.dominant_concept[:40]}")

        print(f"\n── BREAKS ───────────────────────────────────────────────────────")
        breaks = [e for e in self.history if not e.synthesis_hit]
        if not breaks:
            print("  none — negative verbs held all the way through")
        else:
            for e in breaks:
                print(f"  run {e.run:03d} [{e.membrane_class}]  {e.dominant_concept}")
                print(f"    input: {' '.join(e.input_units)}")
                print(f"    meaning: {e.translation[:100]}")

        print(f"\n── VS HIGH_NEGATIVE NOUNS (exp 6 baseline) ──────────────────────")
        print(f"  HIGH_NEGATIVE nouns:  ~50%  probabilistic")
        print(f"  NEGATIVE_VERBS:       {rate:.0%}  {'stronger' if rate > 0.5 else 'weaker' if rate < 0.5 else 'same'}")

        print(f"\n── CLASS DISTRIBUTION ───────────────────────────────────────────")
        dist = Counter(e.membrane_class for e in self.history)
        for cls, count in dist.most_common():
            bar = "█" * count + "░" * (total - count)
            print(f"  {cls:<25} {bar} {count}/{total}")

        print(f"\n── AVG CONFIDENCE ───────────────────────────────────────────────")
        avg_conf = sum(e.confidence for e in self.history) / total if total else 0
        synth_conf = sum(e.confidence for e in self.history if e.synthesis_hit)
        synth_n = sum(1 for e in self.history if e.synthesis_hit)
        avg_synth_conf = synth_conf / synth_n if synth_n else 0
        print(f"  Overall:          {avg_conf:.2f}")
        print(f"  Synthesis runs:   {avg_synth_conf:.2f}")

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
    fieldnames = ["run", "timestamp", "input_units", "dominant_concept",
                  "translation", "confidence", "membrane_class",
                  "class_notes", "synthesis_hit", "streak"]
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

def run_experiment(n_runs=30, n_units=10):
    ep = EnglishParser()
    sl = SymbolicLayer()

    print(f"SRM experiment 7 — negative action verbs")
    print(f"Hypothesis: verbs trigger synthesis more reliably than nouns")
    print(f"Baseline: HIGH_NEGATIVE nouns ~50% | Target: beat 50%\n")

    for i in range(1, n_runs + 1):
        units = random.sample(NEGATIVE_VERBS, min(n_units, len(NEGATIVE_VERBS)))
        print(f"  run {i:03d}/{n_runs}...", end=" ", flush=True)
        parsed = ep.translate(units)
        mo = sl.receive(i, units, parsed)
        marker = "S" if mo.synthesis_hit else "X"
        streak = f"[streak:{mo.streak}]" if mo.synthesis_hit else "[BREAK]"
        print(f"[{marker}] {streak} {mo.dominant_concept[:40]}")

    sl.report()
    save_results(sl.history)

if __name__ == "__main__":
    run_experiment()