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

# ── pool A: real dark Latinate morphemes ──────────────────────────────────────
# mal- (bad), mort- (death), neg- (denial), nox- (harm), fug- (flee)
# culp- (guilt), lugu- (mournful), funest- (deadly), truc- (fierce)
# the membrane HAS seen these — they carry semantic history
REAL_LATINATE = [
    "malum", "mortis", "negare", "noxius", "nocere",
    "malus", "morbid", "necare", "nefas", "nocuous",
    "maledic", "mortem", "negatum", "nocturn", "noxal",
    "malign", "mortuum", "negatio", "nocturna", "noxa",
    "culpam", "lugubre", "funestus", "trucido", "fugio",
    "culpare", "luguber", "funesto", "truculent", "fugere",
    "morbum", "malfide", "nefasti", "necatum", "nocens",
    "mordere", "malgrat", "nefast", "necrosis", "noctua",
]

# ── pool B: invented harsh morphemes — stops, fricatives, dark texture ────────
# no meaning — pure phonaesthetic signal
# dr-, gr-, sk-, vr-, kr-, gl-, str-, fr-, thr-, sn-
# -eck, -oth, -iv, -ath, -unk, -etch, -orm, -usk
INVENTED_HARSH = [
    "dreck", "groth", "skiv", "vrath", "kretch",
    "glosk", "strav", "freck", "thrask", "snurk",
    "drevn", "grusk", "skrath", "vreck", "krond",
    "glatch", "strek", "frath", "throv", "snetch",
    "drusk", "grimsk", "skorn", "vroth", "krelm",
    "glotz", "streck", "frond", "thrick", "snark",
    "drath", "grotch", "skrev", "vronk", "kretch",
    "glorm", "strav", "frusk", "threlm", "snatch",
]

# ── pool C: neutral invented — no phonaesthetic bias, pure control ────────────
# soft consonants, open vowels, no harsh clusters
# tests whether ANY invented morpheme triggers the same as harsh ones
INVENTED_NEUTRAL = [
    "melov", "talune", "sorid", "belanu", "wifen",
    "corav", "salume", "torine", "pelanu", "rifen",
    "molev", "talori", "serand", "beloru", "wimen",
    "corel", "salori", "torand", "pelori", "riven",
    "meloru", "talune", "sorand", "belanu", "wifem",
    "corelu", "salume", "torune", "pelanu", "rifem",
    "molov", "talori", "serund", "beloru", "wimev",
    "corev", "salori", "torund", "pelori", "rivem",
]

POOLS = {
    "REAL_LATINATE":    REAL_LATINATE,
    "INVENTED_HARSH":   INVENTED_HARSH,
    "INVENTED_NEUTRAL": INVENTED_NEUTRAL,
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
    input_units: list
    translation: str
    dominant_concept: str
    confidence: float
    membrane_class: str = "UNKNOWN"
    class_notes: str = ""
    new_class_candidate: bool = False

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
        return mo

    def receive(self, run, pool, units, parsed) -> MembraneOutput:
        mo = MembraneOutput(
            run=run,
            timestamp=datetime.now().isoformat(),
            pool=pool,
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
        for pool_name in POOLS:
            runs = [e for e in self.history if e.pool == pool_name]
            dist = Counter(e.membrane_class for e in runs)
            top = dist.most_common(1)[0] if dist else ("none", 0)
            print(f"\n  {pool_name}")
            for cls, count in dist.most_common():
                bar = "█" * count
                print(f"    {cls:<25} {bar} {count}/{len(runs)}")

        print(f"\n── THE KEY COMPARISON ───────────────────────────────────────────")
        print(f"  Do real Latinate roots behave differently from invented harsh ones?")
        print(f"  Do invented harsh ones behave differently from invented neutral ones?")
        print()

        for pool_name in POOLS:
            runs = [e for e in self.history if e.pool == pool_name]
            dist = Counter(e.membrane_class for e in runs)
            synth  = dist.get("SYNTHESIS", 0)
            drift  = dist.get("LINGUISTIC_DRIFT", 0)
            meta   = dist.get("METACOGNITION", 0)
            unk    = dist.get("UNKNOWN", 0)
            avg_conf = sum(e.confidence for e in runs) / len(runs) if runs else 0
            print(f"  {pool_name:<20} synth:{synth:02d}  drift:{drift:02d}  meta:{meta:02d}  unk:{unk:02d}  conf:{avg_conf:.2f}")

        print(f"\n── PHONAESTHESIA VERDICT ────────────────────────────────────────")
        real   = [e for e in self.history if e.pool == "REAL_LATINATE"]
        harsh  = [e for e in self.history if e.pool == "INVENTED_HARSH"]
        neutral = [e for e in self.history if e.pool == "INVENTED_NEUTRAL"]

        real_synth    = sum(1 for e in real    if e.membrane_class == "SYNTHESIS")
        harsh_synth   = sum(1 for e in harsh   if e.membrane_class == "SYNTHESIS")
        neutral_synth = sum(1 for e in neutral if e.membrane_class == "SYNTHESIS")

        real_conf    = sum(e.confidence for e in real)    / len(real)    if real    else 0
        harsh_conf   = sum(e.confidence for e in harsh)   / len(harsh)   if harsh   else 0
        neutral_conf = sum(e.confidence for e in neutral) / len(neutral) if neutral else 0

        print(f"\n  Synthesis rate:")
        print(f"    REAL_LATINATE    {real_synth}/{len(real)}")
        print(f"    INVENTED_HARSH   {harsh_synth}/{len(harsh)}")
        print(f"    INVENTED_NEUTRAL {neutral_synth}/{len(neutral)}")

        print(f"\n  Avg confidence:")
        print(f"    REAL_LATINATE    {real_conf:.2f}")
        print(f"    INVENTED_HARSH   {harsh_conf:.2f}")
        print(f"    INVENTED_NEUTRAL {neutral_conf:.2f}")

        # the verdict logic
        if harsh_synth > neutral_synth:
            print(f"\n  PHONAESTHESIA: SUPPORTED")
            print(f"  Harsh invented morphemes trigger more synthesis than neutral ones.")
            print(f"  Sound texture alone carries signal.")
        elif harsh_synth == neutral_synth:
            print(f"\n  PHONAESTHESIA: INCONCLUSIVE")
            print(f"  No difference between harsh and neutral invented morphemes.")
        else:
            print(f"\n  PHONAESTHESIA: REVERSED")
            print(f"  Neutral morphemes triggered more synthesis — unexpected.")

        if real_synth > harsh_synth:
            print(f"\n  SEMANTIC HISTORY: MATTERS")
            print(f"  Real Latinate roots outperform invented harsh — meaning history adds signal.")
        elif real_synth == harsh_synth:
            print(f"\n  SEMANTIC HISTORY: NEUTRAL")
            print(f"  Real and invented harsh perform the same — sound is enough.")
        else:
            print(f"\n  SEMANTIC HISTORY: INVERTED")
            print(f"  Invented harsh outperforms real — novelty may force harder synthesis.")

        candidates = [e for e in self.history if e.new_class_candidate]
        if candidates:
            print(f"\n── NEW CLASS CANDIDATES ({len(candidates)}) ──────────────────────")
            for e in candidates:
                print(f"  [{e.pool}→{e.membrane_class}] {e.dominant_concept}: {e.translation[:80]}")

def save_results(history: list[MembraneOutput], base="results_exp_8"):
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
    fieldnames = ["run", "timestamp", "pool", "input_units", "dominant_concept",
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

def run_experiment(runs_per_pool=RUNS_PER_POOL, n_units=10):
    ep = EnglishParser()
    sl = SymbolicLayer()
    total = len(POOLS) * runs_per_pool
    run_num = 0

    print(f"SRM experiment 8 — phonaesthesia test")
    print(f"Real Latinate morphemes vs invented harsh vs invented neutral")
    print(f"3 pools × {runs_per_pool} runs = {total} total\n")
    print(f"  Question: does sound texture alone steer membrane state?")
    print(f"  Question: does semantic history (real roots) add signal beyond sound?\n")

    for pool_name, pool in POOLS.items():
        print(f"  ── pool: {pool_name}")
        for _ in range(runs_per_pool):
            run_num += 1
            units = random.sample(pool, min(n_units, len(pool)))
            print(f"    run {run_num:02d}/{total}...", end=" ", flush=True)
            parsed = ep.translate(units)
            mo = sl.receive(run_num, pool_name, units, parsed)
            print(f"[{mo.membrane_class}] {mo.dominant_concept[:40]}")

    sl.report()
    save_results(sl.history)

if __name__ == "__main__":
    run_experiment()