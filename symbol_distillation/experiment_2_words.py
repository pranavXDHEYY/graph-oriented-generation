import random
import requests
import json

# ── English word pool (no frequency bias — broad dictionary sample) ──────────
# Using a wide spread across rare, common, abstract, concrete
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

# ── the four components ──────────────────────────────────────────────────────

class RandomForce:
    """Pure chaos — draws random English words with no frequency bias."""
    def generate(self, pool=ENGLISH_WORDS):
        return random.choice(pool)


class OrderlyForce:
    """Imposes structure — collects exactly N units."""
    def __init__(self, n=10):
        self.n = n

    def collect(self, generator_fn):
        return [generator_fn() for _ in range(self.n)]


class EnglishParser:
    """The membrane — identical prompt to experiment_1 for pure comparison."""
    def __init__(self, model="qwen2.5:0.5b", host="http://localhost:11434"):
        self.model = model
        self.host = host

    def translate(self, units: list[str]) -> dict:
        raw = " ".join(units)
        prompt = (
            f"You are a minimal language membrane. "
            f"Below are raw phoneme fragments. "
            f"Find whatever meaning emerges — a concept, a feeling, a pattern. "
            f"Output ONLY a JSON object with keys: "
            f"'translation' (1-2 sentences), 'dominant_concept' (1 word), 'confidence' (0.0-1.0).\n\n"
            f"Fragments: {raw}"
        )
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=30
        )
        raw_output = response.json().get("response", "")
        try:
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            return json.loads(raw_output[start:end])
        except Exception:
            return {"translation": raw_output, "dominant_concept": "unknown", "confidence": 0.0}


class SymbolicLayer:
    """Receives whatever the membrane outputs — no predetermined format."""
    def __init__(self):
        self.history = []

    def receive(self, units: list[str], parsed: dict):
        entry = {
            "run": len(self.history) + 1,
            "input_units": units,
            "emerged": parsed
        }
        self.history.append(entry)
        return entry

    def report(self):
        for e in self.history:
            print(f"\n── Run {e['run']} ──")
            print(f"  Input:   {' '.join(e['input_units'])}")
            print(f"  Concept: {e['emerged'].get('dominant_concept', '?')}")
            print(f"  Meaning: {e['emerged'].get('translation', '?')}")
            print(f"  Confidence: {e['emerged'].get('confidence', '?')}")


# ── experiment loop ──────────────────────────────────────────────────────────

def run_experiment(n_runs=5, n_units=10):
    rf = RandomForce()
    of = OrderlyForce(n=n_units)
    ep = EnglishParser()
    sl = SymbolicLayer()

    print(f"Starting SRM emergence experiment 2 — English words: {n_runs} runs × {n_units} words\n")

    for _ in range(n_runs):
        units = of.collect(rf.generate)
        parsed = ep.translate(units)
        sl.receive(units, parsed)

    sl.report()

if __name__ == "__main__":
    run_experiment()