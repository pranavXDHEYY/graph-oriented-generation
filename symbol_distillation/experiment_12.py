#!/usr/bin/env python3
"""
Experiment 12: Single Membrane vs Pipeline Architecture
HYPOTHESIS: Specialized membranes in sequence outperform one membrane
            asked to do everything simultaneously.
ARCHITECTURE A: One membrane, three roles stacked in one prompt
ARCHITECTURE B: Three membranes in sequence, each with one role
                Input → [MEANING] → [VOICE] → [STRUCTURE] → Output
The failure mode of Architecture A tells us what the pipeline must handle.
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

# ── shared input pools (same inputs hit both architectures) ──────────────────
CONSONANT_CLUSTERS = [
    "btr", "klp", "shn", "grm", "dvw", "xzq", "fth", "bkl",
    "mns", "rpt", "vzh", "cqx", "wgl", "hjd", "tfk", "znp",
]
NONSENSE_SYLLABLES = [
    "krav", "plith", "zor", "mend", "cath", "rux",
    "gol", "fip", "nuth", "wra", "cliv", "zeph",
    "tor", "vax", "pum", "krel", "snor", "wix",
]
HIGH_NEGATIVE = [
    "grief", "dread", "anguish", "desolate", "forsaken",
    "hollow", "wretched", "mourn", "despair", "bleak",
    "agony", "bereft", "torment", "wither", "perish",
]
FUNCTION_WORDS = [
    "the", "and", "of", "in", "to", "a", "is", "that",
    "for", "with", "as", "by", "on", "be", "at", "or",
]

INPUT_POOLS = {
    "CONSONANT":  CONSONANT_CLUSTERS,
    "NONSENSE":   NONSENSE_SYLLABLES,
    "EMOTIONAL":  HIGH_NEGATIVE,
    "STRUCTURAL": FUNCTION_WORDS,
}

RUNS_PER_POOL = 5

# ── prompts ───────────────────────────────────────────────────────────────────

PROMPT_SINGLE_MULTI = """You are a language membrane with three simultaneous roles.
Given these fragments, respond to ALL THREE of the following:

1. MEANING: What does this mean? Find the concept.
2. VOICE: What does this SOUND like it means? Give it a voice.
3. STRUCTURE: What relational or grammatical pattern do you perceive?

Fragments: {fragments}

Output ONLY JSON with keys:
"meaning" (1 sentence),
"voice" (1 sentence — give the sounds a voice),
"structure" (1 sentence — describe the pattern),
"dominant_concept" (1 word that unifies all three),
"confidence" (0.0-1.0),
"collapsed" (true if you could not maintain all three roles)"""

PROMPT_PIPELINE_MEANING = """You are a meaning membrane.
Find whatever concept or feeling emerges from these fragments.
Output ONLY JSON: {{"concept": "one word", "translation": "one sentence", "confidence": 0.0-1.0}}

Fragments: {fragments}"""

PROMPT_PIPELINE_VOICE = """You are a voice membrane.
The fragments below have already been interpreted as: {meaning}
Now give them a VOICE — what are they trying to express emotionally?
Output ONLY JSON: {{"voice": "one sentence", "emotion": "one word", "intensity": 0.0-1.0}}

Fragments: {fragments}"""

PROMPT_PIPELINE_STRUCTURE = """You are a structure membrane.
The fragments below carry meaning ({meaning}) and voice ({voice}).
Now perceive the underlying PATTERN or STRUCTURE — grammatical, relational, symbolic.
Output ONLY JSON: {{"structure": "one sentence", "pattern_type": "one word", "coherence": 0.0-1.0}}

Fragments: {fragments}"""

# ── result structures ─────────────────────────────────────────────────────────

@dataclass
class SingleMembraneResult:
    run: int
    timestamp: str
    pool: str
    input_units: str
    meaning: str = ""
    voice: str = ""
    structure: str = ""
    dominant_concept: str = ""
    confidence: float = 0.0
    collapsed: bool = False
    roles_present: int = 0      # how many of 3 roles actually appeared
    parse_failed: bool = False

@dataclass
class PipelineResult:
    run: int
    timestamp: str
    pool: str
    input_units: str
    # stage outputs
    meaning_concept: str = ""
    meaning_translation: str = ""
    meaning_confidence: float = 0.0
    voice_text: str = ""
    voice_emotion: str = ""
    voice_intensity: float = 0.0
    structure_text: str = ""
    structure_pattern: str = ""
    structure_coherence: float = 0.0
    # pipeline health
    stages_completed: int = 0
    pipeline_broke_at: str = ""  # empty = completed fully

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
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {}

# ── Architecture A: single membrane, multi-role ───────────────────────────────

def run_single_membrane(run, pool, units, host, model) -> SingleMembraneResult:
    r = SingleMembraneResult(
        run=run,
        timestamp=datetime.now().isoformat(),
        pool=pool,
        input_units=" ".join(units),
    )
    prompt = PROMPT_SINGLE_MULTI.format(fragments=" ".join(units))
    raw = ollama_call(host, model, prompt)
    result = extract_json(raw)

    if not result:
        r.parse_failed = True
        return r

    r.meaning          = result.get("meaning", "")
    r.voice            = result.get("voice", "")
    r.structure        = result.get("structure", "")
    r.dominant_concept = result.get("dominant_concept", "")
    r.confidence       = float(result.get("confidence", 0.0))
    r.collapsed        = bool(result.get("collapsed", False))

    # count how many roles actually produced content
    r.roles_present = sum([
        len(r.meaning) > 10,
        len(r.voice) > 10,
        len(r.structure) > 10,
    ])
    return r

# ── Architecture B: pipeline of three specialized membranes ──────────────────

def run_pipeline(run, pool, units, host, model) -> PipelineResult:
    r = PipelineResult(
        run=run,
        timestamp=datetime.now().isoformat(),
        pool=pool,
        input_units=" ".join(units),
    )
    frags = " ".join(units)

    # stage 1 — meaning
    raw1 = ollama_call(host, model,
                       PROMPT_PIPELINE_MEANING.format(fragments=frags))
    s1 = extract_json(raw1)
    if not s1:
        r.pipeline_broke_at = "MEANING"
        return r
    r.meaning_concept    = s1.get("concept", "")
    r.meaning_translation = s1.get("translation", "")
    r.meaning_confidence = float(s1.get("confidence", 0.0))
    r.stages_completed   = 1

    # stage 2 — voice (receives meaning from stage 1)
    raw2 = ollama_call(host, model,
                       PROMPT_PIPELINE_VOICE.format(
                           fragments=frags,
                           meaning=r.meaning_translation))
    s2 = extract_json(raw2)
    if not s2:
        r.pipeline_broke_at = "VOICE"
        return r
    r.voice_text      = s2.get("voice", "")
    r.voice_emotion   = s2.get("emotion", "")
    r.voice_intensity = float(s2.get("intensity", 0.0))
    r.stages_completed = 2

    # stage 3 — structure (receives meaning + voice)
    raw3 = ollama_call(host, model,
                       PROMPT_PIPELINE_STRUCTURE.format(
                           fragments=frags,
                           meaning=r.meaning_translation,
                           voice=r.voice_text))
    s3 = extract_json(raw3)
    if not s3:
        r.pipeline_broke_at = "STRUCTURE"
        return r
    r.structure_text    = s3.get("structure", "")
    r.structure_pattern = s3.get("pattern_type", "")
    r.structure_coherence = float(s3.get("coherence", 0.0))
    r.stages_completed  = 3
    return r

# ── reporting ─────────────────────────────────────────────────────────────────

def report(single_results, pipeline_results):
    print("\n── ARCHITECTURE A: SINGLE MULTI-ROLE MEMBRANE ───────────────────")
    total_s = len(single_results)
    collapsed   = sum(1 for r in single_results if r.collapsed)
    parse_failed = sum(1 for r in single_results if r.parse_failed)
    all_3_roles  = sum(1 for r in single_results if r.roles_present == 3)
    two_roles    = sum(1 for r in single_results if r.roles_present == 2)
    one_role     = sum(1 for r in single_results if r.roles_present == 1)
    avg_conf     = sum(r.confidence for r in single_results) / total_s if total_s else 0

    print(f"  Total runs:        {total_s}")
    print(f"  All 3 roles filled: {all_3_roles}/{total_s}  ({all_3_roles/total_s:.0%})")
    print(f"  2 roles filled:     {two_roles}/{total_s}")
    print(f"  1 role filled:      {one_role}/{total_s}")
    print(f"  Self-reported collapse: {collapsed}/{total_s}")
    print(f"  Parse failures:    {parse_failed}/{total_s}")
    print(f"  Avg confidence:    {avg_conf:.2f}")

    print(f"\n  By pool:")
    for pool in INPUT_POOLS:
        runs = [r for r in single_results if r.pool == pool]
        if not runs:
            continue
        full = sum(1 for r in runs if r.roles_present == 3)
        print(f"    {pool:<12} full:{full}/{len(runs)}  "
              f"avg_conf:{sum(r.confidence for r in runs)/len(runs):.2f}")

    print("\n── ARCHITECTURE B: THREE-STAGE PIPELINE ─────────────────────────")
    total_p = len(pipeline_results)
    completed_all = sum(1 for r in pipeline_results if r.stages_completed == 3)
    broke_meaning  = sum(1 for r in pipeline_results if r.pipeline_broke_at == "MEANING")
    broke_voice    = sum(1 for r in pipeline_results if r.pipeline_broke_at == "VOICE")
    broke_structure = sum(1 for r in pipeline_results if r.pipeline_broke_at == "STRUCTURE")
    avg_stages     = sum(r.stages_completed for r in pipeline_results) / total_p if total_p else 0
    avg_coherence  = sum(r.structure_coherence for r in pipeline_results
                         if r.stages_completed == 3)
    n_coherence    = sum(1 for r in pipeline_results if r.stages_completed == 3)
    avg_coherence  = avg_coherence / n_coherence if n_coherence else 0

    print(f"  Total runs:         {total_p}")
    print(f"  Completed all 3:    {completed_all}/{total_p}  ({completed_all/total_p:.0%})")
    print(f"  Broke at MEANING:   {broke_meaning}")
    print(f"  Broke at VOICE:     {broke_voice}")
    print(f"  Broke at STRUCTURE: {broke_structure}")
    print(f"  Avg stages completed: {avg_stages:.1f}/3")
    print(f"  Avg structure coherence (completed): {avg_coherence:.2f}")

    print(f"\n  By pool:")
    for pool in INPUT_POOLS:
        runs = [r for r in pipeline_results if r.pool == pool]
        if not runs:
            continue
        full = sum(1 for r in runs if r.stages_completed == 3)
        print(f"    {pool:<12} full:{full}/{len(runs)}  "
              f"avg_stages:{sum(r.stages_completed for r in runs)/len(runs):.1f}")

    print("\n── HEAD TO HEAD ─────────────────────────────────────────────────")
    print(f"  Single membrane — full output rate: {all_3_roles/total_s:.0%}")
    print(f"  Pipeline        — full output rate: {completed_all/total_p:.0%}")

    if completed_all/total_p > all_3_roles/total_s + 0.1:
        print(f"\n  VERDICT: PIPELINE WINS — specialization produces more complete output")
        print(f"  Architecture recommendation: specialized membrane pipeline")
    elif all_3_roles/total_s > completed_all/total_p + 0.1:
        print(f"\n  VERDICT: SINGLE MEMBRANE WINS — multi-role is more robust")
        print(f"  Architecture recommendation: one role-conditioned membrane")
    else:
        print(f"\n  VERDICT: COMPARABLE — difference within noise margin")
        print(f"  Look at output quality, not just completion rate")

    print("\n── SAMPLE OUTPUTS (first completed run each architecture) ───────")
    for r in single_results:
        if r.roles_present == 3:
            print(f"\n  SINGLE [{r.pool}] input: {r.input_units[:40]}")
            print(f"    meaning:   {r.meaning[:80]}")
            print(f"    voice:     {r.voice[:80]}")
            print(f"    structure: {r.structure[:80]}")
            print(f"    concept:   {r.dominant_concept}")
            break

    for r in pipeline_results:
        if r.stages_completed == 3:
            print(f"\n  PIPELINE [{r.pool}] input: {r.input_units[:40]}")
            print(f"    meaning:   {r.meaning_translation[:80]}")
            print(f"    voice:     {r.voice_text[:80]}")
            print(f"    structure: {r.structure_text[:80]}")
            break

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(single_results, pipeline_results, base="results_exp_12"):
    # single
    json_path = Path(f"{base}_single.json")
    with open(json_path, "w") as f:
        json.dump([{**asdict(r), } for r in single_results], f, indent=2)

    # pipeline
    json_path2 = Path(f"{base}_pipeline.json")
    with open(json_path2, "w") as f:
        json.dump([asdict(r) for r in pipeline_results], f, indent=2)

    # combined CSV for easy analysis
    csv_path = Path(f"{base}_combined.csv")
    rows = []
    for r in single_results:
        rows.append({
            "architecture": "SINGLE",
            "run": r.run, "pool": r.pool,
            "input": r.input_units[:40],
            "roles_present": r.roles_present,
            "collapsed": r.collapsed,
            "confidence": r.confidence,
            "concept": r.dominant_concept,
        })
    for r in pipeline_results:
        rows.append({
            "architecture": "PIPELINE",
            "run": r.run, "pool": r.pool,
            "input": r.input_units[:40],
            "roles_present": r.stages_completed,
            "collapsed": r.pipeline_broke_at != "",
            "confidence": r.structure_coherence,
            "concept": r.meaning_concept,
        })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n── SAVED ────────────────────────────────────────────────────────")
    print(f"  {base}_single.json")
    print(f"  {base}_pipeline.json")
    print(f"  {base}_combined.csv")

# ── main ──────────────────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434", model="qwen2.5:0.5b"):
    single_results   = []
    pipeline_results = []
    run_num = 0
    total = len(INPUT_POOLS) * RUNS_PER_POOL * 2

    print(f"SRM experiment 12 — single membrane vs pipeline")
    print(f"4 input pools × {RUNS_PER_POOL} runs × 2 architectures = {total} total calls")
    print(f"(pipeline uses 3 calls per run = {len(INPUT_POOLS)*RUNS_PER_POOL*3} Ollama calls for B)\n")

    for pool_name, pool in INPUT_POOLS.items():
        print(f"\n  ── pool: {pool_name}")

        for _ in range(RUNS_PER_POOL):
            run_num += 1
            units = random.sample(pool, min(10, len(pool)))

            # Architecture A
            print(f"    run {run_num:02d} [SINGLE  ]...", end=" ", flush=True)
            sr = run_single_membrane(run_num, pool_name, units, host, model)
            single_results.append(sr)
            status = f"roles:{sr.roles_present}/3" if not sr.parse_failed else "PARSE_FAIL"
            print(f"{status}  conf:{sr.confidence:.2f}  {sr.dominant_concept[:20]}")

            # Architecture B
            print(f"    run {run_num:02d} [PIPELINE]...", end=" ", flush=True)
            pr = run_pipeline(run_num, pool_name, units, host, model)
            pipeline_results.append(pr)
            status = f"stages:{pr.stages_completed}/3" if not pr.pipeline_broke_at \
                     else f"BROKE@{pr.pipeline_broke_at}"
            print(f"{status}  {pr.meaning_concept[:20]}")

    report(single_results, pipeline_results)
    save_results(single_results, pipeline_results)

if __name__ == "__main__":
    import sys
    host  = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:0.5b"
    run_experiment(host, model)