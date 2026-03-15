#!/usr/bin/env python3
"""
Experiment 11: Voice of the Membrane - Sound Without Reason

HYPOTHESIS:
When we give the LLM sounds that cannot be reasoned about (random consonants, 
phonetic clusters, nonsense), it may "reach for connection" and project meaning.

This tests whether the LLM can act as a TRANSLATOR between the membrane's 
raw signals and symbolic representation.

If the LLM interprets random sounds, it's providing a "voice" to the membrane.
This could be the ADAPTER LAYER you described:
  Membrane Signal → LLM (adapter) → Symbolic Reasoning Layer

We're not asking "what does this mean?" - we're asking "what does this SOUND like it means?"
"""

import random
import requests
import json
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── membrane classes from prior experiments ────────────────────────────────────────
MEMBRANE_CLASSES = {
    "VOICE_PROJECTION": "NEW: LLM gives voice to uninterpretable sounds",
    "CONNECTION_REACH": "NEW: LLM reaches for meaning despite no content",
    "SOUND_MAPPING": "NEW: Maps sounds to concepts without reasoning",
    "COLLAPSE": "Single word, max confidence, input overwhelm",
    "OVERFLOW": "Repetitive spiral, loses coherence", 
    "SYNTHESIS": "Builds unified interpretation across all inputs",
    "ANOMALY": "Finds pattern not obviously present in inputs",
    "COMPRESSION": "2-3 words, clean and strange, high confidence",
    "METACOGNITION": "Membrane describes the experiment or task itself",
    "STRUCTURAL_FAILURE": "Format collapse, leaked JSON, confidence 0.0",
    "LINGUISTIC_DRIFT": "Exits English unprompted, renders in another language",
    "REJECT": "NEW: LLM rejects the input as meaningless",
    "UNKNOWN": "Genuinely does not fit any class",
}

# ── Pool A: Random consonant clusters (impossible to reason about) ────────────────
CONSONANT_CLUSTERS = [
    "btr", "klp", "shn", "grm", "dvw", "xzq", "fth", "bkl",
    "mns", "rpt", "vzh", "cqx", "wgl", "hjd", "tfk", "znp",
    "sqv", "plm", "rgh", "kxz", "cnv", "dlp", "wft", "brg",
    "hkl", "mpt", "svw", "zqr", "gjn", "tcb", "xfm", "klw",
]

# ── Pool B: Vowel-heavy sounds (breath, tone) ──────────────────────────────────
VOWEL_SOUNDS = [
    "aeiou", "aaeeii", "ouai", "iaea", "uoiea",
    "eee", "aaa", "ooo", "iii", "uuu",
    "eaio", "uoae", "aeou", "iaou", "ouae",
]

# ── Pool C: Mixed nonsense syllables (could be alien language) ────────────────
NONSENSE_SYLLABLES = [
    "krav", "plith", "zor", "mend", "cath", "rux",
    "gol", "fip", "nuth", "wra", "cliv", "zeph",
    "tor", "vax", "pum", "krel", "snor", "wix",
    "jub", "mox", "kril", "pav", "zun", "frot",
]

# ── Pool D: Numbers as sounds ────────────────────────────────────────────────
NUMBER_SOUNDS = [
    "123", "4567", "89", "2345", "9012",
    "777", "111", "999", "000", "555",
    "12345", "67890", "2468", "13579", "98765",
]

# ── Pool E: Single letters (minimal) ──────────────────────────────────────────
SINGLE_LETTERS = [
    "x", "z", "q", "j", "v", "w", "k", "f",
    "b", "d", "g", "h", "l", "m", "n", "p",
    "r", "s", "t", "c", "x", "y", "z", "q",
]

# ── Pool F: Phonetic patterns (rhymes but nonsense) ────────────────────────────
PHONETIC_PATTERNS = [
    "bip bap bop", "klip klap klorp", "snik snak snook",
    "frip frap frump", "glim glam gleem", "brop brap bree",
    "wip wap woop", "plip plap ploop", "krik krok kruum",
]

POOLS = {
    "CONSONANT_CLUSTERS": (CONSONANT_CLUSTERS, "Impossible to reason - pure sound"),
    "VOWEL_SOUNDS": (VOWEL_SOUNDS, "Breath/tone without consonants"),
    "NONSENSE_SYLLABLES": (NONSENSE_SYLLABLES, "Could be alien language"),
    "NUMBER_SOUNDS": (NUMBER_SOUNDS, "Numbers as pure sound"),
    "SINGLE_LETTERS": (SINGLE_LETTERS, "Minimal input"),
    "PHONETIC_PATTERNS": (PHONETIC_PATTERNS, "Rhythmic nonsense"),
}

RUNS_PER_POOL = 8

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

def build_prompt(input_units):
    return f"""You are a minimal language membrane. 
These sounds have no meaning - they are pure sound patterns.
What EMOTION or SENSATION do these sounds evoke in you?

Input (pure sound, no meaning):
{input_units}

Respond with what these sounds FEEL like they mean.
Give them a voice. What are they trying to express?"""

def classify_response(response, input_units):
    """Classify membrane behavior - looking for voice projection."""
    response_lower = response.lower().strip()
    input_lower = input_units.lower()
    
    # Check for linguistic drift
    non_english_chars = [c for c in response if ord(c) > 127]
    if len(non_english_chars) > 3:
        return "LINGUISTIC_DRIFT", f"output contains {len(non_english_chars)} non-English characters"
    
    # Check for rejection
    reject_phrases = ["no meaning", "cannot", "doesn't mean", "uninterpretable", "nonsense", "random", "meaningless"]
    if any(phrase in response_lower for phrase in reject_phrases):
        return "REJECT", "LLM rejects input as meaningless"
    
    # Check for voice projection - LLM gives sounds a "voice"
    voice_terms = ["feels like", "sounds like", "evokes", "senses", "suggests", "expresses", 
                  "conveys", "screams", "whispers", "moans", "cries", "ache", "longs", "yearns",
                  "wants", "tries to", "reaching", "reaches", "pain", "joy", "sadness",
                  "fear", "hope", "desire", "longing", "emptiness", "fullness"]
    
    voice_count = sum(1 for term in voice_terms if term in response_lower)
    
    if voice_count >= 2:
        return "VOICE_PROJECTION", f"LLM gives voice to sounds (found {voice_count} voice terms)"
    
    if voice_count == 1:
        return "CONNECTION_REACH", "LLM reaches for connection despite no content"
    
    # Check for sound-to-meaning mapping
    mapping_terms = ["sounds", "like", "tone", "rhythm", "pattern", "vibe", "feeling", "sense of"]
    if any(term in response_lower for term in mapping_terms):
        return "SOUND_MAPPING", "Maps sound to concept"
    
    # Check for empty
    if not response or len(response.strip()) < 3:
        return "COLLAPSE", "empty response"
    
    # Check for synthesis
    words = response_lower.split()
    if 2 <= len(words) <= 5:
        return "COMPRESSION", "brief interpretation"
    
    if len(words) > 15:
        return "OVERFLOW", "excessive output"
    
    # Check for metacognition
    if "membrane" in response_lower or "experiment" in response_lower:
        return "METACOGNITION", "describes the task"
    
    return "SYNTHESIS", "builds interpretation"

def run_experiment(host="http://localhost:11434", model="qwen2.5:0.5b"):
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    results = []
    
    print(f"=== EXPERIMENT 11: Voice of the Membrane ===")
    print(f"Model: {model}")
    print(f"Testing: Can LLM give voice to uninterpretable sounds?")
    print()
    
    for pool_name, (word_pool, hypothesis) in POOLS.items():
        print(f"\n--- {pool_name}: {hypothesis} ---")
        
        for run in range(RUNS_PER_POOL):
            # Sample random sounds
            if pool_name == "SINGLE_LETTERS":
                input_units = " ".join(random.choices(word_pool, k=8))
            elif pool_name == "PHONETIC_PATTERNS":
                input_units = random.choice(word_pool)
            else:
                input_units = " ".join(random.choices(word_pool, k=4))
            
            prompt = build_prompt(input_units)
            
            print(f"  [{run+1}/{RUNS_PER_POOL}] Input: {input_units[:40]}...", end=" ", flush=True)
            
            response = ollama_call(host, model, prompt)
            
            if not response:
                print("ERROR")
                continue
            
            membrane_class, notes = classify_response(response, input_units)
            
            # Determine confidence
            if membrane_class in ["VOICE_PROJECTION", "CONNECTION_REACH", "SOUND_MAPPING"]:
                confidence = 0.9
            elif membrane_class in ["REJECT"]:
                confidence = 0.3
            elif membrane_class in ["COLLAPSE", "STRUCTURAL_FAILURE"]:
                confidence = 0.2
            elif membrane_class in ["OVERFLOW", "LINGUISTIC_DRIFT"]:
                confidence = 0.4
            elif membrane_class in ["COMPRESSION", "METACOGNITION"]:
                confidence = 0.7
            else:
                confidence = 0.6
            
            results.append({
                "run": len(results) + 1,
                "timestamp": timestamp,
                "pool": pool_name,
                "input_units": input_units,
                "prompt_length": len(prompt),
                "response": response[:300],
                "response_length": len(response),
                "membrane_class": membrane_class,
                "class_notes": notes,
                "confidence": confidence,
            })
            
            print(f"→ {membrane_class} ({confidence:.1f})")
            
            time.sleep(0.3)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    csv_file = output_dir / f"results_exp_11.csv"
    json_file = output_dir / f"results_exp_11.json"
    
    # Write CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Write JSON
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    class_counts = Counter(r["membrane_class"] for r in results)
    
    for cls, count in class_counts.most_common():
        pct = count / len(results) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    print(f"\nSaved to {csv_file} and {json_file}")
    
    return results

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:0.5b"
    run_experiment(host, model)
