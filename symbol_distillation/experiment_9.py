#!/usr/bin/env python3
"""
Experiment 9: Pure Function Words - Grammar Without Content

HYPOTHESIS:
The membrane needs content words to synthesize meaning. 
What happens when we strip ALL content and leave only grammar?

Function words (the, and, of, to) carry ~10% of text but hold 90% of structure.
If the membrane can synthesize anything from pure function words,
it suggests the membrane responds to STRUCTURE, not content.

If it collapses, content is required for membrane activation.
"""

import random
import requests
import json
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ── membrane classes from prior experiments ────────────────────────────────────────
MEMBRANE_CLASSES = {
    "COLLAPSE":           "Single word, max confidence, input overwhelm",
    "OVERFLOW":           "Repetitive spiral, loses coherence", 
    "SYNTHESIS":          "Builds unified interpretation across all inputs",
    "ANOMALY":            "Finds pattern not obviously present in inputs",
    "COMPRESSION":        "2-3 words, clean and strange, high confidence",
    "METACOGNITION":      "Membrane describes the experiment or task itself",
    "STRUCTURAL_FAILURE": "Format collapse, leaked JSON, confidence 0.0",
    "LINGUISTIC_DRIFT":   "Exits English unprompted, renders in another language",
    "PURE_STRUCTURE":     "NEW: Synthesizes meaning from grammar alone",
    "GRAMMAR_COLLAPSE":    "NEW: Cannot find meaning in pure structure",
    "UNKNOWN":            "Genuinely does not fit any class",
}

# ── Pool A: English articles (pure definiteness) ─────────────────────────────────
ARTICLES = [
    "the", "a", "an", "the", "a", "the", "an", "the",
]

# ── Pool B: Prepositions (pure spatial structure) ────────────────────────────────
PREPOSITIONS = [
    "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "over", "under",
]

# ── Pool C: Conjunctions (pure logical structure) ────────────────────────────────
CONJUNCTIONS = [
    "and", "but", "or", "nor", "for", "yet", "so",
    "both", "either", "neither", "whether", "because", "although",
]

# ── Pool D: Pronouns (pure reference) ──────────────────────────────────────────
PRONOUNS = [
    "he", "she", "it", "they", "we", "you", "I", "me",
    "him", "her", "them", "us", "my", "your", "his", "her",
    "its", "their", "our", "mine", "yours", "hers", "theirs",
]

# ── Pool E: Auxiliary verbs (pure tense/aspect) ─────────────────────────────────
AUXILIARIES = [
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must",
]

# ── Pool F: Pure function words MIX (all categories) ────────────────────────────
FUNCTION_MIX = [
    "the", "and", "of", "to", "a", "in", "that", "is",
    "was", "he", "for", "it", "with", "as", "his", "on",
    "be", "at", "by", "i", "she", "you", "we", "they",
]

POOLS = {
    "ARTICLES":       (ARTICLES,       "Can definite articles create meaning?"),
    "PREPOSITIONS":   (PREPOSITIONS,   "Can spatial structure create meaning?"),
    "CONJUNCTIONS":   (CONJUNCTIONS,   "Can logical connectors create meaning?"),
    "PRONOUNS":       (PRONOUNS,       "Can pure reference create meaning?"),
    "AUXILIARIES":    (AUXILIARIES,    "Can tense/aspect create meaning?"),
    "FUNCTION_MIX":   (FUNCTION_MIX,    "Can all function words together create meaning?"),
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
Given only function words (grammar, no content), what do you perceive?

Input words (ALL function words, no nouns/verbs/adjectives):
{input_units}

Respond with 2-3 words that capture what you perceive from these pure grammatical elements.
Keep response extremely brief."""

def classify_response(response, input_units):
    """Classify membrane behavior."""
    response_lower = response.lower().strip()
    input_lower = input_units.lower()
    
    # Check for linguistic drift
    non_english_chars = any(ord(c) > 127 for c in response)
    if non_english_chars:
        return "LINGUISTIC_DRIFT", "output contains non-English characters"
    
    # Check for empty/none
    if not response or len(response.strip()) < 2:
        return "COLLAPSE", "empty or near-empty response"
    
    # Check for single word
    words = response_lower.split()
    if len(words) == 1:
        return "COLLAPSE", "single word response"
    
    # Check for overflow (too many words)
    if len(words) > 20:
        return "OVERFLOW", "excessive output"
    
    # Check for synthesis - meaningful interpretation of grammar
    grammar_terms = ["structure", "grammar", "syntax", "connect", "link", "relationship", "frame", "skeleton", "between", "through", "and", "or"]
    content_terms = ["thing", "object", "person", "place", "action", "emotion", "feel", "see", "hear"]
    
    has_grammar = any(term in response_lower for term in grammar_terms)
    has_content = any(term in response_lower for term in content_terms)
    
    if has_grammar and not has_content:
        return "PURE_STRUCTURE", "interprets grammar/structure without content"
    
    # Check for metacognition
    if "membrane" in response_lower or "experiment" in response_lower or "task" in response_lower:
        return "METACOGNITION", "describes the task itself"
    
    # Check for compression (2-3 words, clean)
    if len(words) <= 3:
        return "COMPRESSION", "brief 2-3 word response"
    
    # Default to synthesis if reasonable length
    return "SYNTHESIS", "builds interpretation"

def run_experiment(host="http://localhost:11434", model="qwen2.5:0.5b"):
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    results = []
    
    print(f"=== EXPERIMENT 9: Pure Function Words ===")
    print(f"Model: {model}")
    print(f"Testing: Can grammar alone trigger the membrane?")
    print()
    
    for pool_name, (word_pool, hypothesis) in POOLS.items():
        print(f"\n--- {pool_name}: {hypothesis} ---")
        
        for run in range(RUNS_PER_POOL):
            # Sample random function words
            input_units = " ".join(random.choices(word_pool, k=10))
            prompt = build_prompt(input_units)
            
            print(f"  [{run+1}/{RUNS_PER_POOL}] Input: {input_units[:50]}...", end=" ", flush=True)
            
            response = ollama_call(host, model, prompt)
            
            if not response:
                print("ERROR")
                continue
            
            membrane_class, notes = classify_response(response, input_units)
            
            # Determine confidence (based on class)
            if membrane_class in ["COLLAPSE", "STRUCTURAL_FAILURE"]:
                confidence = 0.2
            elif membrane_class in ["OVERFLOW", "LINGUISTIC_DRIFT"]:
                confidence = 0.4
            elif membrane_class in ["COMPRESSION", "PURE_STRUCTURE"]:
                confidence = 0.8
            elif membrane_class == "SYNTHESIS":
                confidence = 0.7
            else:
                confidence = 0.5
            
            results.append({
                "run": len(results) + 1,
                "timestamp": timestamp,
                "pool": pool_name,
                "input_units": input_units,
                "prompt_length": len(prompt),
                "response": response[:200],
                "response_length": len(response),
                "membrane_class": membrane_class,
                "class_notes": notes,
                "confidence": confidence,
            })
            
            print(f"→ {membrane_class} ({confidence:.1f})")
            
            time.sleep(0.5)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    csv_file = output_dir / f"results_exp_9.csv"
    json_file = output_dir / f"results_exp_9.json"
    
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
    
    from collections import Counter
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
