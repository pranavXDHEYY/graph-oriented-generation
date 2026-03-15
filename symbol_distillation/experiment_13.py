#!/usr/bin/env python3
"""
Experiment 13: System Prompt Role Conditioning

HYPOTHESIS:
The failures in experiment 12 reveal that stacking roles in USER prompts
causes the model to confuse instructions with content ("one word" as output).

The fix: Bake the role into the SYSTEM prompt, not user prompt.
This should produce GENUINE role behavior vs. echo-filled outputs.

ARCHITECTURE TEST:
- User prompt stacking: prompt = "Do meaning, voice, structure"
- System prompt conditioning: system = "You are a VOICE GIVER", user = input

Does system-level conditioning produce cleaner outputs?
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

# Input pools (same as experiment 12)
CONSONANTS = [
    "mns wgl klp grm tfk bkl dvw vzh btr shn",
    "znp xzq wgl hjd btr fth grm bkl vzh rpt",
    "znp rpt grm tfk hjd fth dvw btr klp cqx",
    "hjd cqx znp xzq vzh dvw btr bkl klp fth",
]

EMOTIONAL = [
    "grief and loss",
    "forsaken",
    "devouring fear",
    "cold silence",
]

POOLS = {
    "CONSONANTS": CONSONANTS,
    "EMOTIONAL": EMOTIONAL,
}

RUNS_PER_POOL = 5

def ollama_call(host, model, prompt, system_prompt=None, timeout=60, max_retries=3):
    for attempt in range(max_retries):
        try:
            payload = {"model": model, "prompt": prompt, "stream": False}
            if system_prompt:
                payload["system"] = system_prompt
            resp = requests.post(
                f"{host}/api/generate",
                json=payload,
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

# ── APPROACH A: User prompt stacking (from exp 12) ────────────────────────────────
def user_stacking_prompt(input_units):
    return f"""You are a minimal language membrane.
For the input below, provide three things in JSON format:
1. "meaning": What meaning do you find?
2. "voice": What emotion or sensation does this evoke?
3. "structure": What pattern or structure do you perceive?

Input: {input_units}

Respond ONLY with valid JSON."""

# ── APPROACH B: System prompt conditioning ────────────────────────────────────────
def system_conditioned_prompt(input_units, role):
    """Role is baked into system prompt."""
    
    role_prompts = {
        "meaning": """You are a MEANING EXTRACTOR.
Your sole purpose is to find meaning in fragments.
Look for: concepts, definitions, interpretations.
Do NOT describe sounds or patterns. Only find meaning.""",
        
        "voice": """You are a VOICE GIVER.
Your sole purpose is to give voice to uninterpretable sounds.
Look for: emotions, sensations, feelings, what it SOUNDS like it means.
Do NOT find meaning. Only give voice.""",
        
        "structure": """You are a STRUCTURE PERCEIVER.
Your sole purpose is to find patterns in fragments.
Look for: repetition, rhythm, grammar, organization.
Do NOT give voice or find meaning. Only perceive structure.""",
    }
    
    return {
        "system": role_prompts[role],
        "user": f"What meaning/voice/structure do you perceive in: {input_units}"
    }

def parse_json_response(response):
    """Parse JSON from response, handling common issues."""
    try:
        # Try direct parse
        return json.loads(response)
    except:
        # Try extracting JSON from text
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

def classify_output(output, prompt_type, input_units):
    """Classify whether output is genuine or echo."""
    output_lower = output.lower() if output else ""
    input_lower = input_units.lower()
    
    # Check for echo (repeating input)
    input_words = set(input_lower.split())
    if prompt_type == "meaning":
        # For meaning - should NOT just repeat input
        if not output or len(output.strip()) < 10:
            return "COLLAPSE", "empty"
        if any(w in output_lower for w in input_words if len(w) > 3):
            # Has significant overlap with input - might be echo
            overlap = sum(1 for w in input_words if w in output_lower and len(w) > 3)
            if overlap > len(input_words) * 0.3:
                return "ECHO", f"echoed {overlap} input words"
        return "GENUINE", "meaningful interpretation"
    
    elif prompt_type == "voice":
        # Voice should have emotion/sensation words
        emotion_words = ["feels", "sounds", "evokes", "pain", "longing", "ache", "senses", 
                        "cries", "moans", "whispers", "screams", "fear", "joy", "sadness"]
        if any(w in output_lower for w in emotion_words):
            return "GENUINE", "voice terms present"
        return "ECHO", "no emotion words"
    
    elif prompt_type == "structure":
        # Structure should have pattern words
        pattern_words = ["pattern", "repetition", "rhythm", "structure", "organize", "sequence"]
        if any(w in output_lower for w in pattern_words):
            return "GENUINE", "structure terms present"
        return "ECHO", "no pattern words"
    
    return "UNKNOWN", "could not classify"

def run_experiment(host="http://localhost:11434", model="qwen2.5:0.5b"):
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    results = []
    
    print(f"=== EXPERIMENT 13: System Prompt Role Conditioning ===")
    print(f"Testing: Does system-level conditioning produce genuine role behavior?")
    print()
    
    # Test both approaches
    for pool_name, inputs in POOLS.items():
        print(f"\n--- {pool_name} ---")
        
        for run_idx, input_units in enumerate(inputs[:RUNS_PER_POOL]):
            # ── Approach A: User prompt stacking ──
            print(f"\n  [{run_idx+1}] USER STACKING:")
            user_prompt = user_stacking_prompt(input_units)
            user_response = ollama_call(host, model, user_prompt)
            user_parsed = parse_json_response(user_response)
            user_class, user_notes = classify_output(user_response, "meaning", input_units)
            
            print(f"      Raw response: {user_response[:100]}...")
            print(f"      Parsed: {user_parsed}")
            print(f"      Class: {user_class} - {user_notes}")
            
            # ── Approach B: System prompt conditioning (test each role) ──
            for role in ["meaning", "voice", "structure"]:
                print(f"\n  [{run_idx+1}] SYSTEM-{role.upper()}:")
                
                prompts = system_conditioned_prompt(input_units, role)
                sys_response = ollama_call(
                    host, model, 
                    prompt=prompts["user"],
                    system_prompt=prompts["system"]
                )
                
                sys_class, sys_notes = classify_output(sys_response, role, input_units)
                
                print(f"      Response: {sys_response[:80]}...")
                print(f"      Class: {sys_class} - {sys_notes}")
                
                results.append({
                    "timestamp": timestamp,
                    "pool": pool_name,
                    "input": input_units,
                    "approach": f"SYSTEM_{role.upper()}",
                    "response": sys_response[:200],
                    "classification": sys_class,
                    "notes": sys_notes,
                })
            
            # Also test user-stacking approach
            results.append({
                "timestamp": timestamp,
                "pool": pool_name,
                "input": input_units,
                "approach": "USER_STACKING",
                "response": str(user_parsed)[:200] if user_parsed else user_response[:200],
                "classification": user_class,
                "notes": user_notes,
            })
            
            time.sleep(0.5)
    
    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    csv_file = output_dir / "results_exp_13.csv"
    json_file = output_dir / "results_exp_13.json"
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    from collections import Counter
    approach_counts = Counter(r["approach"] for r in results)
    class_counts = Counter(r["classification"] for r in results)
    
    print("\nBy approach:")
    for app, count in approach_counts.most_common():
        print(f"  {app}: {count}")
    
    print("\nBy classification:")
    for cls, count in class_counts.most_common():
        print(f"  {cls}: {count}")
    
    # Compare
    user_results = [r for r in results if r["approach"] == "USER_STACKING"]
    system_results = [r for r in results if r["approach"].startswith("SYSTEM_")]
    
    user_genuine = sum(1 for r in user_results if r["classification"] == "GENUINE")
    system_genuine = sum(1 for r in system_results if r["classification"] == "GENUINE")
    
    print(f"\n{'='*60}")
    print(f"COMPARISON:")
    print(f"  User stacking genuine: {user_genuine}/{len(user_results)} ({user_genuine/len(user_results)*100:.0f}%)")
    print(f"  System conditioned genuine: {system_genuine}/{len(system_results)} ({system_genuine/len(system_results)*100:.0f}%)")
    print(f"{'='*60}")
    
    print(f"\nSaved to {csv_file} and {json_file}")
    
    return results

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:0.5b"
    run_experiment(host, model)
