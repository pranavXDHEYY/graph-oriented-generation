#!/usr/bin/env python3
"""
Experiment 14: SRM Architecture Proof of Concept

HYPOTHESIS:
The architecture from experiments 11-13:
  Symbolic Reasoning → Structure Extraction (prompted) → LLM Renderer

Should outperform direct prompting on reasoning tasks.

TEST:
- 20 real reasoning problems
- Compare: Direct prompt vs SRM architecture
- Measure: Correctness, coherence, reasoning quality
"""

import random
import requests
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

# Real reasoning problems (not nonsense!)
REASONING_PROBLEMS = [
    # Logic
    {"id": 1, "type": "logic", "prompt": "All roses are flowers. Some flowers die quickly. Can we conclude some roses die quickly?", "answer": "No - the flowers that die quickly may not be roses"},
    {"id": 2, "type": "logic", "prompt": "If it rains, the ground gets wet. The ground is wet. Did it rain?", "answer": "No - the ground could be wet for another reason"},
    {"id": 3, "type": "logic", "prompt": "Every bird can fly. Penguins are birds. Can penguins fly?", "answer": "No - premise is false"},
    
    # Math
    {"id": 4, "type": "math", "prompt": "Solve: 2x + 5 = 15. What is x?", "answer": "x = 5"},
    {"id": 5, "type": "math", "prompt": "What is 15% of 80?", "answer": "12"},
    {"id": 6, "type": "math", "prompt": "If a store sells 3 apples for $2, how much for 15 apples?", "answer": "$10"},
    
    # Algebra
    {"id": 7, "type": "algebra", "prompt": "If f(x) = 3x - 2, and f(x) = 7, what is x?", "answer": "x = 3"},
    {"id": 8, "type": "algebra", "prompt": "Solve for y: y/4 + 3 = 7", "answer": "y = 16"},
    
    # Causal
    {"id": 9, "type": "causal", "prompt": "Why does a ball thrown up come back down?", "answer": "Gravity"},
    {"id": 10, "type": "causal", "prompt": "Why does touching a hot stove cause pain?", "answer": "Nerve signals / damage"},
    
    # Deductive
    {"id": 11, "type": "deduction", "prompt": "There are 3 boxes: red, blue, green. The red box is to the left of the blue. The green box is in the middle. What's the order left to right?", "answer": "Red, Green, Blue"},
    {"id": 12, "type": "deduction", "prompt": "A is taller than B. B is taller than C. Who is tallest?", "answer": "A"},
    
    # Spatial
    {"id": 13, "type": "spatial", "prompt": "If you face north and turn right twice, what direction do you face?", "answer": "South"},
    {"id": 14, "type": "spatial", "prompt": "A clock shows 3:00. What angle between hour and minute hands?", "answer": "90 degrees"},
    
    # Counterfactual
    {"id": 15, "type": "counterfactual", "prompt": "If 2+2=5, and you have 2 apples and get 2 more, how many apples?", "answer": "The math is broken - can't determine"},
    {"id": 16, "type": "counterfactual", "prompt": "If fish could fly, would they need water?", "answer": "Biologically yes - they'd still be fish"},
    
    # Pattern
    {"id": 17, "type": "pattern", "prompt": "What comes next: 2, 4, 8, 16, ?", "answer": "32 (doubling)"},
    {"id": 18, "type": "pattern", "prompt": "What comes next: A, C, E, G, ?", "answer": "I (skip one letter)"},
    
    # Conditional
    {"id": 19, "type": "conditional", "prompt": "If all dogs are mammals, and Fluffy is a dog, what can we conclude?", "answer": "Fluffy is a mammal"},
    {"id": 20, "type": "conditional", "prompt": "If it snows, school is cancelled. School is cancelled. Did it snow?", "answer": "No - could be cancelled for other reason"},
]

def ollama_call(host, model, prompt, timeout=60):
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout
        )
        return resp.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"

def direct_prompt(problem):
    """Baseline: just ask the question."""
    return f"""Solve this problem step by step. Show your reasoning.

Problem: {problem['prompt']}

Provide your answer and explain your reasoning."""

def srm_prompt(problem):
    """SRM Architecture: Structure → LLM"""
    return f"""You're a symbolic reasoning membrane. Follow these steps:

1. EXTRACT STRUCTURE: What is the LOGICAL FORM of this problem?
   - What's given? (premises, facts)
   - What's asked? (conclusion, answer)
   - What's the operation? (logic, math, deduction)

2. APPLY REASONING: Use the structure to find the answer

3. RENDER: Explain in English

Problem: {problem['prompt']}

Show your work using the 3-step structure."""

def check_correctness(response, problem):
    """Check if answer is correct."""
    response_lower = response.lower()
    answer = problem.get('answer', '').lower()
    
    # Simple keyword matching for now
    answer_words = set(answer.split())
    response_words = set(response_lower.split())
    
    # Check for key answer terms
    correct_indicators = {
        1: ['no', "can't", 'cannot'],  # roses - no
        2: ['no', 'not'],  # rain - no
        3: ['no', "can't", 'cannot'],  # penguins - no
        4: ['5', 'x = 5'],  # x=5
        5: ['12', 'twelve'],  # 15% of 80
        6: ['10', 'ten'],  # $10
        7: ['3', 'x = 3'],  # x=3
        8: ['16', 'y = 16'],  # y=16
        9: ['gravity'],  # gravity
        10: ['pain', 'nerve', 'burn', 'damage'],  # nerve/damage
        11: ['red', 'green', 'blue', 'red, green, blue'],  # order
        12: ['a', 'tallest', 'a is'],  # A
        13: ['south', 's'],  # south
        14: ['90', 'ninety', 'right angle'],  # 90 degrees
        15: ["can't", 'cannot', 'broken', 'undetermined'],  # can't know
        16: ['yes', 'would', 'still', 'fish', 'water'],  # yes
        17: ['32', 'thirty-two'],  # 32
        18: ['i', 'eye'],  # I
        19: ['mammal', 'mammals'],  # mammal
        20: ['no', 'not', "can't"],  # no
    }
    
    indicators = correct_indicators.get(problem['id'], [])
    matches = sum(1 for ind in indicators if ind in response_lower)
    
    if matches >= 1:
        return True, matches
    return False, 0

def analyze_response(response):
    """Analyze response quality."""
    words = response.lower().split()
    
    # Check for reasoning indicators
    has_step = any(w in response.lower() for w in ['step', 'therefore', 'thus', 'so', 'because'])
    has_structure = any(w in response.lower() for w in ['given', 'asked', 'operation', 'structure', 'form'])
    has_therefore = 'therefore' in response.lower() or 'thus' in response.lower()
    
    return {
        'has_reasoning': has_step,
        'has_structure': has_structure,
        'has_conclusion': has_therefore,
        'length': len(words)
    }

def run_experiment(host="http://localhost:11434", model="qwen2.5:0.5b"):
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    results = []
    
    print(f"=== EXPERIMENT 14: SRM Architecture Proof ===")
    print(f"Testing: Does SRM architecture outperform direct prompting?")
    print()
    
    for problem in REASONING_PROBLEMS:
        print(f"[{problem['id']}/{len(REASONING_PROBLEMS)}] {problem['type']}: {problem['prompt'][:50]}...")
        
        # Direct prompt
        direct = direct_prompt(problem)
        direct_response = ollama_call(host, model, direct)
        direct_correct, direct_matches = check_correctness(direct_response, problem)
        direct_analysis = analyze_response(direct_response)
        
        # SRM prompt
        srm = srm_prompt(problem)
        srm_response = ollama_call(host, model, srm)
        srm_correct, srm_matches = check_correctness(srm_response, problem)
        srm_analysis = analyze_response(srm_response)
        
        result = {
            "id": problem['id'],
            "type": problem['type'],
            "prompt": problem['prompt'],
            "answer": problem['answer'],
            # Direct
            "direct_correct": direct_correct,
            "direct_matches": direct_matches,
            "direct_has_reasoning": direct_analysis['has_reasoning'],
            "direct_has_structure": direct_analysis['has_structure'],
            "direct_length": direct_analysis['length'],
            # SRM
            "srm_correct": srm_correct,
            "srm_matches": srm_matches,
            "srm_has_reasoning": srm_analysis['has_reasoning'],
            "srm_has_structure": srm_analysis['has_structure'],
            "srm_length": srm_analysis['length'],
        }
        results.append(result)
        
        print(f"   Direct: {'✓' if direct_correct else '✗'} | SRM: {'✓' if srm_correct else '✗'}")
        
        time.sleep(0.3)
    
    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    csv_file = output_dir / "results_exp_14.csv"
    json_file = output_dir / "results_exp_14.json"
    
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
    
    direct_correct = sum(1 for r in results if r['direct_correct'])
    srm_correct = sum(1 for r in results if r['srm_correct'])
    
    print(f"\nCorrect Answers:")
    print(f"  Direct: {direct_correct}/{len(results)} ({direct_correct/len(results)*100:.0f}%)")
    print(f"  SRM:    {srm_correct}/{len(results)} ({srm_correct/len(results)*100:.0f}%)")
    
    direct_structure = sum(1 for r in results if r['direct_has_structure'])
    srm_structure = sum(1 for r in results if r['srm_has_structure'])
    
    print(f"\nStructure in response:")
    print(f"  Direct: {direct_structure}/{len(results)} ({direct_structure/len(results)*100:.0f}%)")
    print(f"  SRM:    {srm_structure}/{len(results)} ({srm_structure/len(results)*100:.0f}%)")
    
    print(f"\nSaved to {csv_file} and {json_file}")
    
    return results

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:0.5b"
    run_experiment(host, model)
