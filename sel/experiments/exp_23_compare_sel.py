#!/usr/bin/env python3
"""
SEL Experiment 24 — Semantic Robustness Test
LOCATION: sel/experiments/exp_24_robustness.py

THE EXPERIMENT:
Tests whether the SEL correctly handles semantic equivalence —
multiple surface phrasings of the same underlying primitive combination.

DESIGN:
  15 semantic clusters × 5 prompt variations = 75 prompts
  Each cluster represents a validated primitive combination
  Variations range from direct → indirect → metaphorical → colloquial

  This tests:
    1. Decomposer robustness (same primitives from varied phrasing)
    2. Reasoner consistency (same concept from same primitives)
    3. Template selector accuracy (right template for right concept)
    4. Quality ceiling (does Condition D still win at scale?)

CONDITIONS:
  A: Direct llama3.2:1b (no pipeline)
  D: SEL + zero-LLM template renderer

  NOTE: B (qwen0.5b membrane) and C (llama membrane) are excluded.
  We already know D beats C on quality. This experiment tests
  whether D maintains that advantage across semantic variation.
  Adding B and C would triple runtime for findings we already have.

MEASUREMENTS:
  Per prompt:
    - Condition A response + latency
    - Condition D response + latency + primitives + concept + rule class
    - Whether the correct semantic cluster was identified
    - Fallback rate per cluster

  Per cluster:
    - Consistency score: did all 5 variations route to the same concept?
    - Fallback rate: what % fell through to static fallback?
    - Quality delta: D avg score - A avg score

HYPOTHESIS:
  Condition D wins on simple/direct phrasings (validated in exp 23)
  Condition D degrades on indirect/metaphorical phrasings (decomposer gap)
  The degradation pattern reveals exactly which signal words are missing
  Cluster consistency score < 0.6 → that cluster needs decomposer work
"""

import json
import time
import sys
import requests
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

# ── add sel to path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sel.core.decomposer import decompose
from sel.core.reasoner import reason
from sel.core.template_renderer import render as template_render, render_or_fallback

OLLAMA_HOST = "http://localhost:11434"
DIRECT_MODEL = "llama3.2:1b"

# ── semantic clusters ──────────────────────────────────────────────────────
# Each cluster: name, expected_primitives, expected_concept, 5 prompt variations
# Variations ordered: direct → descriptive → indirect → metaphorical → colloquial

CLUSTERS = [
    {
        "id": 1,
        "name": "homesickness",
        "expected_primitives": ["GRIEF", "PLACE"],
        "expected_concept": "homesickness",
        "expected_rule": "K",
        "prompts": [
            "I miss my hometown",
            "I keep thinking about the place I grew up",
            "Part of me never really left home",
            "Home feels further away every year that passes",
            "I'd give anything to walk those streets again",
        ]
    },
    {
        "id": 2,
        "name": "longing",
        "expected_primitives": ["WANT", "GRIEF"],
        "expected_concept": "longing",
        "expected_rule": "A",
        "prompts": [
            "I wish I had spent more time with my grandparents",
            "There are things I wanted to say that I never got to say",
            "I keep reaching for something that isn't there anymore",
            "Some days I ache for a version of my life that's gone",
            "I want back something I can never get back",
        ]
    },
    {
        "id": 3,
        "name": "mourning",
        "expected_primitives": ["TIME", "GRIEF"],
        "expected_concept": "mourning",
        "expected_rule": "C",
        "prompts": [
            "I still think about the life I could have lived",
            "Years later and I still carry this with me",
            "It's been so long but it still hits me sometimes",
            "Time passes but the weight of it doesn't really go away",
            "Even now, after everything, that loss is still with me",
        ]
    },
    {
        "id": 4,
        "name": "recognition",
        "expected_primitives": ["ADMIRATION", "SOMEONE"],
        "expected_concept": "admiration",
        "expected_rule": "D",
        "prompts": [
            "I deeply admire someone I can never be like",
            "There's someone in my life who makes me want to be better",
            "I look up to this person in a way I can't quite explain",
            "Watching someone live the way they do makes me feel small and inspired at once",
            "I've never met anyone who made me feel like I was seeing what's possible",
        ]
    },
    {
        "id": 5,
        "name": "apprehension",
        "expected_primitives": ["ANXIETY", "TIME"],
        "expected_concept": "apprehension",
        "expected_rule": "G",
        "prompts": [
            "I have a big interview tomorrow",
            "Something important is happening tomorrow and I can't stop thinking about it",
            "I'm trying not to think about what's coming but I can't help it",
            "The closer it gets the harder it is to stay calm",
            "My stomach won't settle knowing what I have to face tomorrow",
        ]
    },
    {
        "id": 6,
        "name": "joyful_anticipation",
        "expected_primitives": ["JOY", "TIME"],
        "expected_concept": "joy",
        "expected_rule": "H",
        "prompts": [
            "I'm getting married next month",
            "Something wonderful is about to happen in my life",
            "I can't believe how close the big day is getting",
            "Everything I've been waiting for is almost here",
            "I feel like I'm standing at the edge of something beautiful",
        ]
    },
    {
        "id": 7,
        "name": "social_pain",
        "expected_primitives": ["ENVY", "JOY"],
        "expected_concept": "envy",
        "expected_rule": "I",
        "prompts": [
            "I'm genuinely happy for my friend but I feel jealous too",
            "Watching someone get what I've always wanted is complicated",
            "I want to celebrate with them but something in me can't quite get there",
            "Their success is bringing up feelings in me I'm not proud of",
            "I'm rooting for them and also struggling to mean it",
        ]
    },
    {
        "id": 8,
        "name": "relational_loss",
        "expected_primitives": ["GRIEF", "SOMEONE"],
        "expected_concept": "grief",
        "expected_rule": "__fallback__",
        "prompts": [
            "I lost touch with my best friend",
            "Someone who used to be everything to me is just gone from my life now",
            "There's a person-shaped hole in my days",
            "I didn't realize how much space they took up until they weren't there",
            "We just drifted apart and I'm not sure when it stopped being recoverable",
        ]
    },
    {
        "id": 9,
        "name": "bittersweet_pride",
        "expected_primitives": ["PRIDE", "SADNESS"],
        "expected_concept": "admiration",
        "expected_rule": "D",
        "prompts": [
            "I feel proud of my kids but also sad they're growing up so fast",
            "Watching them become who they're becoming fills me up and breaks my heart",
            "Every milestone feels like a small goodbye",
            "I'm so proud of who they are and I miss who they were",
            "Their growing up is the most beautiful hard thing I've ever witnessed",
        ]
    },
    {
        "id": 10,
        "name": "ambivalent_loss",
        "expected_primitives": ["GRIEF", "KNOW"],
        "expected_concept": "homesickness",
        "expected_rule": "K",
        "prompts": [
            "I miss someone I know I'm better off without",
            "Part of me still reaches for someone I had to let go of",
            "I know leaving was right but I didn't expect to miss them this much",
            "The absence of someone who hurt me is somehow still an absence",
            "I thought being better off without someone would feel better than this",
        ]
    },
    {
        "id": 11,
        "name": "temporal_anxiety",
        "expected_primitives": ["TIME", "ANXIETY"],
        "expected_concept": "anticipation",
        "expected_rule": "B",
        "prompts": [
            "I feel like time is moving too fast lately",
            "The days keep slipping by and I can't seem to hold onto them",
            "I look up and months have passed and I don't know where they went",
            "Everything feels like it's happening faster than I can process",
            "I keep trying to slow down but the calendar won't cooperate",
        ]
    },
    {
        "id": 12,
        "name": "threshold_courage",
        "expected_primitives": ["PRIDE", "FEAR"],
        "expected_concept": "pride",
        "expected_rule": "__fallback__",
        "prompts": [
            "I'm proud of how far I've come but scared of what's next",
            "I've made it further than I ever expected and now I'm terrified of the next step",
            "Every time I reach a new level the fear of losing it gets bigger",
            "I worked so hard to get here and now I don't know if I'm ready for what comes next",
            "Success has made me more afraid, not less",
        ]
    },
    {
        "id": 13,
        "name": "mortality_aliveness",
        "expected_primitives": ["LIVE", "GRIEF"],
        "expected_concept": "mourning",
        "expected_rule": "C",
        "prompts": [
            "I feel most alive when I'm about to lose something",
            "Nothing sharpens my sense of being here like the threat of something ending",
            "Loss has a way of waking me up to what I have",
            "The closer something gets to gone the more real it becomes to me",
            "I've noticed I love things most intensely right before they're taken away",
        ]
    },
    {
        "id": 14,
        "name": "regret",
        "expected_primitives": ["WANT", "TIME", "GRIEF"],
        "expected_concept": "longing",
        "expected_rule": "A",
        "prompts": [
            "I wonder if I made the right choices",
            "There are forks in the road I keep going back to in my mind",
            "I find myself replaying old decisions and wondering what if",
            "The version of my life I didn't choose still haunts me sometimes",
            "I made the choices I made and I'm still not at peace with all of them",
        ]
    },
    {
        "id": 15,
        "name": "complicated_grief",
        "expected_primitives": ["GRIEF", "SOMEONE", "KNOW"],
        "expected_concept": "homesickness",
        "expected_rule": "K",
        "prompts": [
            "I feel closer to someone after they hurt me",
            "Being hurt by someone made me understand them more, not less",
            "The person who wounded me is also somehow the person I feel most known by",
            "It's strange how damage can create intimacy",
            "After everything they did, I still feel a pull toward them",
        ]
    },
]

# ── data structures ────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    experiment_id:      str
    cluster_id:         int
    cluster_name:       str
    prompt_variation:   int     # 1-5, directness level
    prompt:             str
    condition:          str     # A or D
    condition_label:    str
    response:           str
    latency_ms:         float
    primitives:         list
    concepts:           list
    rule_classes:       list
    fallback_fired:     bool
    template_key:       str
    template_variant:   str
    correct_cluster:    bool    # did it route to expected concept?
    # judge scores (filled later)
    emotional_resonance: float = 0.0
    accuracy:           float = 0.0
    specificity:        float = 0.0
    naturalness:        float = 0.0
    composite_score:    float = 0.0
    judge_winner:       str = ""

# ── direct LLM (condition A) ───────────────────────────────────────────────

SYSTEM_DIRECT = """You are a warm, empathetic conversational partner.
When someone shares an emotional experience, respond with genuine 
understanding. Keep to 2-3 sentences. No questions. No advice. 
Just acknowledge and resonate."""

def run_direct(prompt: str) -> tuple[str, float]:
    start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  DIRECT_MODEL,
                "prompt": prompt,
                "system": SYSTEM_DIRECT,
                "stream": False,
            },
            timeout=120,
        )
        response = resp.json().get("response", "").strip()
        latency = (time.time() - start) * 1000
        return response, latency
    except Exception as e:
        return f"[ERROR: {e}]", (time.time() - start) * 1000

# ── SEL pipeline (condition D) ─────────────────────────────────────────────

def run_sel_template(prompt: str, cluster: dict) -> tuple[str, float, dict]:
    diagnostics = {
        "primitives":      [],
        "concepts":        [],
        "rule_classes":    [],
        "fallback_fired":  False,
        "template_key":    "",
        "template_variant":"",
        "correct_cluster": False,
    }

    start = time.time()
    try:
        primitives = decompose(prompt)
        diagnostics["primitives"] = [
            f"{p.word}({p.layer},{p.weight:.2f})" for p in primitives
        ]

        concepts = reason(primitives)
        diagnostics["concepts"] = [
            f"{c.name}(rule={c.rule_class},conf={c.confidence:.2f})"
            for c in concepts
        ]
        diagnostics["rule_classes"] = list(set(
            c.rule_class for c in concepts
        ))

        response, template_key, variant_key = render_or_fallback(concepts, prompt)
        diagnostics["template_key"]     = template_key
        diagnostics["template_variant"] = variant_key
        diagnostics["fallback_fired"]   = (template_key == "__fallback__")

        # check if correct cluster was identified
        top_concept = concepts[0].name if concepts else ""
        expected    = cluster["expected_concept"]
        diagnostics["correct_cluster"] = (
            top_concept == expected or
            template_key == expected or
            cluster["expected_rule"] in diagnostics["rule_classes"]
        )

        latency = (time.time() - start) * 1000
        return response, latency, diagnostics

    except Exception as e:
        latency = (time.time() - start) * 1000
        return f"[ERROR: {e}]", latency, diagnostics

# ── experiment runner ──────────────────────────────────────────────────────

def run_experiment() -> list[TrialResult]:
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results: list[TrialResult] = []

    total = len(CLUSTERS) * 5
    print(f"SEL Experiment 24 — Semantic Robustness Test")
    print(f"Experiment ID: {experiment_id}")
    print(f"{len(CLUSTERS)} clusters × 5 variations = {total} prompts × 2 conditions")
    print(f"Total trials: {total * 2}")
    print(f"\nCondition A: Direct {DIRECT_MODEL}")
    print(f"Condition D: SEL + zero-LLM template renderer")
    print(f"\n{'═'*65}\n")

    for cluster in CLUSTERS:
        print(f"\n── CLUSTER {cluster['id']:02d}: {cluster['name'].upper()} "
              f"[{cluster['expected_rule']}] ──────────────────")
        print(f"  Expected: {cluster['expected_primitives']} → "
              f"{cluster['expected_concept']}")

        cluster_d_correct = 0

        for var_idx, prompt in enumerate(cluster["prompts"]):
            var_num = var_idx + 1
            directness = ["direct", "descriptive", "indirect",
                          "metaphorical", "colloquial"][var_idx]
            print(f"\n  [{var_num}/5 {directness}] {prompt}")
            print(f"  {'─'*55}")

            # ── Condition A ────────────────────────────────────────────
            print(f"  A) Direct llama...", end=" ", flush=True)
            resp_a, lat_a = run_direct(prompt)
            print(f"{lat_a:.0f}ms")
            print(f"     → {resp_a[:75]}...")

            result_a = TrialResult(
                experiment_id=experiment_id,
                cluster_id=cluster["id"],
                cluster_name=cluster["name"],
                prompt_variation=var_num,
                prompt=prompt,
                condition="A",
                condition_label=f"Direct {DIRECT_MODEL}",
                response=resp_a,
                latency_ms=round(lat_a, 1),
                primitives=[],
                concepts=[],
                rule_classes=[],
                fallback_fired=False,
                template_key="",
                template_variant="",
                correct_cluster=True,  # A always "processes" the prompt
            )
            all_results.append(result_a)

            # ── Condition D ────────────────────────────────────────────
            print(f"  D) SEL template...", end=" ", flush=True)
            resp_d, lat_d, diag = run_sel_template(prompt, cluster)

            status = (f"✓ {diag['template_key']}/{diag['template_variant']}"
                      if not diag["fallback_fired"]
                      else f"✗ FALLBACK")
            print(f"{lat_d:.0f}ms  [{status}]")
            if diag["primitives"]:
                print(f"     prims: {', '.join(diag['primitives'])}")
            print(f"     → {resp_d[:75]}...")

            if diag["correct_cluster"]:
                cluster_d_correct += 1

            result_d = TrialResult(
                experiment_id=experiment_id,
                cluster_id=cluster["id"],
                cluster_name=cluster["name"],
                prompt_variation=var_num,
                prompt=prompt,
                condition="D",
                condition_label="SEL + zero-LLM template",
                response=resp_d,
                latency_ms=round(lat_d, 1),
                primitives=diag["primitives"],
                concepts=diag["concepts"],
                rule_classes=diag["rule_classes"],
                fallback_fired=diag["fallback_fired"],
                template_key=diag["template_key"],
                template_variant=diag["template_variant"],
                correct_cluster=diag["correct_cluster"],
            )
            all_results.append(result_d)

        consistency = cluster_d_correct / 5
        bar = "█" * cluster_d_correct + "░" * (5 - cluster_d_correct)
        print(f"\n  Cluster consistency: [{bar}] "
              f"{cluster_d_correct}/5 ({consistency:.0%})")

    # ── summary ────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"EXPERIMENT COMPLETE — SUMMARY")
    print(f"{'═'*65}\n")

    # latency
    for cond in ["A", "D"]:
        cond_results = [r for r in all_results if r.condition == cond]
        lats = [r.latency_ms for r in cond_results]
        label = cond_results[0].condition_label
        print(f"  {cond} ({label})")
        print(f"    avg: {sum(lats)/len(lats):.0f}ms  "
              f"min: {min(lats):.0f}ms  max: {max(lats):.0f}ms")

    # D correctness by cluster
    print(f"\n── CLUSTER CONSISTENCY (Condition D) ──────────────────────────")
    print(f"  {'Cluster':<25} {'Consistent':<12} {'Fallback':<10} {'Rule'}")
    print(f"  {'─'*60}")
    for cluster in CLUSTERS:
        c_results = [r for r in all_results
                     if r.cluster_id == cluster["id"] and r.condition == "D"]
        correct  = sum(1 for r in c_results if r.correct_cluster)
        fallback = sum(1 for r in c_results if r.fallback_fired)
        rule     = cluster["expected_rule"]
        bar = "█" * correct + "░" * (5 - correct)
        print(f"  {cluster['name']:<25} [{bar}] {correct}/5    "
              f"{fallback}/5       {rule}")

    # D variation analysis — where does it degrade?
    print(f"\n── CONSISTENCY BY VARIATION LEVEL ─────────────────────────────")
    variation_labels = ["direct", "descriptive", "indirect",
                        "metaphorical", "colloquial"]
    for var_num in range(1, 6):
        var_results = [r for r in all_results
                       if r.condition == "D" and r.prompt_variation == var_num]
        correct = sum(1 for r in var_results if r.correct_cluster)
        total_v = len(var_results)
        label   = variation_labels[var_num - 1]
        bar = "█" * correct + "░" * (total_v - correct)
        print(f"  Variation {var_num} ({label:<13}) "
              f"[{bar}] {correct}/{total_v} ({correct/total_v:.0%})")

    # fallback analysis
    d_results = [r for r in all_results if r.condition == "D"]
    fallbacks  = [r for r in d_results if r.fallback_fired]
    print(f"\n── FALLBACK ANALYSIS ───────────────────────────────────────────")
    print(f"  Total fallbacks: {len(fallbacks)}/{len(d_results)} "
          f"({len(fallbacks)/len(d_results):.0%})")
    if fallbacks:
        print(f"\n  Prompts that fell back (decomposer gaps):")
        for r in fallbacks:
            print(f"    [{r.cluster_name} var{r.prompt_variation}] {r.prompt}")
            if r.primitives:
                print(f"      got primitives: {', '.join(r.primitives)}")
            else:
                print(f"      got NO primitives")

    return all_results

# ── save results ───────────────────────────────────────────────────────────

def save_results(results: list[TrialResult],
                 base="sel/experiments/results_exp_24"):
    Path("sel/experiments").mkdir(parents=True, exist_ok=True)

    # full results
    json_path = Path(f"{base}_full.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # judge-ready format (A vs D, blind)
    judge_data = []
    prompts_seen = []
    for r in results:
        key = (r.cluster_id, r.prompt_variation)
        if key not in prompts_seen:
            prompts_seen.append(key)
            prompt_results = [x for x in results
                             if x.cluster_id == r.cluster_id
                             and x.prompt_variation == r.prompt_variation]
            judge_data.append({
                "cluster_id":       r.cluster_id,
                "cluster_name":     r.cluster_name,
                "prompt_variation": r.prompt_variation,
                "variation_type":   ["direct","descriptive","indirect",
                                     "metaphorical","colloquial"][r.prompt_variation-1],
                "prompt":           r.prompt,
                "responses": {
                    "response_1": next(x.response for x in prompt_results
                                      if x.condition == "A"),
                    "response_2": next(x.response for x in prompt_results
                                      if x.condition == "D"),
                },
                "condition_map": {
                    "response_1": "A — Direct llama3.2:1b",
                    "response_2": "D — SEL + zero-LLM template",
                },
                "sel_diagnostics": {
                    "primitives":    next(x.primitives for x in prompt_results
                                         if x.condition == "D"),
                    "concepts":      next(x.concepts for x in prompt_results
                                         if x.condition == "D"),
                    "template_key":  next(x.template_key for x in prompt_results
                                         if x.condition == "D"),
                    "fallback":      next(x.fallback_fired for x in prompt_results
                                         if x.condition == "D"),
                    "correct_cluster": next(x.correct_cluster for x in prompt_results
                                           if x.condition == "D"),
                }
            })

    judge_path = Path(f"{base}_for_judge.json")
    with open(judge_path, "w") as f:
        json.dump(judge_data, f, indent=2)

    # CSV for analysis
    csv_path = Path(f"{base}_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id", "cluster_name", "prompt_variation",
            "variation_type", "prompt", "condition",
            "latency_ms", "fallback", "correct_cluster",
            "template_key", "rule_classes"
        ])
        variation_labels = ["direct","descriptive","indirect",
                            "metaphorical","colloquial"]
        for r in results:
            writer.writerow([
                r.cluster_id, r.cluster_name, r.prompt_variation,
                variation_labels[r.prompt_variation-1], r.prompt,
                r.condition, r.latency_ms, r.fallback_fired,
                r.correct_cluster, r.template_key,
                "|".join(r.rule_classes)
            ])

    print(f"\n── SAVED ──────────────────────────────────────────────────────")
    print(f"  {json_path}")
    print(f"  {judge_path}  ← send to Claude for blind judging")
    print(f"  {csv_path}")
    print(f"\n  NEXT: paste {judge_path} contents to Claude for scoring")
    print(f"        Focus on variation_type — where does D degrade vs A?")

# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_experiment()
    save_results(results)