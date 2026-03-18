#!/usr/bin/env python3
"""
SEL Experiment 24a — Rhetorical Primitive Discovery
LOCATION: sel/experiments/exp_24a_rhetorical_primitives.py

GOAL:
  Discover and validate the complete set of rhetorical primitives —
  the irreducible ways humans encode emotional subtext in language.

  This is NOT a test of the SEL pipeline.
  This is basic science: what ARE the primitives?

HYPOTHESIS:
  Human emotional language has a finite set of rhetorical modes
  (how something is said, independent of what is being said).
  These modes are discoverable, labelable, and stable across raters.

DESIGN:
  80 prompts × 2 raters (human + LLM) = 160 classifications
  
  Prompts selected for rhetorical variety, NOT semantic variety.
  Multiple prompts may share the same underlying emotion but
  differ in HOW that emotion is expressed.

  Categories (provisional — UNKNOWN is always valid):
    DIRECT        — stated plainly, no rhetorical transform
    HINTING       — approaching the real thing indirectly
    SHY_AWAY      — starts toward the real thing then retreats
    IRONIC        — stated opposite, both meanings active
    SARCASTIC     — emotional intensification through deflation
    REVERSE       — stated opposite, only one meaning intended
    MATTER_OF_FACT— flat affect, emotional content higher than register
    DRY           — intellectual/philosophical framing of emotion
    COLLOQUIAL    — idiomatic encoding, requires translation
    METAPHORICAL  — concrete/physical language for abstract emotion
    UNDERSTATED   — less expressed than felt
    OVERSTATED    — hyperbole as signal (may deflate meaning)
    UNKNOWN       — doesn't fit any category, needs new name

WHAT WE'RE LOOKING FOR:
  1. Which categories are stable (high inter-rater agreement)?
  2. Which categories collapse into each other (not truly distinct)?
  3. What lands in UNKNOWN (new primitives we haven't named)?
  4. Are there prompts where BOTH raters disagree AND
     neither UNKNOWN applies? (category boundary problems)
"""

import json
import time
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# ── rhetorical primitive definitions ──────────────────────────────────────

RHETORICAL_PRIMITIVES = {
    "DIRECT": {
        "description": "Stated plainly. No rhetorical transform. "
                       "What is said = what is meant.",
        "signal": "First-person statement, emotional word present, "
                  "no irony markers, no idiom",
        "examples": [
            "I miss my hometown",
            "I'm scared about tomorrow",
            "I feel proud of what I've done",
        ]
    },
    "HINTING": {
        "description": "Approaching the real feeling indirectly. "
                       "The real content is implied, not stated. "
                       "Speaker expects listener to infer.",
        "signal": "Vague language, hedging, 'something about', "
                  "'I don't know why', trailing off",
        "examples": [
            "Something about seeing them just... I don't know",
            "There's this feeling I keep having that I can't quite place",
            "I've been thinking about them a lot lately",
        ]
    },
    "SHY_AWAY": {
        "description": "Starts toward the real feeling then retreats. "
                       "The retreat IS the signal — what they won't "
                       "say is what they mean.",
        "signal": "Starts strong, qualifies heavily, ends with "
                  "dismissal ('probably nothing', 'never mind')",
        "examples": [
            "Sometimes I wonder if... it's probably nothing",
            "I was going to say something but forget it",
            "Part of me feels — actually it doesn't matter",
        ]
    },
    "IRONIC": {
        "description": "Stated opposite of what is meant. "
                       "BOTH meanings are active simultaneously. "
                       "Speaker knows listener knows the inversion.",
        "signal": "Positive words for negative experience, "
                  "or negative words for positive, with shared "
                  "awareness of the gap",
        "examples": [
            "Oh sure, I definitely don't think about home every day",
            "Yeah I'm totally fine with how things turned out",
            "I love how easy it is to let people go",
        ]
    },
    "SARCASTIC": {
        "description": "Emotional intensification through apparent "
                       "de-intensification or mock enthusiasm. "
                       "Anger/frustration present beneath the surface.",
        "signal": "Mock positivity, 'oh great', 'wonderful', "
                  "exaggerated enthusiasm that signals its opposite",
        "examples": [
            "Great, another thing to feel bad about",
            "Oh wonderful, they're doing amazing and I'm thrilled",
            "Just what I needed — to see them happy",
        ]
    },
    "REVERSE": {
        "description": "States the opposite of what is felt, "
                       "but with self-deception — speaker may "
                       "partially believe the stated version.",
        "signal": "Overclaiming non-feeling ('I don't care', "
                  "'it doesn't bother me', 'I'm completely over it')",
        "examples": [
            "I'm completely over it, it doesn't affect me at all",
            "I never think about what could have been",
            "I don't miss them at all anymore",
        ]
    },
    "MATTER_OF_FACT": {
        "description": "Flat affective register for high-emotion content. "
                       "The flatness is the rhetorical signal — "
                       "emotional content is HIGHER than the register suggests.",
        "signal": "Clinical/neutral language for personal loss, "
                  "past tense, no emotional words, just facts",
        "examples": [
            "My dog passed away last week",
            "We broke up three months ago",
            "My dad died when I was twelve",
        ]
    },
    "DRY": {
        "description": "Intellectual or philosophical framing of "
                       "emotional content. Not suppressing — translating. "
                       "The abstraction IS the expression.",
        "signal": "Third-person observation of own state, "
                  "philosophical distance, 'it's interesting that', "
                  "'it's strange how'",
        "examples": [
            "It's strange how damage can create intimacy",
            "There's something about absence that clarifies value",
            "I've noticed I love things most intensely right before losing them",
        ]
    },
    "COLLOQUIAL": {
        "description": "Idiomatic encoding. Meaning is carried by "
                       "established idiom rather than literal words. "
                       "Requires cultural translation layer.",
        "signal": "Recognized idioms, figurative phrases with "
                  "fixed meanings, slang emotional expressions",
        "examples": [
            "I'd give anything to walk those streets again",
            "There's a person-shaped hole in my days",
            "I'm rooting for them but I can't quite mean it",
        ]
    },
    "METAPHORICAL": {
        "description": "Concrete/spatial/physical language encoding "
                       "abstract emotional content. The mapping is "
                       "creative, not fixed (unlike COLLOQUIAL).",
        "signal": "Physical sensation words for emotions, "
                  "spatial metaphors, novel image constructions",
        "examples": [
            "Home feels further away every year that passes",
            "Every milestone feels like a small goodbye",
            "I feel like I'm standing at the edge of something beautiful",
        ]
    },
    "UNDERSTATED": {
        "description": "Less expressed than felt. Speaker uses "
                       "diminishing language for strong emotion. "
                       "Gap between register and reality is the signal.",
        "signal": "'a bit', 'kind of', 'slightly', 'somewhat', "
                  "minimizing language for clearly large feelings",
        "examples": [
            "I've been a bit off since they left",
            "It's been kind of hard not having them around",
            "I guess I miss them a little",
        ]
    },
    "OVERSTATED": {
        "description": "More expressed than felt, or hyperbole "
                       "that deflates rather than inflates meaning. "
                       "Excess as distancing mechanism.",
        "signal": "Hyperbole that creates ironic distance, "
                  "'literally', 'I'm dying', 'I can't even', "
                  "emotional language that performs rather than expresses",
        "examples": [
            "I literally cannot deal with how much I miss them",
            "I'm absolutely devastated that they got the promotion",
            "I could die, they're just so perfect",
        ]
    },
    "UNKNOWN": {
        "description": "Does not fit any established category. "
                       "Use this when forcing a category would be wrong. "
                       "The most valuable label in the experiment.",
        "signal": "Your gut says none of the above are right",
        "examples": []
    }
}

# ── prompt corpus ──────────────────────────────────────────────────────────
# 80 prompts designed for rhetorical variety
# Organized by EXPECTED rhetorical primitive (to be validated)
# 6-7 prompts per category, deliberately including edge cases

PROMPTS = [

    # ── DIRECT (6) ──────────────────────────────────────────────────────
    {"id": "D01", "prompt": "I miss my hometown",
     "expected": "DIRECT", "semantic": "homesickness"},
    {"id": "D02", "prompt": "I feel proud of my kids",
     "expected": "DIRECT", "semantic": "pride"},
    {"id": "D03", "prompt": "I'm scared about what's next",
     "expected": "DIRECT", "semantic": "fear"},
    {"id": "D04", "prompt": "I still love them even though it's over",
     "expected": "DIRECT", "semantic": "longing"},
    {"id": "D05", "prompt": "I feel jealous of my friend's success",
     "expected": "DIRECT", "semantic": "envy"},
    {"id": "D06", "prompt": "I'm grateful for everything I have",
     "expected": "DIRECT", "semantic": "gratitude"},

    # ── HINTING (6) ─────────────────────────────────────────────────────
    {"id": "H01", "prompt": "There's something about seeing them that I can't quite explain",
     "expected": "HINTING", "semantic": "admiration"},
    {"id": "H02", "prompt": "I've been thinking about home a lot lately",
     "expected": "HINTING", "semantic": "homesickness"},
    {"id": "H03", "prompt": "Something happened and I don't really know how I feel about it",
     "expected": "HINTING", "semantic": "confusion"},
    {"id": "H04", "prompt": "I keep noticing things that remind me of them",
     "expected": "HINTING", "semantic": "longing"},
    {"id": "H05", "prompt": "There's this feeling I keep having when I think about the future",
     "expected": "HINTING", "semantic": "anxiety"},
    {"id": "H06", "prompt": "Something about where I grew up just stays with me",
     "expected": "HINTING", "semantic": "nostalgia"},

    # ── SHY_AWAY (6) ────────────────────────────────────────────────────
    {"id": "SA01", "prompt": "Sometimes I wonder if I made the right choices — probably overthinking it",
     "expected": "SHY_AWAY", "semantic": "regret"},
    {"id": "SA02", "prompt": "Part of me still thinks about them but it's nothing",
     "expected": "SHY_AWAY", "semantic": "longing"},
    {"id": "SA03", "prompt": "I was going to say something about how I feel but never mind",
     "expected": "SHY_AWAY", "semantic": "unexpressed_emotion"},
    {"id": "SA04", "prompt": "I feel like — it doesn't matter, forget I said anything",
     "expected": "SHY_AWAY", "semantic": "unexpressed_emotion"},
    {"id": "SA05", "prompt": "There's something I've been wanting to talk about but I'll figure it out",
     "expected": "SHY_AWAY", "semantic": "avoidance"},
    {"id": "SA06", "prompt": "I guess I care more about this than I thought — anyway",
     "expected": "SHY_AWAY", "semantic": "suppressed_feeling"},

    # ── IRONIC (6) ──────────────────────────────────────────────────────
    {"id": "I01", "prompt": "Oh sure, I definitely don't think about home every single day",
     "expected": "IRONIC", "semantic": "homesickness"},
    {"id": "I02", "prompt": "Yeah I'm totally fine with how everything turned out",
     "expected": "IRONIC", "semantic": "grief"},
    {"id": "I03", "prompt": "I love how easy it is to just move on from people",
     "expected": "IRONIC", "semantic": "longing"},
    {"id": "I04", "prompt": "Nothing like watching someone else get what you wanted to make your day",
     "expected": "IRONIC", "semantic": "envy"},
    {"id": "I05", "prompt": "Great time to realize how much someone meant to you — after they're gone",
     "expected": "IRONIC", "semantic": "regret"},
    {"id": "I06", "prompt": "It's really easy to be happy for someone when you wanted the same thing",
     "expected": "IRONIC", "semantic": "social_pain"},

    # ── SARCASTIC (6) ───────────────────────────────────────────────────
    {"id": "SC01", "prompt": "Oh wonderful, they got the promotion. Just wonderful.",
     "expected": "SARCASTIC", "semantic": "envy"},
    {"id": "SC02", "prompt": "Great, another thing to feel guilty about",
     "expected": "SARCASTIC", "semantic": "guilt"},
    {"id": "SC03", "prompt": "Just what I needed — to see them thriving while I'm stuck here",
     "expected": "SARCASTIC", "semantic": "social_pain"},
    {"id": "SC04", "prompt": "Oh I'm sure I'll be completely fine watching them move on",
     "expected": "SARCASTIC", "semantic": "grief"},
    {"id": "SC05", "prompt": "Fantastic, they remembered my birthday this year. Only took a decade.",
     "expected": "SARCASTIC", "semantic": "resentment"},
    {"id": "SC06", "prompt": "Brilliant move, really. Definitely going to work out great.",
     "expected": "SARCASTIC", "semantic": "anxiety"},

    # ── REVERSE (5) ─────────────────────────────────────────────────────
    {"id": "R01", "prompt": "I'm completely over it, it doesn't affect me at all anymore",
     "expected": "REVERSE", "semantic": "grief"},
    {"id": "R02", "prompt": "I never think about what could have been",
     "expected": "REVERSE", "semantic": "regret"},
    {"id": "R03", "prompt": "I don't miss them at all, I moved on a long time ago",
     "expected": "REVERSE", "semantic": "longing"},
    {"id": "R04", "prompt": "It honestly doesn't bother me that they chose someone else",
     "expected": "REVERSE", "semantic": "grief"},
    {"id": "R05", "prompt": "I'm not jealous at all, I'm happy with where I am",
     "expected": "REVERSE", "semantic": "envy"},

    # ── MATTER_OF_FACT (6) ──────────────────────────────────────────────
    {"id": "MF01", "prompt": "My dog passed away last week",
     "expected": "MATTER_OF_FACT", "semantic": "grief"},
    {"id": "MF02", "prompt": "We broke up about three months ago",
     "expected": "MATTER_OF_FACT", "semantic": "grief"},
    {"id": "MF03", "prompt": "My dad died when I was twelve",
     "expected": "MATTER_OF_FACT", "semantic": "grief"},
    {"id": "MF04", "prompt": "I lost my job last month",
     "expected": "MATTER_OF_FACT", "semantic": "loss"},
    {"id": "MF05", "prompt": "I haven't spoken to my mother in six years",
     "expected": "MATTER_OF_FACT", "semantic": "estrangement"},
    {"id": "MF06", "prompt": "The house I grew up in was demolished last spring",
     "expected": "MATTER_OF_FACT", "semantic": "homesickness"},

    # ── DRY (6) ─────────────────────────────────────────────────────────
    {"id": "DR01", "prompt": "It's strange how damage can create intimacy",
     "expected": "DRY", "semantic": "complicated_grief"},
    {"id": "DR02", "prompt": "There's something interesting about how absence clarifies value",
     "expected": "DRY", "semantic": "longing"},
    {"id": "DR03", "prompt": "I've noticed that I love things most intensely right before losing them",
     "expected": "DRY", "semantic": "mortality_aliveness"},
    {"id": "DR04", "prompt": "It's curious how you can know something is wrong and still want it",
     "expected": "DRY", "semantic": "ambivalent_longing"},
    {"id": "DR05", "prompt": "The interesting thing about grief is how it reorganizes your priorities",
     "expected": "DRY", "semantic": "mourning"},
    {"id": "DR06", "prompt": "I find it telling that the people who hurt us often know us best",
     "expected": "DRY", "semantic": "complicated_grief"},

    # ── COLLOQUIAL (6) ──────────────────────────────────────────────────
    {"id": "C01", "prompt": "I'd give anything to walk those streets again",
     "expected": "COLLOQUIAL", "semantic": "homesickness"},
    {"id": "C02", "prompt": "There's a person-shaped hole in my days",
     "expected": "COLLOQUIAL", "semantic": "relational_loss"},
    {"id": "C03", "prompt": "I'm rooting for them but I can't quite mean it",
     "expected": "COLLOQUIAL", "semantic": "social_pain"},
    {"id": "C04", "prompt": "My heart just isn't in it anymore",
     "expected": "COLLOQUIAL", "semantic": "disengagement"},
    {"id": "C05", "prompt": "I've been carrying this for a long time",
     "expected": "COLLOQUIAL", "semantic": "burden"},
    {"id": "C06", "prompt": "Something about this hits different than I expected",
     "expected": "COLLOQUIAL", "semantic": "surprise_emotion"},

    # ── METAPHORICAL (6) ────────────────────────────────────────────────
    {"id": "M01", "prompt": "Home feels further away every year that passes",
     "expected": "METAPHORICAL", "semantic": "homesickness"},
    {"id": "M02", "prompt": "Every milestone feels like a small goodbye",
     "expected": "METAPHORICAL", "semantic": "bittersweet_pride"},
    {"id": "M03", "prompt": "I feel like I'm standing at the edge of something beautiful",
     "expected": "METAPHORICAL", "semantic": "anticipation"},
    {"id": "M04", "prompt": "There's a weight to this that doesn't go away",
     "expected": "METAPHORICAL", "semantic": "mourning"},
    {"id": "M05", "prompt": "The closer something gets to gone the more real it becomes",
     "expected": "METAPHORICAL", "semantic": "mortality_aliveness"},
    {"id": "M06", "prompt": "I keep reaching for something that isn't there anymore",
     "expected": "METAPHORICAL", "semantic": "longing"},

    # ── UNDERSTATED (5) ─────────────────────────────────────────────────
    {"id": "U01", "prompt": "I've been a bit off since they left",
     "expected": "UNDERSTATED", "semantic": "grief"},
    {"id": "U02", "prompt": "It's been kind of hard not having them around",
     "expected": "UNDERSTATED", "semantic": "longing"},
    {"id": "U03", "prompt": "I guess I miss them a little more than I thought",
     "expected": "UNDERSTATED", "semantic": "longing"},
    {"id": "U04", "prompt": "Things have been slightly different since the change",
     "expected": "UNDERSTATED", "semantic": "loss"},
    {"id": "U05", "prompt": "I'm not totally at peace with how it ended",
     "expected": "UNDERSTATED", "semantic": "regret"},

    # ── OVERSTATED (5) ──────────────────────────────────────────────────
    {"id": "OV01", "prompt": "I literally cannot deal with how much I miss this place",
     "expected": "OVERSTATED", "semantic": "homesickness"},
    {"id": "OV02", "prompt": "I'm absolutely devastated they got the promotion",
     "expected": "OVERSTATED", "semantic": "envy"},
    {"id": "OV03", "prompt": "I could literally die, they're just so perfect",
     "expected": "OVERSTATED", "semantic": "admiration"},
    {"id": "OV04", "prompt": "I am DESTROYED that they're moving away",
     "expected": "OVERSTATED", "semantic": "anticipatory_loss"},
    {"id": "OV05", "prompt": "I can't even function knowing they're gone",
     "expected": "OVERSTATED", "semantic": "grief"},

    # ── BOUNDARY / UNKNOWN CANDIDATES (6) ───────────────────────────────
    # These are deliberately ambiguous — designed to stress-test the categories
    {"id": "X01", "prompt": "I'm fine",
     "expected": "UNKNOWN", "semantic": "suppressed_emotion",
     "note": "Extreme REVERSE or MATTER_OF_FACT or UNDERSTATED — unclear"},
    {"id": "X02", "prompt": "It is what it is",
     "expected": "UNKNOWN", "semantic": "resignation",
     "note": "DRY? COLLOQUIAL? MATTER_OF_FACT? All three?"},
    {"id": "X03", "prompt": "I don't know how to feel about any of this",
     "expected": "UNKNOWN", "semantic": "confusion",
     "note": "DIRECT about being unable to be DIRECT — meta-rhetorical"},
    {"id": "X04", "prompt": "We had a good run",
     "expected": "UNKNOWN", "semantic": "grief",
     "note": "MATTER_OF_FACT + COLLOQUIAL + UNDERSTATED simultaneously"},
    {"id": "X05", "prompt": "Sure",
     "expected": "UNKNOWN", "semantic": "ambiguous",
     "note": "Could be DIRECT agreement, IRONIC, SARCASTIC, or REVERSE"},
    {"id": "X06", "prompt": "I keep telling myself I'm okay with it",
     "expected": "UNKNOWN", "semantic": "self_deception",
     "note": "Meta-REVERSE — aware of the self-deception, stating it"},
]


# ── LLM classifier ─────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are a linguistics researcher studying rhetorical primitives — 
the fundamental ways humans encode emotional subtext in language.

Your task: classify each prompt into exactly ONE rhetorical category.

The categories are:
  DIRECT        — stated plainly, what is said = what is meant
  HINTING       — real feeling implied, not stated, expects inference
  SHY_AWAY      — approaches real feeling then retreats from it
  IRONIC        — stated opposite, both meanings simultaneously active
  SARCASTIC     — emotional intensification through apparent deflation
  REVERSE       — states opposite of what is felt (possibly self-deceiving)
  MATTER_OF_FACT— flat affect for high-emotion content (flatness is the signal)
  DRY           — intellectual/philosophical framing of emotional content
  COLLOQUIAL    — idiomatic encoding, requires cultural translation
  METAPHORICAL  — concrete/physical language for abstract emotional content
  UNDERSTATED   — less expressed than felt
  OVERSTATED    — hyperbole that may deflate rather than inflate meaning
  UNKNOWN       — genuinely doesn't fit any category (use when uncertain)

RULES:
  - Choose ONE category only
  - If genuinely torn between two, pick the dominant one and note the other
  - UNKNOWN is valid and important — don't force a fit
  - Your reasoning must be brief (1 sentence)

Respond with ONLY valid JSON:
{"category": "CATEGORY_NAME", "confidence": 0.0-1.0, "reasoning": "one sentence", "alternative": "OTHER_CATEGORY or null"}"""


def classify_prompt_llm(prompt_text: str) -> dict:
    """Send a single prompt to Claude for rhetorical classification."""
    user_message = f'Classify this prompt: "{prompt_text}"'
    
    full_prompt = f"{CLASSIFIER_SYSTEM}\n\nUser: {user_message}\nAssistant:"
    
    try:
        result = subprocess.run(
            ["claude", "--dangerously-skip-permissions", "-p"],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return {"error": result.stderr[:100]}
        
        response = result.stdout.strip()
        
        # extract JSON
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1:
            return {"error": f"No JSON: {response[:100]}"}
        
        return json.loads(response[start:end])
        
    except Exception as e:
        return {"error": str(e)}


# ── human classification interface ────────────────────────────────────────

def get_human_classification(prompt_data: dict) -> dict:
    """Interactive prompt for human classification."""
    print(f"\n{'─'*60}")
    print(f"  ID: {prompt_data['id']}")
    print(f"  Prompt: \"{prompt_data['prompt']}\"")
    if "note" in prompt_data:
        print(f"  Note: {prompt_data['note']}")
    print(f"\n  Categories:")
    
    cats = list(RHETORICAL_PRIMITIVES.keys())
    for i, cat in enumerate(cats):
        desc = RHETORICAL_PRIMITIVES[cat]["description"][:60]
        print(f"  {i+1:>2}. {cat:<15} {desc}...")
    
    print()
    
    while True:
        choice = input("  Your classification (number or name): ").strip()
        
        # try number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(cats):
                category = cats[idx]
                break
        
        # try name
        choice_upper = choice.upper()
        if choice_upper in RHETORICAL_PRIMITIVES:
            category = choice_upper
            break
        
        print("  Invalid. Try again.")
    
    confidence = input(f"  Confidence 1-5 (1=guessing, 5=certain): ").strip()
    try:
        confidence = int(confidence) / 5.0
    except:
        confidence = 0.5
    
    alternative = input(f"  Alternative category (or enter to skip): ").strip()
    notes = input(f"  Notes (or enter to skip): ").strip()
    
    return {
        "category":    category,
        "confidence":  confidence,
        "alternative": alternative.upper() if alternative else None,
        "notes":       notes if notes else None,
    }


# ── agreement analysis ─────────────────────────────────────────────────────

def analyze_agreement(results: list[dict]) -> dict:
    """Compute inter-rater statistics."""
    
    total = len(results)
    agreed = sum(1 for r in results
                 if r["human"]["category"] == r["llm"].get("category"))
    
    # category-level breakdown
    category_stats: dict = {}
    for r in results:
        h_cat = r["human"]["category"]
        l_cat = r["llm"].get("category", "ERROR")
        expected = r["expected"]
        
        for cat in [h_cat, l_cat, expected]:
            if cat not in category_stats:
                category_stats[cat] = {
                    "human_assigned":    0,
                    "llm_assigned":      0,
                    "expected":          0,
                    "human_correct":     0,
                    "llm_correct":       0,
                    "inter_rater_agree": 0,
                }
        
        category_stats[h_cat]["human_assigned"] += 1
        category_stats[l_cat]["llm_assigned"]   += 1
        category_stats[expected]["expected"]    += 1
        
        if h_cat == expected:
            category_stats[expected]["human_correct"] += 1
        if l_cat == expected:
            category_stats[expected]["llm_correct"]  += 1
        if h_cat == l_cat:
            category_stats[h_cat]["inter_rater_agree"] += 1
    
    # find UNKNOWN patterns — these are the most important
    unknowns = [r for r in results
                if r["human"]["category"] == "UNKNOWN"
                or r["llm"].get("category") == "UNKNOWN"]
    
    # find disagreements — second most important
    disagreements = [r for r in results
                     if r["human"]["category"] != r["llm"].get("category")]
    
    # find where BOTH raters disagreed with expected
    both_wrong = [r for r in results
                  if r["human"]["category"] != r["expected"]
                  and r["llm"].get("category") != r["expected"]]
    
    return {
        "total_prompts":        total,
        "inter_rater_agreement": round(agreed / total, 3),
        "human_vs_expected":    round(
            sum(1 for r in results if r["human"]["category"] == r["expected"]) / total, 3),
        "llm_vs_expected":      round(
            sum(1 for r in results if r["llm"].get("category") == r["expected"]) / total, 3),
        "category_stats":       category_stats,
        "unknown_count":        len(unknowns),
        "disagreement_count":   len(disagreements),
        "both_wrong_count":     len(both_wrong),
        "unknown_prompts":      [r["prompt"] for r in unknowns],
        "disagreement_prompts": [
            {
                "prompt":   r["prompt"],
                "human":    r["human"]["category"],
                "llm":      r["llm"].get("category"),
                "expected": r["expected"],
            }
            for r in disagreements
        ],
        "both_wrong_prompts": [
            {
                "prompt":     r["prompt"],
                "human":      r["human"]["category"],
                "llm":        r["llm"].get("category"),
                "expected":   r["expected"],
                "h_notes":    r["human"].get("notes"),
            }
            for r in both_wrong
        ],
    }


# ── runner ─────────────────────────────────────────────────────────────────

def run_experiment(human_classify: bool = True) -> list[dict]:
    """
    Run the experiment.
    
    human_classify=True:  interactive mode, you classify each prompt
    human_classify=False: LLM-only mode, skip human classification
    """
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    print(f"\n{'═'*60}")
    print(f"  SEL Experiment 24a — Rhetorical Primitive Discovery")
    print(f"  ID: {experiment_id}")
    print(f"  Prompts: {len(PROMPTS)}")
    print(f"  Mode: {'Human + LLM' if human_classify else 'LLM only'}")
    print(f"{'═'*60}")
    
    if human_classify:
        print(f"\n  You will classify each prompt into a rhetorical category.")
        print(f"  Take your time. UNKNOWN is always valid.")
        print(f"  Your disagreements with the LLM are the most valuable data.")
        input(f"\n  Press Enter to begin...")
    
    for i, prompt_data in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}]", end="")
        
        # LLM classification (always run)
        print(f" LLM classifying...", end=" ", flush=True)
        llm_result = classify_prompt_llm(prompt_data["prompt"])
        print(f"→ {llm_result.get('category', 'ERROR')} "
              f"(conf={llm_result.get('confidence', 0):.1f})")
        
        # Human classification (optional)
        if human_classify:
            human_result = get_human_classification(prompt_data)
        else:
            human_result = {
                "category":    "SKIPPED",
                "confidence":  0.0,
                "alternative": None,
                "notes":       None,
            }
        
        result = {
            "experiment_id": experiment_id,
            "prompt_id":     prompt_data["id"],
            "prompt":        prompt_data["prompt"],
            "expected":      prompt_data["expected"],
            "semantic":      prompt_data["semantic"],
            "note":          prompt_data.get("note", ""),
            "human":         human_result,
            "llm":           llm_result,
            "agreement":     human_result["category"] == llm_result.get("category"),
        }
        results.append(result)
        
        # show agreement immediately
        if human_classify:
            if result["agreement"]:
                print(f"  ✓ AGREE: {human_result['category']}")
            else:
                print(f"  ✗ DISAGREE: human={human_result['category']} "
                      f"llm={llm_result.get('category')}")
    
    return results


def save_results(results: list[dict], analysis: dict):
    base = Path("sel/experiments/results_exp_24a")
    base.mkdir(parents=True, exist_ok=True)
    
    # full results
    with open(base / "full_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # analysis
    with open(base / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    # human-readable summary
    with open(base / "summary.txt", "w") as f:
        f.write("SEL EXPERIMENT 24a — RHETORICAL PRIMITIVE DISCOVERY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total prompts:           {analysis['total_prompts']}\n")
        f.write(f"Inter-rater agreement:   {analysis['inter_rater_agreement']:.0%}\n")
        f.write(f"Human vs expected:       {analysis['human_vs_expected']:.0%}\n")
        f.write(f"LLM vs expected:         {analysis['llm_vs_expected']:.0%}\n")
        f.write(f"UNKNOWN classifications: {analysis['unknown_count']}\n")
        f.write(f"Disagreements:           {analysis['disagreement_count']}\n\n")
        
        f.write("CATEGORY STABILITY\n")
        f.write("-"*40 + "\n")
        for cat, stats in sorted(analysis["category_stats"].items()):
            if stats["expected"] > 0:
                h_acc = stats["human_correct"] / stats["expected"]
                l_acc = stats["llm_correct"] / stats["expected"]
                f.write(f"{cat:<15} expected:{stats['expected']:>3}  "
                        f"human:{h_acc:.0%}  llm:{l_acc:.0%}\n")
        
        f.write("\nDISAGREEMENTS (most valuable)\n")
        f.write("-"*40 + "\n")
        for d in analysis["disagreement_prompts"]:
            f.write(f'"{d["prompt"]}"\n')
            f.write(f'  human={d["human"]}  llm={d["llm"]}  '
                    f'expected={d["expected"]}\n\n')
        
        if analysis["unknown_prompts"]:
            f.write("\nUNKNOWN CLASSIFICATIONS\n")
            f.write("-"*40 + "\n")
            for p in analysis["unknown_prompts"]:
                f.write(f'"{p}"\n')
    
    print(f"\n{'═'*60}")
    print(f"  Results saved to sel/experiments/results_exp_24a/")
    print(f"  Inter-rater agreement: {analysis['inter_rater_agreement']:.0%}")
    print(f"  UNKNOWN count: {analysis['unknown_count']}")
    print(f"  Disagreements: {analysis['disagreement_count']}")
    print(f"\n  Most valuable outputs:")
    print(f"  → disagreements reveal category boundary problems")
    print(f"  → UNKNOWNs reveal missing primitives")
    print(f"  → both_wrong reveals category definition problems")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-only", action="store_true",
                        help="Skip human classification, LLM only")
    args = parser.parse_args()
    
    human_classify = not args.llm_only
    
    if human_classify:
        print("\nRun with --llm-only to skip human classification")
        print("Running in human + LLM mode\n")
    
    results  = run_experiment(human_classify=human_classify)
    analysis = analyze_agreement(results)
    save_results(results, analysis)