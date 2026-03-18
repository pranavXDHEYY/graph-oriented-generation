#!/usr/bin/env python3
"""
sel/expansion/expansion_engine.py

One unit of SEL expansion work per invocation.
Called repeatedly by run_expansion.sh.

Exit codes:
  0  = success
  42 = rate limited, caller should sleep
  43 = all work complete
  1  = error
"""
import json
import sys
import argparse
import subprocess
import random
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
TEMPLATES_PATH = ROOT / "sel" / "data" / "response_templates.json"
TAXONOMY_PATH  = ROOT / "sel" / "core" / "composition_rules.json"
PRIMITIVES_PATH= ROOT / "sel" / "data" / "wierzbicka_primitives.json"
GRAPH_PATH     = ROOT / "sel" / "core" / "primitive_graph.json"

# ── job types (rotate evenly) ──────────────────────────────────────────────────
JOB_TYPES = [
    "TEMPLATE_GAP",   # add missing templates to existing concepts
    "SIGNAL_GAP",     # add missing signal words to parser
    "RULE_GAP",       # identify missing rule classes
    "VARIANT_GAP",    # add context variants to existing templates
    "QUALITY_REVIEW", # flag/replace weak templates
]

# ── prompts (token-efficient, structured output only) ─────────────────────────
PROMPTS = {

"TEMPLATE_GAP": """You are expanding an emotional language template library.
Current concept: {concept}
Rule class: {rule_class}  
Description: {description}
Existing template count: {existing_count}
Existing variants: {existing_variants}

Add 3 new high-quality templates for this concept.
Requirements:
- Each template: 1-2 sentences maximum
- No "I'm sorry", no "I understand", no advice-giving
- Must feel like something a wise friend would say
- Must be emotionally precise, not generic
- Different from existing variants listed above

Respond with ONLY valid JSON, no other text:
{{"new_templates": {{"variant_name": ["template1", "template2", "template3"]}}}}

If concept is fully covered, respond: {{"new_templates": {{}}}}""",

"SIGNAL_GAP": """You are expanding an emotional language parser's signal word map.
Current signals for primitive {primitive}:
{current_signals}

The parser maps natural language words/phrases to semantic primitives.
Primitive layer: {layer} ({primitive_type})

Add 10 signal words/phrases that should map to {primitive}.
Focus on: colloquial expressions, metaphors, indirect references humans use.
Avoid: words already in the list above.

Respond with ONLY valid JSON, no other text:
{{"new_signals": ["word1", "phrase2", "expression3", ...]}}""",

"RULE_GAP": """You are analyzing a semantic composition rule taxonomy.
Existing rule classes: {existing_rules}

These rules describe how semantic primitives combine to produce emotional concepts.
Example: WANT + GRIEF = longing (Rule A: Desire x Loss)

Identify 3 missing rule classes — primitive combinations that produce 
recognizable human emotional concepts not yet covered.

For each: name the primitives, name the concept, give the rule description.

Respond with ONLY valid JSON, no other text:
{{"new_rules": [
  {{"operator": "PRIMITIVE1", "seed": "PRIMITIVE2", 
    "concept": "concept_name", "description": "what this feels like",
    "example_prompt": "I feel X when Y"}}
]}}""",

"VARIANT_GAP": """You are expanding context variants for an emotional template.
Concept: {concept}
Existing variants: {existing_variants}
Sample existing templates: {sample_templates}

Context variants allow the selector to pick the right template for specific situations.
Example: "homesickness" has variants: default, hometown, neighborhood, gone_place, pet_death

Identify 2 missing context variants for this concept.
Name them and provide 3 templates each.

Respond with ONLY valid JSON, no other text:
{{"new_variants": {{
  "variant_name": ["template1", "template2", "template3"],
  "variant_name2": ["template1", "template2", "template3"]
}}}}""",

"QUALITY_REVIEW": """You are reviewing emotional response templates for quality.
Concept: {concept}
Templates to review:
{templates_to_review}

Rate each template 1-5 where:
5 = precise, human, emotionally resonant
4 = good, minor improvements possible  
3 = acceptable but generic
2 = weak, too vague or wrong register
1 = bad, robotic/advice-giving/wrong concept

Flag any rated 1-2 for replacement and provide better alternatives.

Respond with ONLY valid JSON, no other text:
{{"ratings": {{"template_text": score}},
 "replacements": {{"bad_template": "better_template"}}}}"""
}

# ── state management ───────────────────────────────────────────────────────────

def load_state(state_path: Path) -> dict:
    if state_path.exists() and state_path.stat().st_size > 0:
        try:
            return json.loads(state_path.read_text())
        except json.JSONDecodeError:
            pass  # corrupted, start fresh
    return {
        "created": datetime.now().isoformat(),
        "iterations": 0,
        "job_cursor": 0,
        "concept_cursors": {},
        "completed_jobs": {},
        "stats": {
            "templates_added": 0,
            "signals_added": 0,
            "rules_suggested": 0,
            "variants_added": 0,
            "templates_reviewed": 0,
        }
    }

def save_state(state: dict, state_path: Path):
    state_path.write_text(json.dumps(state, indent=2))

# ── data loading ───────────────────────────────────────────────────────────────

def load_templates() -> dict:
    if TEMPLATES_PATH.exists():
        data = json.loads(TEMPLATES_PATH.read_text())
        return data.get("templates", data)
    return {}

def load_taxonomy() -> dict:
    if TAXONOMY_PATH.exists():
        return json.loads(TAXONOMY_PATH.read_text())
    return {}

def load_primitives() -> dict:
    if PRIMITIVES_PATH.exists():
        return json.loads(PRIMITIVES_PATH.read_text())
    return {}

def save_templates(templates: dict):
    existing = {}
    if TEMPLATES_PATH.exists():
        data = json.loads(TEMPLATES_PATH.read_text())
        if "templates" in data:
            existing = data
            existing["templates"] = templates
        else:
            existing = {"metadata": {}, "templates": templates}
    else:
        existing = {"metadata": {
            "version": "2.0",
            "generated_by": "SEL expansion engine",
            "last_updated": datetime.now().isoformat()
        }, "templates": templates}
    existing["metadata"]["last_updated"] = datetime.now().isoformat()
    TEMPLATES_PATH.write_text(json.dumps(existing, indent=2))

# ── claude call ────────────────────────────────────────────────────────────────

def call_claude(prompt: str) -> tuple[str, bool]:
    """
    Call Claude Haiku via claude CLI.
    Returns (response_text, was_rate_limited)
    """
    try:
        result = subprocess.run(
            [
                "claude",
                "--dangerously-skip-permissions",
                "--model", "claude-haiku-4-5-20251001",
                "-p",                # non-interactive output (--print)
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if "rate limit" in result.stderr.lower() or \
           "rate limit" in result.stdout.lower() or \
           result.returncode == 429:
            return "", True
            
        if result.returncode != 0:
            raise RuntimeError(f"Claude exit {result.returncode}: {result.stderr[:200]}")
            
        return result.stdout.strip(), False
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Claude call timed out after 60s")

# ── job executors ──────────────────────────────────────────────────────────────

def job_template_gap(state: dict) -> str:
    templates = load_templates()
    concepts = list(templates.keys())
    if not concepts:
        return "no concepts found"
    
    cursor = state["concept_cursors"].get("TEMPLATE_GAP", 0)
    concept = concepts[cursor % len(concepts)]
    state["concept_cursors"]["TEMPLATE_GAP"] = cursor + 1
    
    data = templates[concept]
    variants = data.get("variants", {})
    existing_count = sum(len(v) for v in variants.values())
    existing_variant_names = list(variants.keys())
    
    prompt = PROMPTS["TEMPLATE_GAP"].format(
        concept=concept,
        rule_class=data.get("rule_class", "unknown"),
        description=data.get("description", ""),
        existing_count=existing_count,
        existing_variants=json.dumps(existing_variant_names)
    )
    
    response, rate_limited = call_claude(prompt)
    if rate_limited:
        sys.exit(42)
    
    result = json.loads(extract_json(response))
    new_templates = result.get("new_templates", {})
    
    added = 0
    for variant_name, template_list in new_templates.items():
        if variant_name not in variants:
            variants[variant_name] = []
        # deduplicate
        existing = set(variants[variant_name])
        for t in template_list:
            if t not in existing:
                variants[variant_name].append(t)
                added += 1
    
    if added > 0:
        templates[concept]["variants"] = variants
        save_templates(templates)
        state["stats"]["templates_added"] += added
    
    return f"TEMPLATE_GAP {concept}: +{added} templates"

def job_signal_gap(state: dict) -> str:
    # Use curated list of primitives — do not read from wierzbicka_primitives.json
    primitive_list = [
        "GRIEF", "FEAR", "JOY", "WANT", "KNOW",
        "TIME", "PLACE", "SOMEONE", "FEEL", "MOVE",
        "THINK", "SAY", "DO", "HAPPEN", "GOOD", "BAD",
        "BIG", "SMALL", "VERY", "MORE", "LIKE", "NOT",
        "BECAUSE", "IF", "WHEN", "BEFORE", "AFTER",
        "UNDER", "SAME", "OTHER", "LIVE", "DIE",
        "ADMIRATION", "PRIDE", "ANGER", "SADNESS",
        "EXCITEMENT", "ANXIETY", "NOSTALGIA", "WONDER",
        "ENVY", "SHAME", "RELIEF", "DISGUST", "AWE",
        "SURPRISE", "GUILT", "LONELINESS", "GRATITUDE"
    ]
    
    cursor = state["concept_cursors"].get("SIGNAL_GAP", 0)
    primitive = primitive_list[cursor % len(primitive_list)]
    state["concept_cursors"]["SIGNAL_GAP"] = cursor + 1
    
    # load current signal map from decomposer
    decomposer_path = ROOT / "sel" / "core" / "decomposer.py"
    current_signals = []
    if decomposer_path.exists():
        content = decomposer_path.read_text()
        # extract signals for this primitive (rough parse)
        import re
        pattern = rf'"{primitive}"[^[]*\[([^\]]+)\]'
        matches = re.findall(pattern, content)
        if matches:
            current_signals = [s.strip().strip('"') 
                              for s in matches[0].split(',')]
    
    layer = "0b" if primitive in ["GRIEF","FEAR","JOY","ANGER","PRIDE",
                                   "SADNESS","EXCITEMENT","ANXIETY",
                                   "NOSTALGIA","ENVY","AWE","WONDER",
                                   "SHAME","GUILT","RELIEF","DISGUST",
                                   "LONELINESS","GRATITUDE","ADMIRATION"] \
            else "0a"
    primitive_type = "phenomenological" if layer == "0b" else "structural"
    
    prompt = PROMPTS["SIGNAL_GAP"].format(
        primitive=primitive,
        current_signals=json.dumps(current_signals[:20]),  # token limit
        layer=layer,
        primitive_type=primitive_type
    )
    
    response, rate_limited = call_claude(prompt)
    if rate_limited:
        sys.exit(42)
    
    result = json.loads(extract_json(response))
    new_signals = result.get("new_signals", [])
    
    # append to a signals expansion file
    signals_path = ROOT / "sel" / "expansion" / "signal_expansions.json"
    expansions = {}
    if signals_path.exists():
        expansions = json.loads(signals_path.read_text())
    
    if primitive not in expansions:
        expansions[primitive] = []
    
    existing = set(expansions[primitive])
    added = [s for s in new_signals if s not in existing]
    expansions[primitive].extend(added)
    
    signals_path.write_text(json.dumps(expansions, indent=2))
    state["stats"]["signals_added"] += len(added)
    
    return f"SIGNAL_GAP {primitive}: +{len(added)} signals"

def job_rule_gap(state: dict) -> str:
    taxonomy = load_taxonomy()
    
    existing_rules = []
    if isinstance(taxonomy, dict):
        for key, val in taxonomy.items():
            if isinstance(val, dict):
                existing_rules.append(
                    f"{key}: {val.get('name','?')} — {val.get('description','')}"
                )
    
    # token-efficient: send max 30 existing rules
    rules_summary = "\n".join(existing_rules[:30])
    
    prompt = PROMPTS["RULE_GAP"].format(
        existing_rules=rules_summary
    )
    
    response, rate_limited = call_claude(prompt)
    if rate_limited:
        sys.exit(42)
    
    result = json.loads(extract_json(response))
    new_rules = result.get("new_rules", [])
    
    # append to rule suggestions file
    suggestions_path = ROOT / "sel" / "expansion" / "rule_suggestions.json"
    suggestions = []
    if suggestions_path.exists():
        suggestions = json.loads(suggestions_path.read_text())
    
    suggestions.extend(new_rules)
    suggestions_path.write_text(json.dumps(suggestions, indent=2))
    state["stats"]["rules_suggested"] += len(new_rules)
    
    return f"RULE_GAP: +{len(new_rules)} rule suggestions"

def job_variant_gap(state: dict) -> str:
    templates = load_templates()
    concepts = list(templates.keys())
    if not concepts:
        return "no concepts found"
    
    cursor = state["concept_cursors"].get("VARIANT_GAP", 0)
    concept = concepts[cursor % len(concepts)]
    state["concept_cursors"]["VARIANT_GAP"] = cursor + 1
    
    data = templates[concept]
    variants = data.get("variants", {})
    existing_variant_names = list(variants.keys())
    
    # sample 2 templates for context
    sample = []
    for v in list(variants.values())[:2]:
        if v:
            sample.append(v[0])
    
    prompt = PROMPTS["VARIANT_GAP"].format(
        concept=concept,
        existing_variants=json.dumps(existing_variant_names),
        sample_templates=json.dumps(sample)
    )
    
    response, rate_limited = call_claude(prompt)
    if rate_limited:
        sys.exit(42)
    
    result = json.loads(extract_json(response))
    new_variants = result.get("new_variants", {})
    
    added = 0
    for variant_name, template_list in new_variants.items():
        if variant_name not in variants:
            variants[variant_name] = template_list
            added += len(template_list)
    
    if added > 0:
        templates[concept]["variants"] = variants
        save_templates(templates)
        state["stats"]["variants_added"] += added
    
    return f"VARIANT_GAP {concept}: +{added} variant templates"

def job_quality_review(state: dict) -> str:
    templates = load_templates()
    concepts = list(templates.keys())
    if not concepts:
        return "no concepts found"
    
    cursor = state["concept_cursors"].get("QUALITY_REVIEW", 0)
    concept = concepts[cursor % len(concepts)]
    state["concept_cursors"]["QUALITY_REVIEW"] = cursor + 1
    
    data = templates[concept]
    variants = data.get("variants", {})
    
    # collect up to 10 templates to review (token efficient)
    to_review = []
    for variant_templates in variants.values():
        to_review.extend(variant_templates[:3])
        if len(to_review) >= 10:
            break
    
    if not to_review:
        return f"QUALITY_REVIEW {concept}: no templates to review"
    
    prompt = PROMPTS["QUALITY_REVIEW"].format(
        concept=concept,
        templates_to_review=json.dumps(to_review[:10])
    )
    
    response, rate_limited = call_claude(prompt)
    if rate_limited:
        sys.exit(42)
    
    result = json.loads(extract_json(response))
    replacements = result.get("replacements", {})
    
    replaced = 0
    if replacements:
        for variant_name, variant_templates in variants.items():
            for i, t in enumerate(variant_templates):
                if t in replacements:
                    variants[variant_name][i] = replacements[t]
                    replaced += 1
        
        if replaced > 0:
            templates[concept]["variants"] = variants
            save_templates(templates)
    
    state["stats"]["templates_reviewed"] += len(to_review)
    return f"QUALITY_REVIEW {concept}: reviewed {len(to_review)}, replaced {replaced}"

# ── utilities ──────────────────────────────────────────────────────────────────

def extract_json(text: str) -> str:
    """Extract JSON from Claude response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    # find first { and last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in: {text[:200]}")
    return text[start:end]

JOB_EXECUTORS = {
    "TEMPLATE_GAP":   job_template_gap,
    "SIGNAL_GAP":     job_signal_gap,
    "RULE_GAP":       job_rule_gap,
    "VARIANT_GAP":    job_variant_gap,
    "QUALITY_REVIEW": job_quality_review,
}

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    
    state_path = Path(args.state)
    state = load_state(state_path)
    state["iterations"] += 1
    
    # round-robin job selection
    job_type = JOB_TYPES[state["job_cursor"] % len(JOB_TYPES)]
    state["job_cursor"] += 1
    
    try:
        result = JOB_EXECUTORS[job_type](state)
        save_state(state, state_path)
        print(result)
        sys.exit(0)
        
    except SystemExit:
        save_state(state, state_path)
        raise
        
    except json.JSONDecodeError as e:
        save_state(state, state_path)
        print(f"JSON parse error in {job_type}: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        save_state(state, state_path)
        print(f"Error in {job_type}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()