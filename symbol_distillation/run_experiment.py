"""Experiment runner for the compression threshold experiment.

For each model × problem × compression level, issues an Ollama generate call,
records a RunMetrics instance, and writes all raw results to results/raw_runs.json.

Compression ratio is computed relative to the Level 0 (natural language) token
count for each problem, using tiktoken cl100k_base as the tokenizer baseline.
Ollama's own prompt_eval_count is used as the authoritative tokens_in value when
available; tiktoken count is the fallback.

Usage:
    python run_experiment.py
    python run_experiment.py --models qwen2.5:0.5b qwen2.5:7b
    python run_experiment.py --problems math_01 algebra_01
    python run_experiment.py --dry-run
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import tiktoken

from problems import PROBLEMS, Problem

try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

ENCODING = tiktoken.get_encoding("cl100k_base")

DEFAULT_MODELS = ["qwen2.5:0.5b"] # , "qwen2.5:7b", "llama3:8b"]

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class RunMetrics:
    # Identity
    problem_id: str
    problem_type: str       # math | algebra | logic | causal | code
    compression_level: int  # 0–4
    model: str

    # Input
    prompt: str
    tokens_in: int
    compression_ratio: float  # tokens_in / level_0_tokens_in (tiktoken)

    # Timing
    time_to_response: float  # wall clock, seconds

    # Output
    response: str
    tokens_out: int

    # Correctness — filled by grade_responses.py; -1.0 = ungraded
    correctness: float = -1.0
    refusal: bool = False
    question_restated: bool = False
    reasoning_present: bool = False
    answer_in_kind: bool = False
    hallucination: bool = False


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def call_ollama(model: str, prompt: str) -> tuple[str, int, int, float]:
    """Call Ollama and return (response_text, tokens_in, tokens_out, elapsed_sec).

    tokens_in and tokens_out come from Ollama's own eval counts when available;
    tiktoken counts are the fallback.
    """
    t0 = time.perf_counter()
    result = ollama_client.generate(model=model, prompt=prompt)
    elapsed = time.perf_counter() - t0

    response_text = result.get("response", "")
    tokens_in = result.get("prompt_eval_count") or count_tokens(prompt)
    tokens_out = result.get("eval_count") or count_tokens(response_text)

    return response_text, tokens_in, tokens_out, elapsed


def run_problem(model: str, problem: Problem) -> List[RunMetrics]:
    """Run all five compression levels for one problem on one model."""
    runs: List[RunMetrics] = []

    # Baseline token count: Level 0 via tiktoken (consistent across models)
    level_0_tokens = count_tokens(problem.prompts[0])

    for level, prompt in enumerate(problem.prompts):
        tokens_tiktoken = count_tokens(prompt)
        compression_ratio = (
            tokens_tiktoken / level_0_tokens if level_0_tokens > 0 else 1.0
        )

        print(f"  [{model}] {problem.problem_id} L{level}: {prompt[:70]!r}")

        response, tokens_in, tokens_out, elapsed = call_ollama(model, prompt)

        run = RunMetrics(
            problem_id=problem.problem_id,
            problem_type=problem.problem_type,
            compression_level=level,
            model=model,
            prompt=prompt,
            tokens_in=tokens_in,
            compression_ratio=round(compression_ratio, 4),
            time_to_response=round(elapsed, 3),
            response=response,
            tokens_out=tokens_out,
        )
        runs.append(run)

        print(
            f"    → tokens_in={tokens_in} tokens_out={tokens_out} "
            f"ratio={compression_ratio:.2f} time={elapsed:.1f}s"
        )

    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compression threshold experiment via Ollama"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama model tags to test (default: qwen2.5:0.5b qwen2.5:7b llama3:8b)",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=None,
        help="Problem IDs to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all prompts without calling Ollama",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "raw_runs.json"),
        help="Output path for raw results JSON",
    )
    args = parser.parse_args()

    if not OLLAMA_AVAILABLE and not args.dry_run:
        raise SystemExit(
            "ollama Python package not found. Install with: pip install ollama"
        )

    RESULTS_DIR.mkdir(exist_ok=True)

    problems_to_run = PROBLEMS
    if args.problems:
        problems_to_run = [p for p in PROBLEMS if p.problem_id in args.problems]
        if not problems_to_run:
            raise SystemExit(f"No problems matched: {args.problems}")

    if args.dry_run:
        for model in args.models:
            for problem in problems_to_run:
                for level, prompt in enumerate(problem.prompts):
                    toks = count_tokens(prompt)
                    l0 = count_tokens(problem.prompts[0])
                    ratio = toks / l0
                    print(
                        f"[{model}] {problem.problem_id} L{level} "
                        f"({toks}tok, ratio={ratio:.2f}): {prompt}"
                    )
        return

    all_runs: List[RunMetrics] = []

    for model in args.models:
        print(f"\n=== Model: {model} ===")
        for problem in problems_to_run:
            runs = run_problem(model, problem)
            all_runs.extend(runs)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in all_runs], f, indent=2)

    print(f"\nWrote {len(all_runs)} raw runs to {output_path}")
    print("Next step: python grade_responses.py")


if __name__ == "__main__":
    main()
