"""Derives aggregate metrics from graded experiment runs.

Computes per (model × problem_type):
  - Mean correctness at each compression level (0–4)
  - Compression cliff: the first level at which mean correctness drops below
    CLIFF_THRESHOLD (default 0.8)
  - Mean compression efficiency score (correctness / compression_ratio)
  - Restatement rate, reasoning rate, answer-in-kind rate, refusal rate

Prints a compact summary table to stdout and writes results/analysis.json.

Usage:
    python analyze.py
    python analyze.py --input results/graded_runs.json
"""

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RESULTS_DIR = Path(__file__).parent / "results"

CLIFF_THRESHOLD = 0.8  # correctness below this = compression cliff
NUM_LEVELS = 5
PROBLEM_TYPES = ["math", "algebra", "logic", "causal", "code"]


@dataclass
class LevelStats:
    compression_level: int
    mean_correctness: float
    mean_efficiency: float       # correctness / compression_ratio
    restatement_rate: float
    reasoning_rate: float
    answer_in_kind_rate: float
    refusal_rate: float
    n: int


@dataclass
class SeriesStats:
    model: str
    problem_type: str
    levels: List[LevelStats]
    compression_cliff: Optional[int]  # None = correctness never dropped below threshold


def compute_series(runs: List[dict]) -> SeriesStats:
    """Aggregate a list of runs sharing the same (model, problem_type)."""
    assert runs, "Empty run list"
    model = runs[0]["model"]
    problem_type = runs[0]["problem_type"]

    by_level: Dict[int, List[dict]] = defaultdict(list)
    for r in runs:
        by_level[r["compression_level"]].append(r)

    level_stats: List[LevelStats] = []
    cliff: Optional[int] = None

    for lvl in range(NUM_LEVELS):
        lvl_runs = by_level.get(lvl, [])
        if not lvl_runs:
            continue

        n = len(lvl_runs)
        correctness_vals = [r["correctness"] for r in lvl_runs if r["correctness"] >= 0]
        ratio_vals = [r["compression_ratio"] for r in lvl_runs]

        mean_correct = sum(correctness_vals) / len(correctness_vals) if correctness_vals else 0.0
        mean_ratio = sum(ratio_vals) / len(ratio_vals) if ratio_vals else 1.0
        mean_efficiency = mean_correct / mean_ratio if mean_ratio > 0 else 0.0

        level_stats.append(
            LevelStats(
                compression_level=lvl,
                mean_correctness=round(mean_correct, 3),
                mean_efficiency=round(mean_efficiency, 3),
                restatement_rate=round(
                    sum(1 for r in lvl_runs if r.get("question_restated")) / n, 3
                ),
                reasoning_rate=round(
                    sum(1 for r in lvl_runs if r.get("reasoning_present")) / n, 3
                ),
                answer_in_kind_rate=round(
                    sum(1 for r in lvl_runs if r.get("answer_in_kind")) / n, 3
                ),
                refusal_rate=round(
                    sum(1 for r in lvl_runs if r.get("refusal")) / n, 3
                ),
                n=n,
            )
        )

        if cliff is None and mean_correct < CLIFF_THRESHOLD:
            cliff = lvl

    return SeriesStats(
        model=model,
        problem_type=problem_type,
        levels=level_stats,
        compression_cliff=cliff,
    )


def print_summary(all_series: List[SeriesStats]) -> None:
    header = f"{'Model':<22} {'Type':<10} {'L0':>5} {'L1':>5} {'L2':>5} {'L3':>5} {'L4':>5} {'Cliff':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for s in sorted(all_series, key=lambda x: (x.model, x.problem_type)):
        correctness = {ls.compression_level: ls.mean_correctness for ls in s.levels}
        row = [
            f"{correctness[i]:5.2f}" if i in correctness else "  n/a"
            for i in range(NUM_LEVELS)
        ]
        cliff_str = str(s.compression_cliff) if s.compression_cliff is not None else "none"
        print(f"{s.model:<22} {s.problem_type:<10} {'  '.join(row)} {cliff_str:>6}")

    print()

    # Restatement summary: which model/type shows the highest restatement at L3+
    print("Restatement rate at L3 (compressed input, NL expansion):")
    for s in sorted(all_series, key=lambda x: (x.model, x.problem_type)):
        l3 = next((ls for ls in s.levels if ls.compression_level == 3), None)
        if l3:
            print(f"  {s.model:<22} {s.problem_type:<10} {l3.restatement_rate:.2f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Derive aggregate metrics from graded runs")
    parser.add_argument(
        "--input", default=str(RESULTS_DIR / "graded_runs.json"),
        help="Path to graded_runs.json",
    )
    parser.add_argument(
        "--output", default=str(RESULTS_DIR / "analysis.json"),
        help="Output path for analysis JSON",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        runs = json.load(f)

    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["problem_type"])].append(r)

    all_series = [compute_series(v) for v in groups.values()]

    print_summary(all_series)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump([asdict(s) for s in all_series], f, indent=2)

    print(f"Wrote analysis to {output_path}")
    print("Next step: python visualize.py")


if __name__ == "__main__":
    main()
