"""Visualizes compression threshold experiment results.

Reads results/analysis.json and writes three figures to results/figures/:

  heatmap_correctness.png
      One heatmap per model. Rows = problem type, columns = compression level.
      Cell color = mean correctness (green=1.0, red=0.0). Compression cliff
      appears as the column where color shifts from green to red.

  curves_correctness.png
      One line plot per model showing correctness across compression levels
      for each problem type. Horizontal dashed line at cliff threshold (0.8).

  heatmap_restatement.png
      Same layout as correctness heatmap but showing question_restatement rate.
      A high value at L3/L4 indicates the model is internally decompressing
      compressed prompts to NL before reasoning (key SRM signal).

Usage:
    python visualize.py
    python visualize.py --input results/analysis.json --output results/figures
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

PROBLEM_TYPES = ["math", "algebra", "logic", "causal", "code"]
LEVEL_LABELS = ["L0\nNatural", "L1\nAbbrev.", "L2\nHybrid", "L3\nSymbolic", "L4\nExtreme"]
CLIFF_THRESHOLD = 0.8


def load_analysis(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def build_matrix(series_list: List[dict], metric: str) -> np.ndarray:
    """Build a (problem_type × compression_level) matrix for a given metric.

    Returns shape (len(PROBLEM_TYPES), 5) with NaN for missing combinations.
    """
    mat = np.full((len(PROBLEM_TYPES), 5), np.nan)
    for series in series_list:
        pt = series["problem_type"]
        if pt not in PROBLEM_TYPES:
            continue
        row = PROBLEM_TYPES.index(pt)
        for ls in series["levels"]:
            col = ls["compression_level"]
            if 0 <= col < 5:
                mat[row, col] = ls.get(metric, np.nan)
    return mat


def _annotate_cells(ax, matrix: np.ndarray) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "black" if 0.25 < val < 0.75 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)


def plot_heatmap(ax, matrix: np.ndarray, title: str,
                 vmin: float = 0.0, vmax: float = 1.0, cmap: str = "RdYlGn"):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(5))
    ax.set_xticklabels(LEVEL_LABELS, fontsize=8)
    ax.set_yticks(range(len(PROBLEM_TYPES)))
    ax.set_yticklabels([t.capitalize() for t in PROBLEM_TYPES], fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    _annotate_cells(ax, matrix)
    return im


def plot_correctness_curves(ax, series_list: List[dict], model_name: str) -> None:
    for pt in PROBLEM_TYPES:
        matching = [s for s in series_list if s["problem_type"] == pt]
        if not matching:
            continue
        levels = sorted(matching[0]["levels"], key=lambda x: x["compression_level"])
        xs = [ls["compression_level"] for ls in levels]
        ys = [ls["mean_correctness"] for ls in levels]
        ax.plot(xs, ys, marker="o", label=pt.capitalize(), linewidth=1.5, markersize=5)

    ax.axhline(CLIFF_THRESHOLD, color="gray", linestyle="--", linewidth=0.8,
               label=f"Cliff threshold ({CLIFF_THRESHOLD})")
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["L0", "L1", "L2", "L3", "L4"])
    ax.set_xlabel("Compression level", fontsize=9)
    ax.set_ylabel("Mean correctness", fontsize=9)
    ax.set_title(f"Correctness curves — {model_name}", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize compression threshold results")
    parser.add_argument("--input", default=str(RESULTS_DIR / "analysis.json"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "figures"))
    args = parser.parse_args()

    figures_dir = Path(args.output)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_series = load_analysis(args.input)
    models = sorted({s["model"] for s in all_series})
    n_models = len(models)

    # --- Correctness heatmap ---
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        model_series = [s for s in all_series if s["model"] == model]
        mat = build_matrix(model_series, "mean_correctness")
        im = plot_heatmap(ax, mat, title=model)
    fig.colorbar(im, ax=axes, label="Mean correctness", fraction=0.02, pad=0.04)
    fig.suptitle(
        "Compression Threshold — Correctness by Level and Problem Type",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = figures_dir / "heatmap_correctness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # --- Correctness curves ---
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        model_series = [s for s in all_series if s["model"] == model]
        plot_correctness_curves(ax, model_series, model)
    plt.tight_layout()
    out = figures_dir / "curves_correctness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # --- Restatement rate heatmap ---
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        model_series = [s for s in all_series if s["model"] == model]
        mat = build_matrix(model_series, "restatement_rate")
        im = plot_heatmap(ax, mat, title=model, cmap="Blues")
    fig.colorbar(im, ax=axes, label="Restatement rate", fraction=0.02, pad=0.04)
    fig.suptitle(
        "Question Restatement Rate — Model Decompresses Compressed Prompt to NL",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = figures_dir / "heatmap_restatement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    print("\nAll figures written to", figures_dir)


if __name__ == "__main__":
    main()
