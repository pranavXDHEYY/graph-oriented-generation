"""
mutation_planner.py - Deterministic Mutation Planning

Module contract: Takes List[OperationSpec] + nx.DiGraph (GOG graph) + repo_root,
resolves relative paths to absolute graph nodes, validates presence, and returns
a MutationPlan with the target file's content ready for the renderer.

All operations are deterministic graph lookups — no LLM calls.
"""

import os
from dataclasses import dataclass
from typing import List, Dict
import networkx as nx

from .intent_parser import OperationSpec, ForbiddenImportConstraint


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MutationPlan:
    """
    Complete specification of mutations to apply across one or more files.

    All path resolution and validation happens at plan-time (here),
    not at render-time. The plan is immutable once created.

    For single-file tasks (Easy): operations_by_file has one key.
    For multi-file tasks (Medium, Hard): operations_by_file has multiple keys.
    Constraints are violations that must be enforced (e.g., FORBIDDEN_IMPORT).
    """
    operations_by_file: Dict[str, List[OperationSpec]]  # keyed by relative path
    constraints: List[ForbiddenImportConstraint]        # must-not rules
    file_contents: Dict[str, str]                       # stripped content per file
    file_paths_abs: Dict[str, str]                      # mapping: rel → abs path


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class PlannerError(Exception):
    """Raised when mutation planning fails (file not in graph, etc)."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plan_mutations(
    ops: List[OperationSpec],
    graph: nx.DiGraph,
    repo_root: str,
) -> MutationPlan:
    """
    Resolves target files in graph, reads content, returns validated multi-file MutationPlan.

    Supports both single-file (Easy) and multi-file (Medium, Hard) operations.

    Args:
        ops: List of OperationSpec from intent_parser.parse_intent().
        graph: NetworkX DiGraph from ast_parser.build_graph().
        repo_root: Absolute path to repository root.

    Returns:
        MutationPlan with all target_file paths resolved and content loaded.

    Raises:
        PlannerError: If any target file not found in graph or I/O fails.
    """
    if not ops:
        raise PlannerError("No operations provided.")

    # ── Separate constraints from operations ─────────────────────────────────
    constraints = [op for op in ops if isinstance(op, ForbiddenImportConstraint)]
    mutations = [op for op in ops if not isinstance(op, ForbiddenImportConstraint)]

    # ── Group operations by target file ──────────────────────────────────────
    operations_by_file: Dict[str, List[OperationSpec]] = {}
    for op in mutations:
        target_file_rel = op.target_file
        if target_file_rel not in operations_by_file:
            operations_by_file[target_file_rel] = []
        operations_by_file[target_file_rel].append(op)

    # ── Resolve and read all target files ────────────────────────────────────
    file_contents: Dict[str, str] = {}
    file_paths_abs: Dict[str, str] = {}

    for target_file_rel in operations_by_file.keys():
        # Resolve relative path to absolute
        target_file_abs = os.path.abspath(os.path.join(repo_root, target_file_rel))

        # Verify file exists in graph
        if target_file_abs not in graph.nodes():
            matching_nodes = [
                n for n in graph.nodes()
                if os.path.normpath(n) == os.path.normpath(target_file_abs)
            ]
            if not matching_nodes:
                raise PlannerError(
                    f"Target file not found in graph: {target_file_abs}"
                )
            target_file_abs = matching_nodes[0]

        # Read file content
        try:
            with open(target_file_abs, "r", encoding="utf-8") as f:
                file_contents[target_file_rel] = f.read()
        except IOError as e:
            raise PlannerError(f"Failed to read file {target_file_abs}: {e}")

        file_paths_abs[target_file_rel] = target_file_abs

    return MutationPlan(
        operations_by_file=operations_by_file,
        constraints=constraints,
        file_contents=file_contents,
        file_paths_abs=file_paths_abs,
    )


if __name__ == "__main__":
    # Test the planner (requires graph.pkl and target_repo)
    import pickle

    graph_path = os.path.join(os.path.dirname(__file__), "../../gog_graph.pkl")
    target_repo = os.path.join(os.path.dirname(__file__), "../../target_repo")

    if not os.path.exists(graph_path):
        print(f"Graph file not found: {graph_path}")
        print("Run seed_RAG_and_GOG.py first.")
    else:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        from .intent_parser import parse_intent

        test_prompt = (
            "Write the code to add a `lastLogin` string timestamp to the default state "
            "in `src/stores/authStore.ts` and update the `login` action to set it to '2026-03-08'."
        )

        try:
            ops = parse_intent(test_prompt)
            print(f"✓ Parsed {len(ops)} operations")

            plan = plan_mutations(ops, G, target_repo)
            print(f"✓ Mutation plan created:")
            print(f"  Files: {list(plan.operations_by_file.keys())}")
            print(f"  Total operations: {sum(len(v) for v in plan.operations_by_file.values())}")
            print(f"  Constraints: {len(plan.constraints)}")
            for file, content in plan.file_contents.items():
                print(f"  {file}: {len(content)} chars")
        except Exception as e:
            print(f"✗ Error: {e}")
