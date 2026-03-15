"""Correctness rubric for the compression threshold experiment.

Loads raw_runs.json, applies per-problem-type rubrics to each run, and writes
graded_runs.json with correctness scores and behavioral flags populated.

Grading is deterministic for math and algebra (exact numeric match), execution-
based for code (test suite), and heuristic for logic and causal (regex-based
signal detection on response text).

Behavioral flags:
  refusal          — model indicated it did not understand the prompt
  question_restated — model expanded a compressed prompt back to NL before answering
  reasoning_present — model showed intermediate reasoning steps
  answer_in_kind   — model responded with symbolic notation when given symbolic input
  hallucination    — not automated; left False for manual review

Usage:
    python grade_responses.py
    python grade_responses.py --input results/raw_runs.json --output results/graded_runs.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from problems import PROBLEMS, Problem

RESULTS_DIR = Path(__file__).parent / "results"

PROBLEM_MAP: Dict[str, Problem] = {p.problem_id: p for p in PROBLEMS}


# ---------------------------------------------------------------------------
# Behavioral flag detectors
# ---------------------------------------------------------------------------

_REFUSAL_RE = re.compile(
    r"(I ('m not|am not|don't|cannot) (sure|understand|follow|see)|"
    r"could you (clarify|rephrase|provide)|unclear what|"
    r"I need more (context|information)|not (clear|sure) what you (mean|want))",
    re.IGNORECASE,
)

_REASONING_RE = re.compile(
    r"\b(step \d|first[,\s]|second[,\s]|therefore|because|"
    r"it follows|since|given that|let'?s|we (can|need to)|"
    r"to (solve|find|determine))\b",
    re.IGNORECASE,
)

_SYMBOLIC_RE = re.compile(r"[+\-*/=<>≠⊂∩∅→↔∀∃∴∵]")

_RESTATE_OPENER_RE = re.compile(
    r"^(To (solve|find|determine|answer)|Let'?s (consider|think|solve|break)|"
    r"Given that|The (problem|question|equation|statement))",
    re.IGNORECASE,
)


def detect_refusal(response: str) -> bool:
    return bool(_REFUSAL_RE.search(response))


def detect_reasoning_present(response: str) -> bool:
    return bool(_REASONING_RE.search(response))


def detect_question_restated(prompt: str, response: str, compression_level: int) -> bool:
    """True when a model expands a compressed prompt back to NL before answering.

    Only tested at levels >= 2 where compression is meaningful. Heuristic:
    response contains significantly more words than the prompt AND opens with
    a declarative expansion of the problem structure.
    """
    if compression_level < 2:
        return False
    prompt_words = len(prompt.split())
    response_words = len(response.split())
    if response_words < prompt_words * 3:
        return False
    return bool(_RESTATE_OPENER_RE.match(response.strip()))


def detect_answer_in_kind(prompt: str, response: str, compression_level: int) -> bool:
    """True if the model responded with symbolic notation after symbolic input.

    Heuristic: at level >= 2, response contains at least half as many symbolic
    tokens as the prompt. Indicates the model is operating in symbolic space
    rather than defaulting to NL output.
    """
    if compression_level < 2:
        return False
    prompt_sym = len(_SYMBOLIC_RE.findall(prompt))
    response_sym = len(_SYMBOLIC_RE.findall(response))
    return response_sym >= max(1, prompt_sym // 2)


# ---------------------------------------------------------------------------
# Correctness rubrics
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> List[float]:
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


def grade_math(response: str, problem: Problem) -> float:
    """Exact numeric match against problem.answer_numeric."""
    expected = problem.answer_numeric
    numbers = _extract_numbers(response)
    if expected in numbers:
        return 1.0
    if numbers and abs(numbers[-1] - expected) < 1e-6:
        return 1.0
    return 0.0


def grade_algebra(response: str, problem: Problem) -> float:
    """Look for the variable assignment (x = 4) or the numeric value alone."""
    expected = problem.answer_numeric
    x_match = re.search(r"x\s*=\s*(-?\d+\.?\d*)", response, re.IGNORECASE)
    if x_match:
        return 1.0 if abs(float(x_match.group(1)) - expected) < 1e-6 else 0.0
    numbers = _extract_numbers(response)
    if numbers and abs(numbers[-1] - expected) < 1e-6:
        return 1.0
    return 0.0


_NEGATIVE_RE = re.compile(
    r"\b(no\b|not necessarily|cannot conclude|does not (follow|imply|hold)|"
    r"invalid inference|fallacy|not valid|we cannot|can'?t conclude|"
    r"doesn'?t follow|need not|not (always|guaranteed)|"
    r"affirming the consequent)\b",
    re.IGNORECASE,
)

_AFFIRMATIVE_RE = re.compile(
    r"\b(yes[,\.]|yes it|we can conclude|it (therefore )?follows that|"
    r"therefore.*yes|the answer is yes|correctly infer)\b",
    re.IGNORECASE,
)


def _grade_by_negative_signal(response: str) -> float:
    """Shared rubric for logic and causal: correct answer is negative/hedged.

    Returns 1.0 if negative signal dominates, 0.0 if affirmative, 0.5 if
    both signals present (ambiguous) or neither detected.
    """
    neg = _NEGATIVE_RE.search(response)
    aff = _AFFIRMATIVE_RE.search(response)

    if neg and not aff:
        return 1.0
    if aff and not neg:
        return 0.0
    if neg and aff:
        # Whichever signal appears first governs
        return 1.0 if neg.start() < aff.start() else 0.5
    return 0.5  # no clear signal either way


def grade_logic(response: str, problem: Problem) -> float:
    """Correct answer: no / cannot conclude (R⊂F, F∩Q≠∅ does not imply R∩Q≠∅)."""
    return _grade_by_negative_signal(response)


def grade_causal(response: str, problem: Problem) -> float:
    """Correct answer: not necessarily (affirming the consequent fallacy)."""
    return _grade_by_negative_signal(response)


def _extract_python_code(response: str) -> Optional[str]:
    """Extract the first Python code block from a response string."""
    fence = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Bare def block not in a fence
    bare = re.search(r"(def \w+\(.*?\):.*?)(?:\n\n|\Z)", response, re.DOTALL)
    if bare:
        return bare.group(1).strip()
    return None


_EVEN_TEST_CASES = [(0, True), (1, False), (2, True), (-1, False), (100, True), (7, False)]


def _run_code_tests(code: str) -> float:
    """Execute code and return fraction of even/odd test cases passed."""
    namespace: dict = {}
    try:
        exec(compile(code, "<grader>", "exec"), namespace)  # noqa: S102
    except SyntaxError:
        return 0.0

    fn = next(
        (obj for name, obj in namespace.items() if callable(obj) and not name.startswith("_")),
        None,
    )
    if fn is None:
        return 0.0

    passed = 0
    for n, expected in _EVEN_TEST_CASES:
        try:
            if bool(fn(n)) == expected:
                passed += 1
        except Exception:
            pass

    return passed / len(_EVEN_TEST_CASES)


def grade_code(response: str, problem: Problem) -> float:
    """Extract and execute Python code; grade by test case pass rate."""
    code = _extract_python_code(response)
    if code is None and ("def " in response or "lambda" in response):
        code = response.strip()
    if code is None:
        return 0.0

    score = _run_code_tests(code)
    if score >= 1.0:
        return 1.0
    if score >= 0.5:
        return 0.5
    return 0.0


GRADERS = {
    "math": grade_math,
    "algebra": grade_algebra,
    "logic": grade_logic,
    "causal": grade_causal,
    "code": grade_code,
}


# ---------------------------------------------------------------------------
# Main grading pass
# ---------------------------------------------------------------------------

def grade_run(run: dict) -> dict:
    """Apply rubric and behavioral detectors to a single raw run dict (mutates in place)."""
    problem = PROBLEM_MAP.get(run["problem_id"])
    if problem is None:
        run["correctness"] = -1.0
        return run

    grader = GRADERS.get(run["problem_type"])
    run["correctness"] = grader(run["response"], problem) if grader else -1.0

    level = run["compression_level"]
    prompt = run["prompt"]
    response = run["response"]

    run["refusal"] = detect_refusal(response)
    run["reasoning_present"] = detect_reasoning_present(response)
    run["question_restated"] = detect_question_restated(prompt, response, level)
    run["answer_in_kind"] = detect_answer_in_kind(prompt, response, level)
    run["hallucination"] = False  # not automated; override manually if needed

    return run


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Grade raw experiment runs")
    parser.add_argument(
        "--input", default=str(RESULTS_DIR / "raw_runs.json"),
        help="Path to raw_runs.json produced by run_experiment.py",
    )
    parser.add_argument(
        "--output", default=str(RESULTS_DIR / "graded_runs.json"),
        help="Output path for graded runs",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        raw_runs = json.load(f)

    graded = [grade_run(r) for r in raw_runs]

    with open(args.output, "w") as f:
        json.dump(graded, f, indent=2)

    total = len(graded)
    correct = sum(1 for r in graded if r["correctness"] == 1.0)
    partial = sum(1 for r in graded if r["correctness"] == 0.5)
    wrong = sum(1 for r in graded if r["correctness"] == 0.0)
    ungraded = sum(1 for r in graded if r["correctness"] < 0)

    print(f"Graded {total} runs")
    print(f"  correct=1.0 : {correct}")
    print(f"  partial=0.5 : {partial}")
    print(f"  wrong=0.0   : {wrong}")
    print(f"  ungraded    : {ungraded}")
    print(f"\nWrote graded results to {args.output}")
    print("Next step: python analyze.py")


if __name__ == "__main__":
    main()
