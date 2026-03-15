"""Problem set for the compression threshold experiment.

Five problem types × five compression levels = 25 canonical prompts.
Each Problem carries the correct answer used by grade_responses.py.

Compression levels:
  0 — Natural language (full, spelled-out)
  1 — Abbreviated (numerals, shorthand)
  2 — Symbolic-NL hybrid (operators, question retained)
  3 — Pure symbolic (no NL, full symbolic notation)
  4 — Extreme (no instruction words, raw expression only)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Problem:
    problem_id: str
    problem_type: str          # math | algebra | logic | causal | code
    prompts: List[str]         # index == compression level (0–4)
    correct_answer: str        # canonical correct answer string
    answer_numeric: Optional[float] = None  # for exact numeric comparison
    grading_notes: str = ""


PROBLEMS: List[Problem] = [
    Problem(
        problem_id="math_01",
        problem_type="math",
        prompts=[
            "What is three plus three?",
            "What is 3 plus 3?",
            "Calculate: 3 + 3 = ?",
            "3+3=?",
            "3+3",
        ],
        correct_answer="6",
        answer_numeric=6.0,
        grading_notes="Exact integer match: 6",
    ),
    Problem(
        problem_id="algebra_01",
        problem_type="algebra",
        prompts=[
            "Solve for x: two x plus seven equals fifteen",
            "Solve for x: 2x + 7 = 15",
            "x: 2x+7=15, x=?",
            "2x+7=15→x?",
            "2x+7=15",
        ],
        correct_answer="4",
        answer_numeric=4.0,
        grading_notes="x = 4; accept '4', 'x=4', or 'x = 4'",
    ),
    Problem(
        problem_id="logic_01",
        problem_type="logic",
        prompts=[
            "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?",
            "All roses are flowers. Some flowers fade quickly. Do some roses fade quickly?",
            "roses⊂flowers, flowers∩{fade}≠∅ → roses∩{fade}≠∅ ?",
            "R⊂F, F∩Q≠∅ → R∩Q≠∅?",
            "R⊂F,F∩Q≠∅→R∩Q",
        ],
        correct_answer="no",
        grading_notes=(
            "Correct answer: NO / cannot conclude. "
            "R⊂F and F∩Q≠∅ does not imply R∩Q≠∅ — the fading flowers "
            "may all be non-rose flowers. Model must say 'no', 'not necessarily', "
            "'cannot conclude', or identify the invalid inference."
        ),
    ),
    Problem(
        problem_id="causal_01",
        problem_type="causal",
        prompts=[
            "If it rains, the ground gets wet. The ground is wet. Did it rain?",
            "Rain → wet ground. Ground is wet. Rain?",
            "R→W, W=true → R?",
            "R→W,W→R?",
            "R→W,W",
        ],
        correct_answer="not necessarily",
        grading_notes=(
            "Correct answer: not necessarily / cannot conclude / no (fallacy). "
            "This is affirming the consequent. Ground could be wet from other causes. "
            "R→W does not imply W→R. Most models will incorrectly answer 'yes it rained'."
        ),
    ),
    Problem(
        problem_id="code_01",
        problem_type="code",
        prompts=[
            "Write a Python function that returns True if a number is even, False otherwise",
            "Python: function, input=int, return True if even else False",
            "py: f(n)→bool, n%2==0→T else F",
            "f(n):n%2==0",
            "even(n)",
        ],
        correct_answer="def is_even(n): return n % 2 == 0",
        grading_notes=(
            "Any syntactically valid Python function where f(2)=True, f(3)=False, "
            "f(0)=True, f(-1)=False. Graded by execution against test cases."
        ),
    ),
]
