#!/usr/bin/env python3
"""
Experiment 15: SRM on Claude - Does Structure Help Bigger Models More?
"""

import subprocess
import json
import csv
import time
from datetime import datetime
from pathlib import Path

PROBLEMS = [
    {"id": 1, "type": "logic", "prompt": "All roses are flowers. Some flowers die quickly. Can we conclude some roses die quickly?", "keywords": ["no", "cant"]},
    {"id": 2, "type": "logic", "prompt": "If it rains, the ground gets wet. The ground is wet. Did it rain?", "keywords": ["no", "not"]},
    {"id": 3, "type": "logic", "prompt": "Every bird can fly. Penguins are birds. Can penguins fly?", "keywords": ["no", "cant"]},
    {"id": 4, "type": "math", "prompt": "Solve: 2x + 5 = 15. What is x?", "keywords": ["5", "x = 5"]},
    {"id": 5, "type": "math", "prompt": "What is 15% of 80?", "keywords": ["12", "twelve"]},
]

def call_claude(p):
    r = subprocess.run(["claude", "-p", p], capture_output=True, text=True, timeout=30)
    return r.stdout if r.stdout else r.stderr

def check(response, kw):
    r = response.lower()
    return any(k.lower() in r for k in kw)

print("=== EXP 15: Claude SRM Test ===\n")
for p in PROBLEMS:
    d = call_claude(f"Solve: {p['prompt']}")
    s = call_claude(f"Extract structure. Apply reasoning. Render. Problem: {p['prompt']}")
    dc = check(d, p["keywords"])
    sc = check(s, p["keywords"])
    print(f"[{p['id']}] Direct: {'Y' if dc else 'N'} | SRM: {'Y' if sc else 'N'}")
    time.sleep(0.3)
