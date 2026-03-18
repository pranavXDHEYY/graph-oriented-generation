#!/bin/bash
# sel/expansion/monitor.sh
# Run in a separate terminal: bash monitor.sh

STATE="$(dirname "$0")/expansion_state.json"
LOG="$(dirname "$0")/expansion.log"

while true; do
    clear
    echo "═══════════════════════════════════════════"
    echo "  SEL EXPANSION MONITOR"
    echo "  $(date)"
    echo "═══════════════════════════════════════════"
    
    if [ -f "$STATE" ] && [ -s "$STATE" ]; then
    python3 -c "
import json
s = json.load(open('$STATE'))
print(f'  Iterations:        {s[\"iterations\"]}')
print(f'  Templates added:   {s[\"stats\"][\"templates_added\"]}')
print(f'  Signals added:     {s[\"stats\"][\"signals_added\"]}')
print(f'  Rules suggested:   {s[\"stats\"][\"rules_suggested\"]}')
print(f'  Variants added:    {s[\"stats\"][\"variants_added\"]}')
print(f'  Templates reviewed:{s[\"stats\"][\"templates_reviewed\"]}')
print()
print('  Next job:', ['TEMPLATE_GAP','SIGNAL_GAP','RULE_GAP',
    'VARIANT_GAP','QUALITY_REVIEW'][s['job_cursor'] % 5])
"
else
    echo "  Waiting for first iteration to complete..."
fi
    
    echo ""
    echo "─── Last 10 operations ─────────────────────"
    if [ -f "$LOG" ]; then
        tail -10 "$LOG"
    fi
    
    echo ""
    echo "─── Template library size ──────────────────"
    python3 -c "
import json, pathlib
p = pathlib.Path('$STATE').parent.parent / 'data' / 'response_templates.json'
if p.exists():
    d = json.loads(p.read_text())
    t = d.get('templates', d)
    total = sum(
        sum(len(v) for v in concept.get('variants',{}).values())
        for concept in t.values()
    )
    print(f'  Concepts: {len(t)}')
    print(f'  Total templates: {total}')
    print(f'  File size: {p.stat().st_size / 1024:.1f} KB')
" 2>/dev/null
    
    sleep 5
done