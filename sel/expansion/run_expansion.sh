#!/bin/bash
# sel/expansion/run_expansion.sh
# Usage: bash run_expansion.sh
# Stop: Ctrl+C
# Monitor: tail -f expansion.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$SCRIPT_DIR/expansion_engine.py"
LOG="$SCRIPT_DIR/expansion.log"
STATE="$SCRIPT_DIR/expansion_state.json"
SLEEP_NORMAL=2      # seconds between successful calls
SLEEP_RATE_LIMIT=60 # seconds when rate limited
SLEEP_ERROR=10      # seconds on other errors
MAX_CONSECUTIVE_ERRORS=5

consecutive_errors=0
iteration=0

echo "$(date): SEL Expansion Loop starting" | tee -a "$LOG"
echo "$(date): State file: $STATE" | tee -a "$LOG"
echo "$(date): Press Ctrl+C to stop" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

while true; do
    iteration=$((iteration + 1))
    
    result=$(python3 "$ENGINE" --state "$STATE" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        consecutive_errors=0
        echo "$(date) [iter=$iteration] OK: $result" | tee -a "$LOG"
        sleep $SLEEP_NORMAL
        
    elif [ $exit_code -eq 42 ]; then
        # Rate limit signal from engine
        consecutive_errors=0
        echo "$(date) [iter=$iteration] RATE_LIMIT: sleeping ${SLEEP_RATE_LIMIT}s" | tee -a "$LOG"
        sleep $SLEEP_RATE_LIMIT
        
    elif [ $exit_code -eq 43 ]; then
        # All jobs complete signal
        echo "$(date) [iter=$iteration] COMPLETE: all expansion jobs done" | tee -a "$LOG"
        break
        
    else
        consecutive_errors=$((consecutive_errors + 1))
        echo "$(date) [iter=$iteration] ERROR ($consecutive_errors): $result" | tee -a "$LOG"
        
        if [ $consecutive_errors -ge $MAX_CONSECUTIVE_ERRORS ]; then
            echo "$(date) FATAL: $MAX_CONSECUTIVE_ERRORS consecutive errors, stopping" | tee -a "$LOG"
            exit 1
        fi
        sleep $SLEEP_ERROR
    fi
done

echo "$(date): Expansion loop finished after $iteration iterations" | tee -a "$LOG"