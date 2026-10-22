#!/bin/bash
# Fresh full pipeline: references, cnet baseline, three Whisper runs, evaluation.
# Usage: bash scripts/run_all.sh        (logs in outputs/logs/)
set -e
cd "$(dirname "$0")/.."
LOGS=outputs/logs
mkdir -p $LOGS

python3 scripts/prepare_references.py
python3 scripts/cnet_baseline.py
for mode in none atc callsigns; do
  echo "=== whisper prompt=$mode  $(date '+%H:%M:%S')"
  python3 scripts/transcribe.py --prompt $mode 2>&1 | grep -v Warning > $LOGS/transcribe_$mode.log
  tail -1 $LOGS/transcribe_$mode.log
done
python3 scripts/evaluate.py | tee $LOGS/evaluate.log
echo "=== finished $(date '+%H:%M:%S')"
