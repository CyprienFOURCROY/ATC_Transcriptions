#!/bin/bash
# Run the three prompt modes on the same subset of recordings (ids in outputs/logs/subset_ids.txt), then score them.
set -e
cd "$(dirname "$0")/.."
IDS=$(cat outputs/logs/subset_ids.txt)
for mode in none atc callsigns; do
  echo "=== whisper prompt=$mode  $(date '+%H:%M:%S')"
  python3 -u scripts/transcribe.py --prompt $mode --ids $IDS 2>&1 | grep --line-buffered -v Warning | tee outputs/logs/transcribe_$mode.log
done
python3 scripts/evaluate.py whisper-turbo whisper-turbo-atc whisper-turbo-callsigns atco2-cnet --common | tee outputs/logs/evaluate.log
