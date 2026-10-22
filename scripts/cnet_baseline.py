#!/usr/bin/env python3
"""Extract the best path of the dataset's own ASR (.cnet) as a baseline system.

Writes outputs/transcripts/atco2-cnet/hyps.jsonl so evaluate.py can score it
alongside Whisper.
"""
from common import DATA, TRANSCRIPTS, recording_ids, write_jsonl


def best_path(path):
    words = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 6 and p[4] not in ("<eps>", "<gbg>"):
            words.append(p[4].replace("_", " "))
    return " ".join(words)


def main():
    rows, missing = [], 0
    for rec_id in recording_ids():
        path = DATA / f"{rec_id}.cnet"
        if path.exists():
            rows.append({"id": rec_id, "text": best_path(path)})
        else:
            missing += 1
    out = TRANSCRIPTS / "atco2-cnet" / "hyps.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} hypotheses to {out} ({missing} recordings without .cnet)")


if __name__ == "__main__":
    main()
