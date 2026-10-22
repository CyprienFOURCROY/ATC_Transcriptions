#!/usr/bin/env python3
"""Score every system in outputs/transcripts/ against the human references.

  python3 scripts/evaluate.py                       # all systems found
  python3 scripts/evaluate.py whisper-turbo atco2-cnet --common   # same files for both

Metrics (on normalised text, see common.normalize):
  WER       word error rate over all words (substitutions + deletions + insertions) / reference words
  callsign  share of tagged callsigns that appear word-for-word in the hypothesis
Writes outputs/eval/summary.csv and outputs/eval/<system>_per_file.csv.
"""
import argparse
import csv
from collections import defaultdict

import jiwer

from common import EVAL, REFERENCES, TRANSCRIPTS, load_jsonl, normalize


def score(pairs):
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    o = jiwer.process_words(refs, hyps)
    n = o.hits + o.substitutions + o.deletions
    return {"wer": o.wer, "sub": o.substitutions / n, "del": o.deletions / n, "ins": o.insertions / n, "ref_words": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systems", nargs="*", help="folder names under outputs/transcripts (default: all)")
    ap.add_argument("--common", action="store_true", help="score only recordings transcribed by every listed system")
    args = ap.parse_args()

    if not REFERENCES.exists():
        raise SystemExit("Run scripts/prepare_references.py first.")
    refs = {r["id"]: r for r in load_jsonl(REFERENCES) if r["norm"]}
    systems = args.systems or sorted(p.parent.name for p in TRANSCRIPTS.glob("*/hyps.jsonl"))
    if not systems:
        raise SystemExit("No hyps.jsonl found in outputs/transcripts/. Run transcribe.py or cnet_baseline.py first.")

    all_hyps = {s: {h["id"]: normalize(h["text"]) for h in load_jsonl(TRANSCRIPTS / s / "hyps.jsonl")} for s in systems}
    if args.common:
        refs = {i: r for i, r in refs.items() if all(i in h for h in all_hyps.values())}

    EVAL.mkdir(parents=True, exist_ok=True)
    summary = []
    for system in systems:
        hyps = all_hyps[system]
        ids = [i for i in refs if i in hyps]
        if not ids:
            print(f"{system}: no overlapping recordings, skipped")
            continue
        overall = score([(refs[i]["norm"], hyps[i]) for i in ids])

        cs_total = cs_hit = 0
        by_airport = defaultdict(list)
        with open(EVAL / f"{system}_per_file.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "airport", "wer", "ref_words", "callsigns_found", "callsigns_total", "reference", "hypothesis"])
            for i in ids:
                ref, hyp = refs[i], hyps[i]
                padded = f" {hyp} "
                found = sum(f" {c} " in padded for c in ref["callsigns"])
                cs_total += len(ref["callsigns"])
                cs_hit += found
                by_airport[ref["airport"]].append((ref["norm"], hyp))
                w.writerow([i, ref["airport"], round(jiwer.wer(ref["norm"], hyp), 4), len(ref["norm"].split()),
                            found, len(ref["callsigns"]), ref["norm"], hyp])

        row = {"system": system, "files": len(ids), **{k: round(v, 4) for k, v in overall.items()},
               "callsign_acc": round(cs_hit / cs_total, 4) if cs_total else None}
        for airport, pairs in sorted(by_airport.items()):
            row[f"wer_{airport}"] = round(score(pairs)["wer"], 4)
        summary.append(row)

    cols = list(dict.fromkeys(k for r in summary for k in r))
    with open(EVAL / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)

    airports = [c for c in cols if c.startswith("wer_")]
    print(f"\n{'system':28} {'files':>5} {'WER':>6} {'sub':>5} {'del':>5} {'ins':>5} {'callsign':>8}   " + " ".join(f"{a[4:8]:>5}" for a in airports))
    for r in summary:
        cs = f"{r['callsign_acc']:.1%}" if r["callsign_acc"] is not None else "-"
        print(f"{r['system']:28} {r['files']:5} {r['wer']:6.1%} {r['sub']:5.1%} {r['del']:5.1%} {r['ins']:5.1%} {cs:>8}   "
              + " ".join(f"{r[a]:5.1%}" if r.get(a) is not None else "    -" for a in airports))
    print(f"\nDetails: {EVAL}/summary.csv and {EVAL}/<system>_per_file.csv")


if __name__ == "__main__":
    main()
