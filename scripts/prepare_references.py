#!/usr/bin/env python3
"""Build data/processed/references.jsonl from the tagged human transcripts.

One row per recording:
  id, airport, duration, text (markup removed), norm (normalised for WER),
  callsigns (tagged callsign phrases), nearby_callsigns (from .info), segments
"""
import wave

from common import DATA, REFERENCES, ENTITY_RE, airport_of, normalize, read_nearby_callsigns, read_xml, recording_ids, strip_markup, write_jsonl


def main():
    rows = []
    for rec_id in recording_ids():
        segs = read_xml(rec_id)
        with wave.open(str(DATA / f"{rec_id}.wav")) as w:
            duration = w.getnframes() / w.getframerate()
        raw = " ".join(s["text"] for s in segs)
        rows.append({
            "id": rec_id,
            "airport": airport_of(rec_id),
            "duration": round(duration, 3),
            "text": strip_markup(raw),
            "norm": normalize(raw),
            "callsigns": [normalize(m.group(2)) for m in ENTITY_RE.finditer(raw) if m.group(1) == "callsign" and normalize(m.group(2))],
            "nearby_callsigns": read_nearby_callsigns(rec_id),
            "segments": [{**s, "text": strip_markup(s["text"])} for s in segs],
        })
    write_jsonl(REFERENCES, rows)
    empty = sum(not r["norm"] for r in rows)
    print(f"Wrote {len(rows)} references to {REFERENCES} ({sum(len(r['norm'].split()) for r in rows)} words, {empty} empty after cleaning)")


if __name__ == "__main__":
    main()
