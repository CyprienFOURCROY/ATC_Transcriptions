#!/usr/bin/env python3
"""Transcribe the ATCO2 recordings with a faster-whisper (CTranslate2) model.

Examples:
  python3 scripts/transcribe.py                                   # whisper-turbo, no prompt
  python3 scripts/transcribe.py --prompt atc                      # -> whisper-turbo-atc
  python3 scripts/transcribe.py --prompt callsigns                # -> whisper-turbo-callsigns
  python3 scripts/transcribe.py --limit 10                        # quick test

Writes one JSON per recording to outputs/transcripts/<name>/ (finished files
are skipped on re-run, so an interrupted run resumes) and collects them into
outputs/transcripts/<name>/hyps.jsonl for evaluate.py. The exact prompt given
for each recording is saved in outputs/transcripts/<name>/prompts.json.
"""
import argparse
import json
import os
import time
from pathlib import Path
import wave

import numpy as np

from common import DATA, MODELS, TRANSCRIPTS, clean_callsign, read_nearby_callsigns, recording_ids, write_jsonl

# Shows Whisper the ATC style we want: numbers and letters spelled out as spoken.
ATC_PROMPT = ("Air traffic control radio. Oscar Kilo Papa Mike Bravo, descend flight level one hundred, "
              "QNH one zero one three. Runway two four, cleared to land, wind two seven zero degrees one five knots. "
              "Contact tower one one eight decimal three.")

# faster-whisper keeps only the last (448 // 2 - 1) prompt tokens and silently drops the start
MAX_PROMPT_TOKENS = 223

PROMPT_COMMENTS = {
    "none": "No prompt: Whisper transcribes each recording with no prior text. This is the plain baseline.",
    "atc": ("Same prompt for every recording ('fixed_part'): a short example of ATC radio phraseology with numbers "
            "and letters spelled out. It is passed as Whisper's initial_prompt, i.e. treated as if it were the "
            "transcript of the audio just before, so Whisper copies its style (spelled-out words, ATC vocabulary)."),
    "callsigns": ("Personalised prompt per recording: the same ATC example as the 'atc' run ('fixed_part'), followed by "
                  "'Nearby traffic:' and the spoken callsigns of the aircraft listed in that recording's .info file "
                  "(aircraft near the airport at recording time, in the file's order, i.e. alphabetical by ICAO code). "
                  "Callsigns are added only while the prompt fits in Whisper's {max_tokens}-token prompt window, so "
                  "nothing is truncated; 'callsigns_used' vs 'callsigns_available' shows how many fit. Spellings are "
                  "aligned with the references ('Ryan Air' -> 'Ryanair', 'Csa / Czech' -> 'Csa'). Passed as Whisper's "
                  "initial_prompt so that the callsign actually spoken becomes more likely."),
}


class PromptBuilder:
    def __init__(self, mode, model_dir):
        from tokenizers import Tokenizer
        self.mode = mode
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    def n_tokens(self, text):
        # same encoding faster-whisper applies to initial_prompt
        return len(self.tok.encode(" " + text.strip(), add_special_tokens=False).ids)

    def build(self, rec_id):
        """Return (prompt, details) for one recording."""
        if self.mode == "none":
            return None, None
        if self.mode == "atc":
            return ATC_PROMPT, {"prompt": ATC_PROMPT, "tokens": self.n_tokens(ATC_PROMPT)}
        available = [clean_callsign(c) for c in read_nearby_callsigns(rec_id)]
        used, prompt = [], ATC_PROMPT
        for c in available:
            candidate = f"{ATC_PROMPT} Nearby traffic: {', '.join(used + [c])}."
            if self.n_tokens(candidate) > MAX_PROMPT_TOKENS:
                break
            used.append(c)
            prompt = candidate
        return prompt, {"prompt": prompt, "tokens": self.n_tokens(prompt),
                        "callsigns_used": len(used), "callsigns_available": len(available)}

    def write(self, out_dir, ids):
        """Save every recording's exact prompt to prompts.json, with an explanation at the top."""
        prompts = {rec_id: self.build(rec_id)[1] for rec_id in ids}
        entries = [p for p in prompts.values() if p]
        stats = {"recordings": len(ids), "with_prompt": len(entries), "max_tokens": MAX_PROMPT_TOKENS}
        if self.mode == "callsigns":
            stats["all_callsigns_fit"] = sum(p["callsigns_used"] == p["callsigns_available"] for p in entries)
            stats["no_callsigns_available"] = sum(p["callsigns_available"] == 0 for p in entries)
        doc = {
            "comment": PROMPT_COMMENTS[self.mode].format(max_tokens=MAX_PROMPT_TOKENS),
            "mode": self.mode,
            "fixed_part": None if self.mode == "none" else ATC_PROMPT,
            "stats": stats,
            "prompts": prompts,
        }
        (out_dir / "prompts.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        return stats


def load_wav(path):
    """16 kHz mono PCM -> float32 array. Avoids faster-whisper's PyAV decoder (broken with av>=16)."""
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, f"unexpected format: {path}"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODELS / "whisper-turbo"), help="local CTranslate2 model dir or a faster-whisper model name")
    ap.add_argument("--name", help="output folder name (default: model folder name + prompt mode)")
    ap.add_argument("--prompt", choices=["none", "atc", "callsigns"], default="none",
                    help="none | atc: ATC-style example prompt | callsigns: atc + nearby callsigns from .info (fitted to 223 tokens)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--compute-type", default="int8", help="int8 (fast on CPU), int8_float32, float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--vad", action="store_true", help="enable Silero VAD filtering")
    ap.add_argument("--limit", type=int, help="only the first N recordings")
    ap.add_argument("--ids", nargs="*", help="only these recording ids")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--prompts-only", action="store_true", help="only write prompts.json, do not run the model")
    args = ap.parse_args()

    name = args.name or os.path.basename(args.model.rstrip("/")) + ("" if args.prompt == "none" else f"-{args.prompt}")
    out_dir = TRANSCRIPTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.ids or recording_ids()
    if args.limit:
        ids = ids[: args.limit]

    prompter = PromptBuilder(args.prompt, args.model)
    stats = prompter.write(out_dir, args.ids or recording_ids())
    print(f"Wrote {out_dir / 'prompts.json'}: {stats}")
    if args.prompts_only:
        return

    from faster_whisper import WhisperModel

    print(f"Loading {args.model} ({args.device}, {args.compute_type}, {args.threads} threads)...", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, cpu_threads=args.threads)

    config = {k: v for k, v in vars(args).items() if k not in ("ids", "limit", "overwrite", "prompts_only")}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    t_start, audio_done, n_done = time.time(), 0.0, 0
    for i, rec_id in enumerate(ids, 1):
        out = out_dir / f"{rec_id}.json"
        if out.exists() and not args.overwrite:
            continue
        t0 = time.time()
        segments, info = model.transcribe(
            load_wav(DATA / f"{rec_id}.wav"),
            language="en",
            beam_size=args.beam_size,
            initial_prompt=prompter.build(rec_id)[0],
            condition_on_previous_text=False,
            vad_filter=args.vad,
            word_timestamps=True,
        )
        segs = [{
            "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip(), "prob": round(w.probability, 3)} for w in s.words or []],
        } for s in segments]
        elapsed = time.time() - t0
        out.write_text(json.dumps({"id": rec_id, "text": " ".join(s["text"] for s in segs), "duration": info.duration,
                                   "elapsed": round(elapsed, 2), "segments": segs}, ensure_ascii=False, indent=1))
        audio_done += info.duration
        n_done += 1
        rtf = (time.time() - t_start) / audio_done
        print(f"[{i}/{len(ids)}] {rec_id}  {info.duration:5.1f}s audio in {elapsed:4.1f}s  (RTF {rtf:.2f})  {segs[0]['text'][:70] if segs else ''}", flush=True)

    rows = [{"id": r["id"], "text": r["text"]} for r in (json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json")) if p.name not in ("config.json", "prompts.json"))]
    write_jsonl(out_dir / "hyps.jsonl", rows)
    print(f"Done: {n_done} new, {len(rows)} total in {out_dir / 'hyps.jsonl'}")


if __name__ == "__main__":
    main()
