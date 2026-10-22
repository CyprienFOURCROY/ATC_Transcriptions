#!/usr/bin/env python3
"""ATCO2 transcript viewer.

Run:  python3 viewer/app.py [--port 8000] [--no-browser]

Serves a local dashboard to browse the ATCO2 1h test set: play each recording,
follow the human transcript (with callsign / command / value tagging) and
compare it with the automatic ASR transcript (.cnet). Standard library only.
"""
import argparse
import json
import mimetypes
import re
import threading
import wave
import webbrowser
import xml.etree.ElementTree as ET
from collections import Counter
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DATA = RAW / "ATCO2-ASRdataset-v1_beta" / "DATA"
XML_TAGGED = RAW / "xml_tagged"
STATIC = Path(__file__).resolve().parent / "static"

TAG_RE = re.compile(r"\[#(\w+)\](.*?)\[/#\1\]", re.S)
NAME_RE = re.compile(r"^(?P<icao>[A-Z]{4})_(?P<city>[A-Z]+)_(?P<channel>.+)_(?P<freq>\d+_\d+MHz)_(?P<date>\d{8})_(?P<time>\d{6})$")


def parse_tagged(text):
    """Split '[#callsign]Oscar Kilo[/#callsign] descend' into [{tag, text}, ...]."""
    spans, pos = [], 0
    for m in TAG_RE.finditer(text):
        if m.start() > pos and text[pos:m.start()].strip():
            spans.append({"tag": None, "text": text[pos:m.start()].strip()})
        if m.group(2).strip():
            spans.append({"tag": m.group(1), "text": m.group(2).strip()})
        pos = m.end()
    if text[pos:].strip():
        spans.append({"tag": None, "text": text[pos:].strip()})
    return spans


def read_info(path):
    info, callsigns, in_cs = {}, [], False
    if not path.exists():
        return info, callsigns
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if in_cs:
            if ":" in line:
                code, spoken = line.split(":", 1)
                callsigns.append({"code": code.strip(), "spoken": spoken.strip()})
            continue
        if line.startswith("callsigns nearby"):
            in_cs = True
        elif ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info, callsigns


def read_cnet(path):
    """Best path of the Kaldi confusion network -> list of {start, end, word, conf}."""
    words = []
    if not path.exists():
        return words
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        start, dur, word, conf = float(p[2]), float(p[3]), p[4], float(p[5])
        if word != "<eps>":
            words.append({"start": start, "end": round(start + dur, 2), "word": word, "conf": conf})
    return words


def read_segments(stem):
    path = XML_TAGGED / f"{stem}.xml"
    if not path.exists():
        path = DATA / f"{stem}.xml"
    segs = []
    for s in ET.parse(path).getroot().iter("segment"):
        text = s.findtext("text") or ""
        segs.append({
            "start": float(s.findtext("start")),
            "end": float(s.findtext("end")),
            "speaker": s.findtext("speaker"),
            "label": s.findtext("speaker_label") or "",
            "spans": parse_tagged(text),
            "text": " ".join(sp["text"] for sp in parse_tagged(text)),
            "correct_transcript": s.findtext("tags/correct_transcript") == "1",
            "non_english": s.findtext("tags/non_english") == "1",
        })
    return segs


def wav_duration(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


@lru_cache(maxsize=1)
def catalog():
    files = []
    for wav in sorted(DATA.glob("*.wav")):
        stem = wav.stem
        m = NAME_RE.match(stem)
        meta = m.groupdict() if m else {"icao": "?", "city": "?", "channel": "?", "freq": "?", "date": "", "time": ""}
        segs = read_segments(stem)
        tags = Counter(sp["tag"] for s in segs for sp in s["spans"] if sp["tag"])
        files.append({
            "id": stem,
            "airport": f'{meta["icao"]} {meta["city"].title()}',
            "icao": meta["icao"],
            "channel": meta["channel"].replace("_", " "),
            "frequency": meta["freq"].replace("_", ".").replace("MHz", " MHz"),
            "datetime": f'{meta["date"][:4]}-{meta["date"][4:6]}-{meta["date"][6:]} {meta["time"][:2]}:{meta["time"][2:4]}:{meta["time"][4:]}',
            "duration": round(wav_duration(wav), 2),
            "n_segments": len(segs),
            "n_words": sum(len(s["text"].split()) for s in segs),
            "speakers": sorted({s["speaker"] for s in segs}),
            "tags": dict(tags),
            "preview": " / ".join(s["text"] for s in segs)[:200],
        })
    return files


@lru_cache(maxsize=1)
def summary():
    files = catalog()
    by_airport = Counter()
    dur_airport = Counter()
    tags = Counter()
    for f in files:
        by_airport[f["airport"]] += 1
        dur_airport[f["airport"]] += f["duration"]
        tags.update(f["tags"])
    return {
        "n_files": len(files),
        "total_duration": round(sum(f["duration"] for f in files), 1),
        "n_segments": sum(f["n_segments"] for f in files),
        "n_words": sum(f["n_words"] for f in files),
        "airports": [{"airport": a, "files": n, "duration": round(dur_airport[a], 1)} for a, n in by_airport.most_common()],
        "tags": dict(tags.most_common()),
    }


def file_detail(stem):
    info, callsigns = read_info(DATA / f"{stem}.info")
    return {
        "id": stem,
        "segments": read_segments(stem),
        "asr": read_cnet(DATA / f"{stem}.cnet"),
        "info": info,
        "callsigns": callsigns,
    }


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        """Serve a file, honouring Range requests so the audio element can seek."""
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else end
            elif m.group(2):
                start = size - int(m.group(2))
            end = min(end, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            self.wfile.write(fh.read(end - start + 1))

    def safe_stem(self, raw):
        stem = Path(unquote(raw)).name
        return stem if (DATA / f"{stem}.wav").exists() else None

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self.send_file(STATIC / "index.html")
            if path == "/api/files":
                return self.send_json(catalog())
            if path == "/api/summary":
                return self.send_json(summary())
            if path.startswith("/api/file/"):
                stem = self.safe_stem(path[len("/api/file/"):])
                if stem:
                    return self.send_json(file_detail(stem))
            if path.startswith("/audio/"):
                stem = self.safe_stem(path[len("/audio/"):].removesuffix(".wav"))
                if stem:
                    return self.send_file(DATA / f"{stem}.wav")
            self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(f"Dataset not found at {DATA}")
    print("Indexing recordings...", flush=True)
    catalog()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"ATCO2 viewer running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
