"""Shared paths, dataset loading and ATC text normalisation."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DATA = RAW / "ATCO2-ASRdataset-v1_beta" / "DATA"
XML_TAGGED = RAW / "xml_tagged"
PROCESSED = ROOT / "data" / "processed"
REFERENCES = PROCESSED / "references.jsonl"
MODELS = ROOT / "models"
TRANSCRIPTS = ROOT / "outputs" / "transcripts"
EVAL = ROOT / "outputs" / "eval"

ENTITY_RE = re.compile(r"\[#(\w+)\](.*?)\[/#\1\]", re.S)


def recording_ids():
    return sorted(p.stem for p in DATA.glob("*.wav"))


def airport_of(rec_id):
    return "_".join(rec_id.split("_")[:2])


def read_xml(rec_id):
    path = XML_TAGGED / f"{rec_id}.xml"
    if not path.exists():
        path = DATA / f"{rec_id}.xml"
    segs = []
    for s in ET.parse(path).getroot().iter("segment"):
        segs.append({
            "start": float(s.findtext("start")),
            "end": float(s.findtext("end")),
            "speaker": s.findtext("speaker"),
            "label": s.findtext("speaker_label") or "",
            "text": s.findtext("text") or "",
        })
    return segs


def read_nearby_callsigns(rec_id):
    """Spoken forms of the callsigns listed in the .info file."""
    path = DATA / f"{rec_id}.info"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if "callsigns nearby:" not in text:
        return []
    return [line.split(":", 1)[1].strip() for line in text.split("callsigns nearby:", 1)[1].splitlines() if ":" in line]


def clean_callsign(spoken):
    """'Csa / Czech Two Delta Zulu' -> 'Csa Two Delta Zulu', 'Ryan Air' -> 'Ryanair' (as the references write it)."""
    spoken = re.sub(r"\s*/\s*\S+", "", spoken)
    for name in COMPOUND_NAMES:
        spoken = re.sub(name, name.replace(" ", "").capitalize(), spoken, flags=re.I)
    return " ".join(spoken.split())


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Normalisation
#
# References spell everything out ("flight level one hundred", "one two one
# decimal seven five") while Whisper tends to write digits ("FL100", "121.75").
# Both sides go through the same pipeline so WER measures recognition errors,
# not formatting:  digits -> spoken words, tens/teens -> digit words
# ("twenty three" -> "two three"), NATO spelling variants unified.
# --------------------------------------------------------------------------- #

DIGITS = "zero one two three four five six seven eight nine".split()
TEENS = {"ten": "one zero", "eleven": "one one", "twelve": "one two", "thirteen": "one three",
         "fourteen": "one four", "fifteen": "one five", "sixteen": "one six",
         "seventeen": "one seven", "eighteen": "one eight", "nineteen": "one nine"}
TENS = {"twenty": "two", "thirty": "three", "forty": "four", "fifty": "five",
        "sixty": "six", "seventy": "seven", "eighty": "eight", "ninety": "nine"}
SPELLING = {"alfa": "alpha", "juliett": "juliet", "xray": "x ray", "niner": "nine",
            "okay": "ok", "decimals": "decimal", "fl": "flight level"}
NATO = dict(zip("abcdefghijklmnopqrstuvwxyz",
                "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
                "november oscar papa quebec romeo sierra tango uniform victor whiskey x_ray yankee zulu".split()))
# Airline radio names written as one or two words (refs say both "Ryan Air" and "Ryanair")
COMPOUND_NAMES = ["ryan air", "time air", "sun turk", "sky travel", "finn air", "vista jet", "wizz air", "easy jet",
                  "euro wings", "okay jet", "top swiss", "edel weiss", "jet star", "air baltic"]
# Upper-case tokens that are spoken as words, not spelled letter by letter
ACRONYMS = {"QNH", "QFE", "ILS", "VOR", "DME", "ATIS", "VFR", "IFR", "ATC", "RNAV", "NDB", "TCAS",
            "GPS", "KLM", "SAS", "TAP", "OK", "CAVOK", "RNP", "LOC", "PAPI", "TWR", "APP", "ETA", "UTC"}


def strip_markup(text):
    """Human transcript -> plain words: drop entity tags, non-English spans and noise markers."""
    text = ENTITY_RE.sub(lambda m: m.group(2), text)
    text = re.sub(r"\[/?#\w+\]", " ", text)                                    # stray / broken entity tags
    text = re.sub(r"\[ne\b[^\]]*\].*?\[/ne\]", " ", text, flags=re.I | re.S)    # non-English speech
    text = re.sub(r"\[[^\]]*\]", " ", text)                                    # [hes] [noise] [spk] [XT] ...
    text = re.sub(r"\([^)]*\)", " ", text)
    return " ".join(text.replace("]", " ").split())


def _number_words(num):
    """'121.75' -> 'one two one decimal seven five', '2000' -> 'two thousand', '100' -> 'one hundred'."""
    if "." in num or "," in num and len(num.split(",")[-1]) != 3:
        left, right = re.split(r"[.,]", num, maxsplit=1)
        return f"{_number_words(left)} decimal {' '.join(DIGITS[int(d)] for d in right if d.isdigit())}".strip()
    num = num.replace(",", "")
    spell = lambda s: " ".join(DIGITS[int(d)] for d in s)
    if len(num) in (4, 5) and num.endswith("000"):
        return f"{spell(num[:-3])} thousand"
    if len(num) in (4, 5) and num.endswith("00"):
        return f"{spell(num[:-3])} thousand {DIGITS[int(num[-3])]} hundred"
    if len(num) == 3 and num.endswith("00"):
        return f"{DIGITS[int(num[0])]} hundred"
    return spell(num)


def _expand_token(tok):
    # FL100, 26R, 8HT, KLM73W, HB-JVA: split letters from digits
    if re.fullmatch(r"FL\d+", tok, re.I):
        return "flight level " + _number_words(tok[2:])
    if m := re.fullmatch(r"(\d{2})([LRC])", tok):                     # runway 16R
        return f"{_number_words(m.group(1))} {dict(L='left', R='right', C='center')[m.group(2)]}"
    if re.search(r"\d", tok) and re.search(r"[A-Za-z]", tok):
        parts = re.findall(r"\d+(?:[.,]\d+)?|[A-Za-z]+", tok)
        return " ".join(_expand_token(p) if p[0].isdigit() or p.upper() in ACRONYMS or len(p) > 3 or p.islower()
                        else " ".join(NATO[c.lower()] for c in p) for p in parts)
    if re.fullmatch(r"\d+(?:[.,]\d+)*", tok):
        return _number_words(tok)
    # all-caps letter groups like "JVA" are spelled callsign letters
    if tok.isupper() and tok.isalpha() and 1 < len(tok) <= 5 and tok not in ACRONYMS:
        return " ".join(NATO[c.lower()] for c in tok)
    return tok


def normalize(text):
    """Canonical lower-case word string used for WER on both references and hypotheses."""
    text = strip_markup(text)
    text = text.replace("°", " degrees ").replace("%", " percent ")
    text = re.sub(r"(?<=\d)\.(?!\d)", " ", text)                  # sentence-final dots after numbers
    tokens = []
    for tok in re.split(r"[\s/\-–—]+", text):
        tok = tok.strip(".,;:!?\"'`()")
        if tok:
            tokens.extend(_expand_token(tok).split())
    words = []
    for w in " ".join(tokens).lower().replace("'", "").replace("_", " ").split():
        w = re.sub(r"[^a-z0-9]", "", w)
        if w:
            words.extend(TEENS.get(SPELLING.get(w, w), SPELLING.get(w, w)).split())
    joined = " " + " ".join(words) + " "
    for name in COMPOUND_NAMES:
        joined = joined.replace(f" {name} ", f" {name.replace(' ', '')} ")
    words = joined.split()
    # tens: "twenty three" -> "two three", a bare "twenty" -> "two zero"
    out = []
    for i, w in enumerate(words):
        if w in TENS:
            nxt = words[i + 1] if i + 1 < len(words) else ""
            out.append(TENS[w])
            if nxt not in DIGITS[1:]:
                out.append("zero")
        else:
            out.append(w)
    return " ".join(out)
