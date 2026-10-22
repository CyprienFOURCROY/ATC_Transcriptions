# ATC_Transcriptions

Speech recognition experiments on the **ATCO2 1h test set**: 560 air-traffic-control radio recordings (~66 min, 7 airports) with human transcripts.

## Layout

```
data/
  raw/
    ATCO2-ASRdataset-v1_beta/   original dataset: DATA/*.wav .xml .cnet .info, Kaldi/
    xml_tagged/                 human transcripts with entity tags on all 560 files (use these)
    *.tgz, *.tar.gz             original archives
  processed/
    references.jsonl            cleaned + normalised references (scripts/prepare_references.py)
models/
  whisper-turbo/                faster-whisper large-v3-turbo (CTranslate2)
scripts/
  common.py                     paths, loading, ATC text normalisation
  prepare_references.py         xml_tagged -> data/processed/references.jsonl
  cnet_baseline.py              dataset's own ASR (.cnet best path) -> outputs/transcripts/atco2-cnet/
  transcribe.py                 Whisper on the .wav -> outputs/transcripts/<system>/
  evaluate.py                   WER + callsign accuracy -> outputs/eval/
outputs/
  transcripts/<system>/         one JSON per recording + hyps.jsonl
  eval/                         summary.csv, <system>_per_file.csv
viewer/                         dashboard: python3 viewer/app.py
```

## Pipeline

```bash
pip install faster-whisper jiwer
hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo --local-dir models/whisper-turbo

python3 scripts/prepare_references.py
python3 scripts/cnet_baseline.py
python3 scripts/transcribe.py                                   # -> outputs/transcripts/whisper-turbo
python3 scripts/transcribe.py --prompt atc                      # -> whisper-turbo-atc
python3 scripts/transcribe.py --prompt callsigns                # -> whisper-turbo-callsigns
python3 scripts/evaluate.py
```

`transcribe.py` resumes where it stopped; use `--limit N` for a quick test and `--overwrite` to redo files.

## Normalisation

References spell everything out ("flight level one hundred"), Whisper writes digits ("FL100").
Before scoring, both sides go through `common.normalize`: markup and non-English spans removed,
digits expanded to spoken words, "twenty three" -> "two three", runway "16R" -> "one six right",
spelled letters "HB-JVA" -> "hotel bravo juliet victor alpha", NATO spelling variants unified.
