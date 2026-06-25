# Redrob Intelligent Candidate Discovery — baseline solver

This repo contains a fast, CPU-only ranking baseline for the Redrob hackathon.

## What it does

- Reads `candidates.jsonl` or `candidates.jsonl.gz`
- Scores each candidate using:
  - JD-fit keywords for embeddings, retrieval, ranking, vector search, and evaluation
  - experience fit centered on the 5–9 year band
  - location and relocation fit for Pune/Noida / Tier-1 India
  - availability and engagement signals from `redrob_signals`
  - consistency checks to reduce keyword-stuffing and obvious mismatches
- Writes a `submission.csv` with exactly 100 ranked candidates

## Run

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

The same command works with `candidates.jsonl.gz`.

## Validate

```bash
python validate_submission.py ./submission.csv
```

## Files

- `rank.py` — the ranking pipeline
- `validate_submission.py` — official format validator from the bundle
- `submission_metadata.yaml` — metadata template to mirror portal fields
- `requirements.txt` — minimal dependencies

## Notes

- The ranking step is deterministic and uses no network.
- The output CSV includes a short 1–2 sentence reasoning for each ranked candidate.
- The score is monotonic by rank and tied scores are broken by candidate_id ascending.
