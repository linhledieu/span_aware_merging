#!/usr/bin/env python3
"""
Convert unified_benchmark predictions.jsonl → nullspace_projection_compute.py
calibration format.

Only fields used by nullspace_projection_compute.py are written:
  - prompt_long_native   (from `prompt`)
  - raw_output_long_native (from `prediction_text`)

Usage:
  python convert_predictions_to_calibration.py \
      --input  /data/uqlinh/merged_models/book_fixed/eval_results/book_3b_cal/predictions.jsonl \
      --output /data/uqlinh/merged_models/book_fixed/eval_results/book_3b_cal/calibration.jsonl
"""
import argparse
import json
from pathlib import Path

def convert(src: Path, dst: Path) -> None:
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    out = []
    skipped = 0
    for row in rows:
        if not row.get("parse_ok", False):
            skipped += 1
            continue
        gt  = row.get("gt", None)
        err = row.get("abs_error", float("inf"))
        # low ratings get a relaxed threshold
        threshold = 1.0 if gt in (1.0, 2.0, 3.0) else 0.0
        if err > threshold:
            skipped += 1
            continue
        out.append({
            "prompt_long_native":    row["prompt"],
            "raw_output_long_native": row["prediction_text"],
        })
    dst.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"Wrote {len(out)} records (skipped {skipped}) → {dst}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    convert(Path(args.input), Path(args.output))
