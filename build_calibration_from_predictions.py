#!/usr/bin/env python3
"""
Build instruction_calibration_set_from_reason_correct_direct_fail.jsonl
from a unified_benchmark predictions.jsonl or reason_correct_direct_fail JSON file.

Supports two input formats:
  1. predictions.jsonl  — output of unified_benchmark eval
     Fields: prompt_long_native, raw_output_long_native, parse_ok, abs_error, gt
  2. reason_correct_direct_fail JSON — richer format with prompt/response/meta
     Fields: prompt, response, reason, instruction_id_list, ...

Usage:
  python build_calibration_from_predictions.py \
      --input  /path/to/predictions.jsonl \
      --output /home/uqlinh/RAIN-Merging/data/yelp/instruction_calibration_set_from_reason_correct_direct_fail.jsonl

  python build_calibration_from_predictions.py \
      --input  /home/uqlinh/RAIN-Merging/data/yelp/reason_correct_direct_fail_stage1.json \
      --output /home/uqlinh/RAIN-Merging/data/yelp/instruction_calibration_set_from_reason_correct_direct_fail.jsonl
"""

import argparse
import json
import re
from pathlib import Path


def _strip_chat_template(text: str) -> str:
    """Strip <|im_start|>...<|im_end|> chat template wrappers, return bare prompt."""
    text = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_start\|>user\s*", "", text)
    text = re.sub(r"\s*<\|im_end\|>\s*<\|im_start\|>user\s*", "\n", text)
    text = re.sub(r"\s*<\|im_end\|>\s*<\|im_start\|>assistant\s*$", "", text.rstrip())
    text = re.sub(r"\s*<\|im_end\|>\s*$", "", text.rstrip())
    return text.strip()


def _load(path: Path):
    text = path.read_text()
    # Try JSON array first
    stripped = text.strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    # JSONL
    return [json.loads(line) for line in stripped.splitlines() if line.strip()]


def _from_predictions(rows: list) -> list:
    """Convert unified_benchmark predictions.jsonl → calibration format."""
    out = []
    skipped = 0
    for row in rows:
        if not row.get("parse_ok", False):
            skipped += 1
            continue
        gt = row.get("gt")
        err = row.get("abs_error", float("inf"))
        threshold = 1.0 if gt in (1.0, 2.0, 3.0) else 0.0
        if err > threshold:
            skipped += 1
            continue

        prompt_raw = row["prompt_long_native"]
        response = row["raw_output_long_native"]
        prompt = _strip_chat_template(prompt_raw)

        out.append({
            "id": str(row.get("sample_id", row.get("id", len(out)))),
            "prompt": prompt,
            "response": response,
            "reason": row.get("reasoning", ""),
            "instruction_id_list": ["task:predict_rating", "format:structured_output"],
            "instruction_list": [
                "Predict the final rating for the target item",
                "Follow the required tagged output structure",
            ],
            "related_prompt_list": [],
            "unrelated_response_spans": [],
            "verdict": {
                "task:predict_rating": True,
                "format:structured_output": True,
            },
            "meta": {
                "source": "reason_correct_direct_fail",
                "assigned_regime": "reason_correct_direct_fail",
                "pred_rating": row.get("pred_rating"),
                "gt": gt,
                "parse_ok": True,
            },
        })
    print(f"predictions format: kept {len(out)}, skipped {skipped}")
    return out


def _from_predictions_prefiltered(rows: list) -> list:
    """Convert pre-filtered stage1 JSON (prompt_long_native, no abs_error) → calibration format."""
    out = []
    skipped = 0
    for row in rows:
        if not row.get("parse_ok", False):
            skipped += 1
            continue
        prompt = _strip_chat_template(row["prompt_long_native"])
        response = row["raw_output_long_native"]
        out.append({
            "id": str(row.get("sample_id", row.get("id", len(out)))),
            "prompt": prompt,
            "response": response,
            "reason": row.get("reasoning", ""),
            "instruction_id_list": ["task:predict_rating", "format:structured_output"],
            "instruction_list": [
                "Predict the final rating for the target item",
                "Follow the required tagged output structure",
            ],
            "related_prompt_list": [],
            "unrelated_response_spans": [],
            "verdict": {
                "task:predict_rating": True,
                "format:structured_output": True,
            },
            "meta": {
                "source": row.get("source", "reason_correct_direct_fail"),
                "assigned_regime": row.get("assigned_regime", "reason_correct_direct_fail"),
                "reason_for_assignment": row.get("reason_for_assignment", ""),
                "pred_rating": row.get("pred_rating"),
                "gt": row.get("gt", row.get("ground_truth")),
                "parse_ok": True,
            },
        })
    print(f"prefiltered stage1 format: kept {len(out)}, skipped {skipped}")
    return out


def _from_rich(rows: list) -> list:
    """Pass through rich-format records, normalising field names."""
    out = []
    for row in rows:
        # Already in the right format — just ensure required keys exist
        record = {
            "id": str(row.get("id", row.get("sample_id", len(out)))),
            "prompt": row.get("prompt", ""),
            "response": row.get("response", ""),
            "reason": row.get("reason", ""),
            "instruction_id_list": row.get("instruction_id_list", []),
            "instruction_list": row.get("instruction_list", []),
            "related_prompt_list": row.get("related_prompt_list", []),
            "unrelated_response_spans": row.get("unrelated_response_spans", []),
            "verdict": row.get("verdict", {}),
            "meta": row.get("meta", {}),
        }
        out.append(record)
    print(f"rich format: kept {len(out)} records")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="predictions.jsonl or reason_correct_direct_fail JSON")
    p.add_argument("--output", required=True, help="Output calibration JSONL path")
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = _load(src)
    print(f"Loaded {len(rows)} records from {src}")

    # Detect format by inspecting first record.
    # predictions.jsonl has prompt_long_native + abs_error (unfiltered eval output).
    # stage1 JSON has prompt_long_native but no abs_error (already pre-filtered).
    # rich format has prompt/response/meta etc.
    first = rows[0]
    if "prompt_long_native" in first and "abs_error" in first:
        out = _from_predictions(rows)
    elif "prompt_long_native" in first:
        # Pre-filtered stage1 format — treat every parse_ok record as kept
        out = _from_predictions_prefiltered(rows)
    else:
        out = _from_rich(rows)

    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    print(f"Wrote {len(out)} records → {dst}")


if __name__ == "__main__":
    main()
