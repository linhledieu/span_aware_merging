"""Step 1: Parse and save span masks from calibration data. Requires no GPU."""
import argparse
import json
import os
import pickle
import random

from config import load_config
from utils.tag_parser import batch_parse_masks
from transformers import AutoTokenizer


def load_jsonl(path):
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for field in ("id", "full_text", "gt"):
                if field not in rec:
                    raise ValueError(
                        f"Missing required field '{field}' in {path} at line {lineno}: {rec}"
                    )
            records.append(rec)
    return records


def print_span_summary(label, records):
    valid = [r for r in records if r["valid"]]
    print(f"\n=== {label} Span Summary (valid={len(valid)}/{len(records)}) ===")
    print(f"{'Span':<18} {'Count':>7} {'Mean':>8} {'Min':>6} {'Max':>6}")
    for span in ("user", "item", "match", "tag", "rate"):
        lengths = [len(r["span_masks"][span]) for r in valid if len(r["span_masks"][span]) > 0]
        if lengths:
            print(
                f"{span:<18} {len(lengths):>7} {sum(lengths)/len(lengths):>8.1f}"
                f" {min(lengths):>6} {max(lengths):>6}"
            )
        else:
            print(f"{span:<18} {'0':>7}")
    # Extra rows for instruction-level masks
    for mask_name, getter in (
        ("instruction_mask", lambda r: r["instruction_mask"]),
        ("user_history_mask", lambda r: r["user_history_mask"]),
    ):
        lengths = [len(getter(r)) for r in valid if len(getter(r)) > 0]
        if lengths:
            print(
                f"{mask_name:<18} {len(lengths):>7} {sum(lengths)/len(lengths):>8.1f}"
                f" {min(lengths):>6} {max(lengths):>6}"
            )
        else:
            print(f"{mask_name:<18} {'0':>7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.override_json)

    os.makedirs(cfg["masks_dir"], exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_r_path"])

    dr_examples = load_jsonl(cfg["dr_path"])
    di_examples = load_jsonl(cfg["di_path"])

    print(f"\n=== Processing D_R ({len(dr_examples)} examples) ===")
    dr_records = batch_parse_masks(dr_examples, tokenizer, cfg)

    print(f"\n=== Processing D_I ({len(di_examples)} examples) ===")
    di_records = batch_parse_masks(di_examples, tokenizer, cfg)

    assert len(dr_records) == len(dr_examples), (
        f"D_R record/line count mismatch: {len(dr_records)} != {len(dr_examples)}"
    )
    assert len(di_records) == len(di_examples), (
        f"D_I record/line count mismatch: {len(di_records)} != {len(di_examples)}"
    )

    dr_out = os.path.join(cfg["masks_dir"], "dr_masks.pkl")
    di_out = os.path.join(cfg["masks_dir"], "di_masks.pkl")
    with open(dr_out, "wb") as f:
        pickle.dump(dr_records, f)
    with open(di_out, "wb") as f:
        pickle.dump(di_records, f)

    print(f"\nSaved {len(dr_records)} D_R records → {dr_out}")
    print(f"Saved {len(di_records)} D_I records → {di_out}")

    # Sanity-check: assert user_history_mask is non-empty in a random sample
    def _check_user_history_masks(records, label):
        valid = [r for r in records if r["valid"]]
        if not valid:
            return
        sample = random.sample(valid, min(5, len(valid)))
        for r in sample:
            if not r.get("user_history_mask"):
                print(
                    f"WARNING [{label}] record id={r['id']} has empty user_history_mask.\n"
                    f"  instruction_mask length: {len(r['instruction_mask'])}\n"
                    f"  full_text prefix (first 500 chars): {r['full_text'][:500]!r}"
                )

    _check_user_history_masks(dr_records, "D_R")
    _check_user_history_masks(di_records, "D_I")

    print_span_summary("D_R", dr_records)
    print_span_summary("D_I", di_records)


if __name__ == "__main__":
    main()
