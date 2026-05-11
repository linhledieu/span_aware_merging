"""Convert raw data files to the schema required by span_aware_merging.

Expected output schema per record (jsonl):
  id        str   unique identifier
  full_text str   prompt concatenated with model response (contains all 8 tags)
  gt        float ground truth rating

Usage:
  python prepare_data.py --out_dir /path/to/output/data

Input files are read from config (dr_path and di_path before conversion).
Pass the raw paths via --dr_raw and --di_raw.
"""
import argparse
import ast
import json
import os


def convert_dr(in_path, out_path):
    """
    Input: JSON array. Each record has:
      sample_id, prompt_long_native, raw_output_long_native, gt
    The prompt ends with '<|im_start|>assistant' (no trailing newline).
    The response starts with '<analyze user>'.
    full_text = prompt_long_native + "\n" + raw_output_long_native
    """
    with open(in_path) as f:
        records = json.load(f)

    out = []
    for r in records:
        prompt = r["prompt_long_native"]
        response = r["raw_output_long_native"]
        # Ensure single newline separator between prompt and response
        if not prompt.endswith("\n"):
            full_text = prompt + "\n" + response
        else:
            full_text = prompt + response
        out.append({
            "id": str(r["sample_id"]),
            "full_text": full_text,
            "gt": float(r["gt"]),
        })

    with open(out_path, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    print(f"D_R: wrote {len(out)} records to {out_path}")
    return len(out)


def convert_di(in_path, out_path):
    """
    Input: jsonl. Each record has:
      id, prompt, response, meta (string repr of dict with gt key)
    The prompt does NOT end with the assistant turn marker.
    full_text = prompt + "\n" + response  (tag parser finds last <analyze user>)
    gt is in meta['gt'].
    """
    out = []
    with open(in_path) as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            prompt = r["prompt"]
            response = r["response"]
            # Ensure single newline separator
            if not prompt.endswith("\n"):
                full_text = prompt + "\n" + response
            else:
                full_text = prompt + response

            # Parse gt from meta (stored as string repr of dict)
            meta_raw = r.get("meta", "{}")
            try:
                # meta is a Python dict repr, not JSON
                meta = ast.literal_eval(str(meta_raw))
                gt = float(meta.get("gt", 0.0))
            except Exception:
                gt = 0.0

            out.append({
                "id": str(r["id"]),
                "full_text": full_text,
                "gt": gt,
            })

    with open(out_path, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    print(f"D_I: wrote {len(out)} records to {out_path}")
    return len(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dr_raw",
        default="/home/uqlinh/ReasonMerge/Long-to-Short-via-Model-Merging/RAIN-Merging/data/reason_correct_direct_fail_stage1.json",
    )
    parser.add_argument(
        "--di_raw",
        default="/home/uqlinh/ReasonMerge/Long-to-Short-via-Model-Merging/RAIN-Merging/data/instruction_calibration_set_from_reason_correct_direct_fail.jsonl",
    )
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dr_out = os.path.join(args.out_dir, "dr_calibration.jsonl")
    di_out = os.path.join(args.out_dir, "di_calibration.jsonl")

    n_dr = convert_dr(args.dr_raw, dr_out)
    n_di = convert_di(args.di_raw, di_out)

    print(f"\nDone. Use these paths in your config:")
    print(f'  "dr_path": "{dr_out}"')
    print(f'  "di_path": "{di_out}"')


if __name__ == "__main__":
    main()
