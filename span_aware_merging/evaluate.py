"""Evaluate a merged checkpoint on a test set."""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config
from utils.metrics import evaluate_model
from utils.tag_parser import batch_parse_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--override_json", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.override_json)
    os.makedirs(cfg["eval_dir"], exist_ok=True)

    dtype = getattr(torch, cfg["dtype_str"])
    print(f"Loading model from {args.checkpoint_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, torch_dtype=dtype, device_map=cfg["device"]
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    examples = []
    with open(args.test_data) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for field in ("id", "full_text", "gt"):
                if field not in rec:
                    raise ValueError(f"Missing '{field}' in {args.test_data} line {lineno}")
            examples.append(rec)

    print(f"Loaded {len(examples)} test examples")
    eval_records = batch_parse_masks(examples, tokenizer, cfg)

    results = evaluate_model(model, tokenizer, eval_records, cfg)
    results["checkpoint_dir"] = args.checkpoint_dir
    results["test_data"] = args.test_data
    results["num_test_examples"] = len(examples)
    results["config_snapshot"] = cfg

    out_path = os.path.join(cfg["eval_dir"], f"{args.output_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'Metric':<25} {'Value':>12}")
    print("-" * 38)
    for key in ("mae", "rmse", "format_intact_rate", "mean_output_tokens",
                "mean_user_tokens", "mean_item_tokens", "mean_match_tokens"):
        val = results[key]
        print(f"{key:<25} {val:>12.4f}")


if __name__ == "__main__":
    main()
