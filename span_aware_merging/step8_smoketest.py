"""Step 8: Smoke test the merged checkpoint for format integrity, then run full eval."""
import argparse
import csv
import json
import math
import os
import pickle
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config
from utils.metrics import evaluate_model, extract_rating, check_format_intact
from utils.tag_parser import extract_prompt_from_full_text, parse_span_masks

UNIFIED_BENCHMARK_DIR = "/home/uqlinh/ReasonMerge/EXP3RT/theory_test"


def _mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")


def _rmse(errs_sq):
    return math.sqrt(sum(errs_sq) / len(errs_sq)) if errs_sq else float("nan")


def load_prepared_examples(data_path, regime):
    """Load parquet/jsonl and convert to {"prompt", "gt"} records via unified_benchmark."""
    if UNIFIED_BENCHMARK_DIR not in sys.path:
        sys.path.insert(0, UNIFIED_BENCHMARK_DIR)
    from unified_benchmark import load_records, prepare_examples
    rows = load_records(data_path)
    examples = prepare_examples(rows, regime=regime, converted_input=False)
    return [ex for ex in examples if ex.get("gt") is not None]


def run_full_eval(model, tokenizer, records, cfg, output_dir, model_path, model_name,
                  max_new_tokens=512, regime=None):
    """
    Run per-example eval and save unified_benchmark-style outputs:
      predictions.jsonl, failure_cases.jsonl, metrics.json, run_meta.json

    Records may be either:
      - D_I mask records: {"full_text": ..., "gt": ...}
      - Prepared examples:  {"prompt": ..., "gt": ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    failure_path = os.path.join(output_dir, "failure_cases.jsonl")
    run_meta_path = os.path.join(output_dir, "run_meta.json")

    device = cfg["device"]
    t0 = time.time()

    predictions = []
    failures = []

    with open(pred_path, "w", encoding="utf-8") as pred_fh, \
         open(failure_path, "w", encoding="utf-8") as fail_fh:

        for i, record in enumerate(records):
            ex_t0 = time.time()

            # Support both record formats
            if "prompt" in record:
                prompt_text = record["prompt"]
            else:
                prompt_text = extract_prompt_from_full_text(record["full_text"], tokenizer, cfg)

            prompt_enc = tokenizer(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            ).to(device)
            prompt_len = prompt_enc["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **prompt_enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            generated_ids = out[0, prompt_len:].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            latency = time.time() - ex_t0

            pred_rating = extract_rating(generated_text)
            gt_rating = float(record["gt"])
            parse_ok = pred_rating is not None
            exact_match = parse_ok and abs(pred_rating - gt_rating) < 1e-6
            within_0_5 = parse_ok and abs(pred_rating - gt_rating) <= 0.5
            format_ok = check_format_intact(generated_text, cfg)

            # Span lengths (only meaningful for span-aware outputs)
            span_lens = {}
            if not regime or regime not in {"native_reczero", "native_tallrec"}:
                gen_enc = tokenizer(
                    generated_text, add_special_tokens=False, return_offsets_mapping=True
                )
                masks = parse_span_masks(
                    gen_enc["input_ids"], tokenizer, cfg,
                    full_text=generated_text,
                    offset_mapping=gen_enc["offset_mapping"],
                )
                if masks["valid"]:
                    for span in ("user", "item", "match"):
                        span_lens[f"{span}_tokens"] = len(masks["span_masks"][span])

            row = {
                "idx": i,
                "sample_id": record.get("sample_id", record.get("id", i)),
                "prompt": prompt_text,
                "generated_text": generated_text,
                "generated_tokens": len(generated_ids),
                "prompt_tokens": prompt_len,
                "gt": gt_rating,
                "pred": pred_rating,
                "parse_ok": parse_ok,
                "exact_match": exact_match,
                "within_0_5": within_0_5,
                "format_intact": format_ok,
                "latency_sec": latency,
                **span_lens,
            }

            predictions.append(row)
            pred_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            if not parse_ok or (parse_ok and abs(pred_rating - gt_rating) > 1.0):
                failures.append(row)
                fail_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0:
                fir_so_far = sum(r["format_intact"] for r in predictions) / len(predictions)
                print(f"  [{i+1}/{len(records)}] format_intact={fir_so_far:.3f} "
                      f"pred={pred_rating} gt={gt_rating}")

    total_sec = time.time() - t0

    n = len(predictions)
    n_parse = sum(1 for r in predictions if r["parse_ok"])
    n_exact = sum(1 for r in predictions if r["exact_match"])
    n_within = sum(1 for r in predictions if r["within_0_5"])
    n_format = sum(1 for r in predictions if r["format_intact"])
    pred_values = [r["pred"] for r in predictions if r["parse_ok"]]
    gt_values = [r["gt"] for r in predictions if r["parse_ok"]]
    mae_val = _mean([abs(p - g) for p, g in zip(pred_values, gt_values)])
    rmse_val = _rmse([(p - g) ** 2 for p, g in zip(pred_values, gt_values)])
    gen_toks = [r["generated_tokens"] for r in predictions]
    latencies = [r["latency_sec"] for r in predictions]

    run_meta = {
        "model_path": model_path,
        "model_name": model_name,
        "regime": regime,
        "num_eval_examples": n,
        "max_new_tokens": max_new_tokens,
        "command": " ".join(sys.argv),
    }

    metrics = {
        "model_path": model_path,
        "model_name": model_name,
        "regime": regime,
        "num_examples": n,
        "num_parse_ok": n_parse,
        "parse_rate": n_parse / n if n else 0.0,
        "format_intact_count": n_format,
        "format_intact_rate": n_format / n if n else 0.0,
        "exact_match_count": n_exact,
        "exact_match_rate_over_total": n_exact / n if n else 0.0,
        "exact_match_rate_over_comparable": (n_exact / n_parse) if n_parse else None,
        "within_0_5_count": n_within,
        "within_0_5_rate_over_total": n_within / n if n else 0.0,
        "within_0_5_rate_over_comparable": (n_within / n_parse) if n_parse else None,
        "mae": mae_val,
        "rmse": rmse_val,
        "mean_generated_tokens": _mean(gen_toks),
        "mean_latency_sec": _mean(latencies),
        "total_seconds": total_sec,
        "examples_per_second": n / total_sec if total_sec > 0 else None,
        "failure_examples": len(failures),
        "output_dir": output_dir,
        "run_meta": run_meta,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in metrics.items() if k != "run_meta"}, indent=2, ensure_ascii=False))
    print(f"[done] metrics={metrics_path}")
    print(f"[done] predictions={pred_path}")
    print(f"[done] failures={failure_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--override_json", type=str, default=None)
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Test set for full eval (.parquet or .jsonl). "
                             "If omitted, uses D_I records from masks_dir.")
    parser.add_argument("--eval_regime", type=str, default="native_reczero",
                        help="Regime passed to unified_benchmark prepare_examples when loading eval_data.")
    parser.add_argument("--eval_output_dir", type=str, default=None,
                        help="Where to write full eval outputs. "
                             "Defaults to <eval_dir>/full_eval.")
    parser.add_argument("--eval_max_new_tokens", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for smoke test generation (requires left-padding).")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Display name for eval metrics. Defaults to checkpoint dir basename.")
    args = parser.parse_args()

    cfg = load_config(args.override_json)
    dtype = getattr(torch, cfg["dtype_str"])
    model_name = args.model_name or os.path.basename(args.checkpoint_dir.rstrip("/"))
    full_eval_dir = args.eval_output_dir or os.path.join(cfg["eval_dir"], "full_eval")

    print(f"Loading model from {args.checkpoint_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, torch_dtype=dtype, device_map=cfg["device"]
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # ── Smoke test ────────────────────────────────────────────────────────────
    with open(os.path.join(cfg["masks_dir"], "di_masks.pkl"), "rb") as f:
        di_records = pickle.load(f)

    valid_records = [r for r in di_records if r["valid"]]
    assert len(valid_records) > 0, "No valid D_I records found"

    test_record = valid_records[0]
    prompt = extract_prompt_from_full_text(test_record["full_text"], tokenizer, cfg)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    test_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=cfg["device"])
    with torch.no_grad():
        test_out = model.generate(test_tensor, max_new_tokens=16, do_sample=False)
    assert test_out.shape[1] > len(prompt_ids), "Model produced no tokens on sanity check"
    print("Sanity check passed.")

    smoke_records = valid_records[:200]
    print(f"Running smoke test on {len(smoke_records)} examples...")
    os.makedirs(cfg["eval_dir"], exist_ok=True)
    smoke_outputs_path = os.path.join(cfg["eval_dir"], "smoketest_outputs.jsonl")
    results = evaluate_model(model, tokenizer, smoke_records, cfg,
                             max_new_tokens=args.eval_max_new_tokens,
                             batch_size=args.batch_size,
                             output_path=smoke_outputs_path)
    print(f"Per-example outputs saved to {smoke_outputs_path}")

    fir = results["format_intact_rate"]
    num_ok = int(round(fir * len(smoke_records)))

    lines = []
    lines.append(f"format_intact_rate:  {fir:.4f} ({num_ok}/{len(smoke_records)})")
    lines.append(f"mae:                 {results['mae']:.4f}")
    lines.append(f"rmse:                {results['rmse']:.4f}")
    lines.append(f"mean_output_tokens:  {results['mean_output_tokens']:.1f}")

    if fir < 0.99:
        lines.append(f"\nFormat failure rate: {fir:.1%}. Increase delta_fmt by 0.1 in config and rerun from step5_floors.py.")

        alpha_csv = os.path.join(cfg["coefficients_dir"], "alpha_summary.csv")
        if os.path.exists(alpha_csv):
            with open(alpha_csv) as f:
                rows = list(csv.DictReader(f))

            fisher_loaded = {}
            for proj_type in ("q", "k", "v", "o"):
                fpath = os.path.join(cfg["fisher_dir"], f"fisher_norm_tag_{proj_type}.pt")
                if os.path.exists(fpath):
                    fisher_loaded[proj_type] = torch.load(fpath, map_location="cpu")

            candidates = []
            for row in rows:
                proj = row["proj_type"]
                l = int(row["layer"])
                h = int(row["head_index"])
                final = float(row["final_coefficient"])
                floor = float(row["floor_val"])
                if proj in fisher_loaded:
                    tag_f = fisher_loaded[proj][l, h].item()
                    if tag_f > 0.5:
                        candidates.append((final, proj, l, h, floor, tag_f))

            candidates.sort(key=lambda x: x[0])
            top10 = candidates[:10]
            if top10:
                lines.append("\nTop 10 low-coefficient high-tag-Fisher heads:")
                lines.append(f"{'proj':<6} {'layer':>6} {'head':>6} {'floor':>8} {'final':>10} {'tag_fisher':>11}")
                for final, proj, l, h, floor, tag_f in top10:
                    lines.append(f"{proj:<6} {l:>6} {h:>6} {floor:>8.4f} {final:>10.4f} {tag_f:>11.4f}")
    else:
        lines.append(f"\nFormat check passed: {fir:.1%} intact")
        lines.append(f"mean_user_tokens:    {results['mean_user_tokens']:.1f}")
        lines.append(f"mean_item_tokens:    {results['mean_item_tokens']:.1f}")
        lines.append(f"mean_match_tokens:   {results['mean_match_tokens']:.1f}")

    smoke_output = "\n".join(lines)
    print(smoke_output)

    smoke_path = os.path.join(cfg["eval_dir"], "smoketest_results.txt")
    with open(smoke_path, "w") as f:
        f.write(smoke_output + "\n")
    print(f"\nSmoke test results saved to {smoke_path}")

    # ── Full eval ─────────────────────────────────────────────────────────────
    if args.eval_data:
        print(f"\nLoading eval data from {args.eval_data} (regime={args.eval_regime})...")
        eval_records = load_prepared_examples(args.eval_data, args.eval_regime)
        print(f"Running full eval on {len(eval_records)} examples...")
        regime = args.eval_regime
    else:
        eval_records = valid_records
        regime = None
        print(f"\nRunning full eval on {len(eval_records)} D_I examples...")

    run_full_eval(
        model=model,
        tokenizer=tokenizer,
        records=eval_records,
        cfg=cfg,
        output_dir=full_eval_dir,
        model_path=args.checkpoint_dir,
        model_name=model_name,
        max_new_tokens=args.eval_max_new_tokens,
        regime=regime,
    )

    sys.exit(1 if fir < 0.99 else 0)


if __name__ == "__main__":
    main()
