import json
import math
import os
import re
import torch
from .tag_parser import extract_prompt_from_full_text, parse_span_masks


def extract_rating(text):
    m = re.search(r"<rate>\s*([-+]?\d+\.?\d*)\s*</rate>", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def check_format_intact(text, cfg):
    """Check all eight tags appear in correct order using string-level search."""
    tags = cfg["tags"]
    pos = -1
    for tag in tags:
        idx = text.find(tag, pos + 1)
        if idx == -1:
            return False
        pos = idx
    return True


def _process_generated(generated_text, record, cfg, tokenizer, maes, rmse_sq,
                        format_intact_list, output_token_counts, span_lengths,
                        out_fh, idx):
    n_toks = len(tokenizer.encode(generated_text, add_special_tokens=False))
    output_token_counts.append(n_toks)
    pred = extract_rating(generated_text)
    gt = float(record["gt"])
    if pred is not None:
        maes.append(abs(pred - gt))
        rmse_sq.append((pred - gt) ** 2)
    fmt_ok = check_format_intact(generated_text, cfg)
    format_intact_list.append(1 if fmt_ok else 0)
    gen_enc = tokenizer(generated_text, add_special_tokens=False, return_offsets_mapping=True)
    masks = parse_span_masks(gen_enc["input_ids"], tokenizer, cfg,
                             full_text=generated_text,
                             offset_mapping=gen_enc["offset_mapping"])
    if masks["valid"]:
        for span in ("user", "item", "match"):
            span_lengths[span].append(len(masks["span_masks"][span]))
    if out_fh is not None:
        row = {
            "idx": idx,
            "gt": gt,
            "pred": pred,
            "generated_tokens": n_toks,
            "format_intact": fmt_ok,
            "generated_text": generated_text,
        }
        out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_fh.flush()


def evaluate_model(model, tokenizer, eval_records, cfg, max_new_tokens=512,
                   batch_size=1, output_path=None):
    """
    output_path: if given, stream per-example JSONL (generated_text, pred, gt,
                 generated_tokens, format_intact) to disk as each example finishes.
    """
    device = cfg["device"]

    maes = []
    rmse_sq = []
    format_intact_list = []
    output_token_counts = []
    span_lengths = {"user": [], "item": [], "match": []}

    orig_padding_side = tokenizer.padding_side

    out_fh = None
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out_fh = open(output_path, "w", encoding="utf-8")

    try:
        if batch_size == 1:
            for idx, record in enumerate(eval_records):
                prompt_text = extract_prompt_from_full_text(record["full_text"], tokenizer, cfg)
                prompt_enc = tokenizer(
                    prompt_text, add_special_tokens=False, return_tensors="pt"
                ).to(device)
                prompt_len = prompt_enc["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **prompt_enc, max_new_tokens=max_new_tokens, do_sample=False,
                    )
                generated_ids = out[0, prompt_len:].tolist()
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                _process_generated(generated_text, record, cfg, tokenizer,
                                   maes, rmse_sq, format_intact_list,
                                   output_token_counts, span_lengths,
                                   out_fh, idx)
        else:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            global_idx = 0
            for i in range(0, len(eval_records), batch_size):
                batch_records = eval_records[i: i + batch_size]
                prompts = [
                    extract_prompt_from_full_text(r["full_text"], tokenizer, cfg)
                    for r in batch_records
                ]
                enc = tokenizer(
                    prompts, add_special_tokens=False, return_tensors="pt",
                    padding=True, truncation=False,
                ).to(device)
                prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    )

                for j, record in enumerate(batch_records):
                    seq_len = out.shape[1]
                    prompt_len = int(prompt_lens[j])
                    pad_len = seq_len - max_new_tokens - prompt_len
                    start = pad_len + prompt_len
                    generated_ids = out[j, start:].tolist()
                    eos_id = tokenizer.eos_token_id
                    if eos_id is not None:
                        try:
                            cut = generated_ids.index(eos_id)
                            generated_ids = generated_ids[: cut + 1]
                        except ValueError:
                            pass
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                    _process_generated(generated_text, record, cfg, tokenizer,
                                       maes, rmse_sq, format_intact_list,
                                       output_token_counts, span_lengths,
                                       out_fh, global_idx)
                    global_idx += 1

            tokenizer.padding_side = orig_padding_side
    finally:
        if out_fh is not None:
            out_fh.close()

    num_valid = len(maes)
    num_invalid = len(eval_records) - num_valid

    mae = sum(maes) / len(maes) if maes else float("nan")
    rmse = math.sqrt(sum(rmse_sq) / len(rmse_sq)) if rmse_sq else float("nan")
    fir = sum(format_intact_list) / len(format_intact_list) if format_intact_list else 0.0
    mean_out = sum(output_token_counts) / len(output_token_counts) if output_token_counts else 0.0

    def mean_or_nan(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "format_intact_rate": fir,
        "mean_output_tokens": mean_out,
        "mean_user_tokens": mean_or_nan(span_lengths["user"]),
        "mean_item_tokens": mean_or_nan(span_lengths["item"]),
        "mean_match_tokens": mean_or_nan(span_lengths["match"]),
        "num_valid": num_valid,
        "num_invalid": num_invalid,
    }
