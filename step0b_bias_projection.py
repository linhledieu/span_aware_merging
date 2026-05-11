"""
step0b_bias_projection.py

Takes the existing schema-projected task vector pickle file and applies a second
null-space projection that removes the rating calibration bias direction from every
delta tensor. Produces a new pickle file in the same format that can be passed to
step 7 as a drop-in replacement.
"""

import argparse
import json
import os
import pickle
import random
import sys

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from raw_ckpt_to_projected import (
    build_raw_delta_projected,
    _parse_selected_layers,
    _parse_selected_heads,
)


# ---------------------------------------------------------------------------
# Calibration data loading
# ---------------------------------------------------------------------------

def load_calibration_examples(
    reczero_jsonl_path: str,
    tallrec_jsonl_path: str,
    low_rating_threshold: float,
    high_rating_threshold: float,
    tallrec_high_threshold: float,
):
    def _load_jsonl(path):
        rows = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows[str(obj["id"])] = obj
                except (json.JSONDecodeError, KeyError):
                    continue
        return rows

    reczero_rows = _load_jsonl(reczero_jsonl_path)
    tallrec_rows = _load_jsonl(tallrec_jsonl_path)

    shared_ids = set(reczero_rows) & set(tallrec_rows)
    print(f"RecZero: {len(reczero_rows)} examples, TALLRec: {len(tallrec_rows)} examples, overlap: {len(shared_ids)}")

    raw_examples = []
    correct_low_validation = []

    for eid in sorted(shared_ids, key=lambda x: int(x)):
        rz = reczero_rows[eid]
        tr = tallrec_rows[eid]

        try:
            gt_rating = float(rz["gt"])
        except (KeyError, TypeError, ValueError):
            continue

        try:
            tallrec_pred = float(tr["pred_rating"])
        except (KeyError, TypeError, ValueError):
            continue

        rz_prompt = rz.get("prompt")
        rz_output = rz.get("prediction_text")
        if rz_prompt is None or rz_output is None:
            continue

        # prompt is already chat-templated; concatenate directly with no separator
        full_text = rz_prompt + rz_output

        tallrec_output_text = tr.get("prediction_text")

        example = {
            "gt_rating": gt_rating,
            "tallrec_pred": tallrec_pred,
            "full_text": full_text,
            "tallrec_output_text": tallrec_output_text,
            "prompt_text": rz_prompt,
        }

        if gt_rating <= 3.0 and tallrec_pred <= 3.0:
            correct_low_validation.append(dict(example, bin_label="correct_low_relaxed"))

        # --- fit-group label ---
        if gt_rating <= low_rating_threshold and tallrec_pred >= tallrec_high_threshold:
            bin_label = "low"
        elif gt_rating >= high_rating_threshold and tallrec_pred >= tallrec_high_threshold:
            bin_label = "high"
        else:
            bin_label = "mid"

        raw_examples.append(dict(example, bin_label=bin_label))

    low_examples = [e for e in raw_examples if e["bin_label"] == "low"]
    high_examples = [e for e in raw_examples if e["bin_label"] == "high"]
    mid_examples = [e for e in raw_examples if e["bin_label"] == "mid"]

    raw_d_low = len(low_examples)
    raw_d_high = len(high_examples)
    d_mid = len(mid_examples)

    rng = random.Random(42)
    if raw_d_low > 0 and raw_d_high > raw_d_low:
        high_examples = rng.sample(high_examples, raw_d_low)

    examples = low_examples + high_examples + mid_examples
    d_low = len(low_examples)
    d_high = len(high_examples)

    stats = {
        "joined_examples": len(raw_examples),
        "raw_d_harm": raw_d_low,
        "raw_d_correct_high": raw_d_high,
        "subsampled_d_harm": d_low,
        "subsampled_d_correct_high": d_high,
        "d_mid": d_mid,
        "d_correct_low_relaxed": len(correct_low_validation),
        "subsample_seed": 42,
    }

    print(f"Calibration data: {len(raw_examples)} joined examples")
    print(f"  Raw D_harm={raw_d_low}, raw D_correct_high={raw_d_high}, D_mid={d_mid}")
    print(f"  Subsampled D_harm={d_low}, subsampled D_correct_high={d_high}")
    print(f"  D_correct_low_relaxed probe={len(correct_low_validation)}")
    return examples, correct_low_validation, stats


# ---------------------------------------------------------------------------
# Finding </match> position
# ---------------------------------------------------------------------------

def find_match_end_position(full_text: str, tokenizer):
    char_pos = full_text.rfind("</match>")
    if char_pos == -1:
        return None

    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]

    target_end = char_pos + len("</match>")
    token_idx = None
    for i, (start, end) in enumerate(offsets):
        if start <= char_pos and end >= target_end:
            token_idx = i
            break
        if start >= char_pos:
            token_idx = i
            break

    return token_idx


# ---------------------------------------------------------------------------
# Chunked KV-cache forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def chunked_forward_hidden_state(
    model,
    input_ids,
    attention_mask,
    target_position: int,
    selected_layers: list,
    chunk_size: int = 2048,
    device: str = "cuda:0",
):
    seq_len = input_ids.shape[1]

    if seq_len <= chunk_size:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        result = {
            li: outputs.hidden_states[li + 1][0, target_position].detach().cpu().float()
            for li in selected_layers
        }
        del outputs
        torch.cuda.empty_cache()
        return result

    # chunked path: build KV cache up to the chunk containing target_position
    past_key_values = None
    chunk_start = 0

    while chunk_start <= target_position:
        chunk_end = min(chunk_start + chunk_size, target_position + 1)
        is_final = chunk_end == target_position + 1

        position_ids = torch.arange(chunk_start, chunk_end, dtype=torch.long).unsqueeze(0).to(device)

        if is_final:
            outputs = model(
                input_ids=input_ids[:, chunk_start:chunk_end],
                attention_mask=attention_mask[:, :chunk_end],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                output_hidden_states=True,
            )
            local_pos = target_position - chunk_start
            result = {
                li: outputs.hidden_states[li + 1][0, local_pos].detach().cpu().float()
                for li in selected_layers
            }
            del outputs
            torch.cuda.empty_cache()
            return result
        else:
            outputs = model(
                input_ids=input_ids[:, chunk_start:chunk_end],
                attention_mask=attention_mask[:, :chunk_end],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
            )
            past_key_values = outputs.past_key_values
            del outputs
            torch.cuda.empty_cache()
            chunk_start = chunk_end

    # unreachable, but satisfies type checkers
    return {}


# ---------------------------------------------------------------------------
# Computing bias directions from RecZero
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_bias_directions(
    model,
    tokenizer,
    calibration_examples,
    selected_layers,
    device,
    n_bias_directions: int,
    min_group_size: int,
    batch_size: int,
):
    low_examples = [e for e in calibration_examples if e["bin_label"] == "low"]
    high_examples = [e for e in calibration_examples if e["bin_label"] == "high"]

    # per-layer accumulators: layer_idx -> list of tensors
    layer_vecs_low = {li: [] for li in selected_layers}
    layer_vecs_high = {li: [] for li in selected_layers}

    def process_group(examples, vecs_dict, group_name):
        skipped = 0
        i = 0
        while i < len(examples):
            batch = examples[i:i + batch_size]
            i += batch_size

            # find </match> position for each example; drop examples without one
            texts = []
            positions = []
            for ex in batch:
                pos = find_match_end_position(ex["full_text"], tokenizer)
                if pos is None:
                    skipped += 1
                else:
                    texts.append(ex["full_text"])
                    positions.append(pos)

            if not texts:
                continue

            enc = tokenizer(
                texts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).to(device)

            # drop examples where </match> falls outside the (padded) token range
            valid_mask = [enc["input_ids"].shape[1] > p for p in positions]
            if not any(valid_mask):
                skipped += sum(1 for _ in texts)
                continue

            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )

            for idx, (pos, valid) in enumerate(zip(positions, valid_mask)):
                if not valid:
                    skipped += 1
                    continue
                for li in selected_layers:
                    hs = outputs.hidden_states[li + 1][idx, pos].detach().cpu().float()
                    vecs_dict[li].append(hs)

            del outputs, enc
            torch.cuda.empty_cache()

        if skipped:
            print(f"  {group_name}: skipped {skipped} examples (no </match> token found)")

    print("Computing bias directions from RecZero...")
    print(f"  Processing D_low ({len(low_examples)} examples)...")
    process_group(low_examples, layer_vecs_low, "D_low")
    print(f"  Processing D_high ({len(high_examples)} examples)...")
    process_group(high_examples, layer_vecs_high, "D_high")

    bias_directions = {}

    for li in selected_layers:
        low_vecs = layer_vecs_low[li]
        high_vecs = layer_vecs_high[li]

        n_low = len(low_vecs)
        n_high = len(high_vecs)

        if n_low < min_group_size or n_high < min_group_size:
            print(
                f"  Layer {li}: WARNING — insufficient examples "
                f"(D_low={n_low}, D_high={n_high}, min={min_group_size}). "
                f"Skipping bias projection for this layer."
            )
            continue

        L = torch.stack(low_vecs)   # [n_low, d_model]
        H = torch.stack(high_vecs)  # [n_high, d_model]

        if n_bias_directions == 1:
            mu_low = L.mean(dim=0)
            mu_high = H.mean(dim=0)
            raw_dir = mu_high - mu_low
            norm = raw_dir.norm()
            d_hat = raw_dir / norm.clamp(min=1e-8)

            proj_high = (H @ d_hat).mean().item()
            proj_low = (L @ d_hat).mean().item()
            gap = proj_high - proj_low

            print(
                f"  Layer {li}: D_low={n_low}, D_high={n_high}, "
                f"dir_norm={norm:.4f}, proj_gap={gap:.4f}"
            )
            if gap < 0.05:
                print(f"  Layer {li}: WARNING — projection gap {gap:.4f} < 0.05, direction may be weak.")

            bias_directions[li] = d_hat  # shape [d_model]

        else:
            k = n_bias_directions
            mu_L = L.mean(dim=0, keepdim=True)
            D_mat = H - mu_L  # [n_high, d_model]
            U, S, Vh = torch.linalg.svd(D_mat, full_matrices=False)
            B = Vh[:k].T  # [d_model, k]

            proj_high = (H @ B).norm(dim=1).mean().item()
            proj_low = (L @ B).norm(dim=1).mean().item()
            gap = proj_high - proj_low

            print(
                f"  Layer {li}: D_low={n_low}, D_high={n_high}, "
                f"top-{k} singular values={S[:k].tolist()}, proj_gap={gap:.4f}"
            )
            if gap < 0.05:
                print(f"  Layer {li}: WARNING — projection gap {gap:.4f} < 0.05, direction may be weak.")

            bias_directions[li] = B  # shape [d_model, k]

    return bias_directions


@torch.no_grad()
def compute_group_projection_stats(
    model,
    tokenizer,
    groups,
    bias_directions,
    device,
    batch_size: int,
    text_builder=None,
):
    active_layers = sorted(bias_directions.keys())
    if text_builder is None:
        text_builder = lambda ex, tok: ex["full_text"]

    group_vecs = {
        group_name: {li: [] for li in active_layers}
        for group_name in groups
    }

    for group_name, examples in groups.items():
        skipped = 0
        i = 0
        while i < len(examples):
            batch = examples[i:i + batch_size]
            i += batch_size

            texts = []
            positions = []
            for ex in batch:
                full_text = text_builder(ex, tokenizer)
                pos = find_match_end_position(full_text, tokenizer)
                if pos is None:
                    skipped += 1
                else:
                    texts.append(full_text)
                    positions.append(pos)

            if not texts:
                continue

            enc = tokenizer(
                texts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).to(device)

            valid_mask = [enc["input_ids"].shape[1] > p for p in positions]
            if not any(valid_mask):
                skipped += len(texts)
                continue

            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )

            for idx, (pos, valid) in enumerate(zip(positions, valid_mask)):
                if not valid:
                    skipped += 1
                    continue
                for li in active_layers:
                    hs = outputs.hidden_states[li + 1][idx, pos].detach().cpu().float()
                    group_vecs[group_name][li].append(hs)

            del outputs, enc
            torch.cuda.empty_cache()

        if skipped:
            print(f"  {group_name}: skipped {skipped} examples")

    projection_stats = {}
    for li, direction in bias_directions.items():
        layer_stats = {}
        for group_name in groups:
            vecs = group_vecs[group_name][li]
            if not vecs:
                continue
            X = torch.stack(vecs)
            if direction.dim() == 1:
                proj_mean = (X @ direction).mean().item()
            else:
                proj_mean = (X @ direction).norm(dim=1).mean().item()
            layer_stats[group_name] = {
                "proj_mean": proj_mean,
                "n": len(vecs),
            }
        projection_stats[str(li)] = layer_stats

    return projection_stats


# ---------------------------------------------------------------------------
# Validating with TALLRec
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_direction_with_tallrec(
    model_i,
    tokenizer_i,
    calibration_examples,
    bias_directions,
    device,
    selected_layers,
    output_dir,
    batch_size: int,
):
    print("Validating bias directions with TALLRec...")

    low_examples = [e for e in calibration_examples if e["bin_label"] == "low"]
    high_examples = [e for e in calibration_examples if e["bin_label"] == "high"]

    def build_tallrec_text(ex, tokenizer):
        prompt_text = ex.get("prompt_text", "")
        tallrec_out = ex.get("tallrec_output_text")
        try:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            templated = prompt_text

        if tallrec_out is not None:
            return templated + tallrec_out
        return ex["full_text"]

    active_layers = [li for li in selected_layers if li in bias_directions]
    layer_vecs_low = {li: [] for li in active_layers}
    layer_vecs_high = {li: [] for li in active_layers}

    def process_group(examples, vecs_dict, group_name):
        skipped = 0
        i = 0
        while i < len(examples):
            batch = examples[i:i + batch_size]
            i += batch_size

            texts = []
            positions = []
            for ex in batch:
                full_text = build_tallrec_text(ex, tokenizer_i)
                pos = find_match_end_position(full_text, tokenizer_i)
                if pos is None:
                    skipped += 1
                else:
                    texts.append(full_text)
                    positions.append(pos)

            if not texts:
                continue

            enc = tokenizer_i(
                texts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).to(device)

            valid_mask = [enc["input_ids"].shape[1] > p for p in positions]
            if not any(valid_mask):
                skipped += sum(1 for _ in texts)
                continue

            outputs = model_i(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )

            for idx, (pos, valid) in enumerate(zip(positions, valid_mask)):
                if not valid:
                    skipped += 1
                    continue
                for li in vecs_dict:
                    hs = outputs.hidden_states[li + 1][idx, pos].detach().cpu().float()
                    vecs_dict[li].append(hs)

            del outputs, enc
            torch.cuda.empty_cache()

        if skipped:
            print(f"  TALLRec {group_name}: skipped {skipped} examples")

    process_group(low_examples, layer_vecs_low, "D_low")
    process_group(high_examples, layer_vecs_high, "D_high")

    validation_stats = {}

    for li in bias_directions:
        direction = bias_directions[li]
        low_vecs = layer_vecs_low.get(li, [])
        high_vecs = layer_vecs_high.get(li, [])

        if not low_vecs or not high_vecs:
            print(f"  Layer {li}: insufficient TALLRec vectors, skipping validation.")
            continue

        L = torch.stack(low_vecs)
        H = torch.stack(high_vecs)

        if direction.dim() == 1:
            d_hat = direction
            proj_low = (L @ d_hat).mean().item()
            proj_high = (H @ d_hat).mean().item()
        else:
            B = direction
            proj_low = (L @ B).norm(dim=1).mean().item()
            proj_high = (H @ B).norm(dim=1).mean().item()

        gap = proj_high - proj_low
        print(
            f"  Layer {li}: TALLRec proj D_low={proj_low:.4f}, "
            f"D_high={proj_high:.4f}, gap={gap:.4f}"
        )
        if gap > 0:
            print(f"  Layer {li}: direction VALID — TALLRec positive bias aligns with direction.")
        else:
            print(f"  Layer {li}: WARNING — gap is non-positive, direction may not capture TALLRec bias.")

        validation_stats[str(li)] = {
            "tallrec_proj_low_mean": proj_low,
            "tallrec_proj_high_mean": proj_high,
            "gap": gap,
            "n_low": len(low_vecs),
            "n_high": len(high_vecs),
        }

    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "tallrec_validation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(validation_stats, f, indent=2)
    print(f"TALLRec validation stats saved to {stats_path}")

    return validation_stats


def print_probe_summary(reczero_probe_stats):
    print("RecZero probe validation with D_correct_low_relaxed...")
    for layer_str, layer_stats in reczero_probe_stats.items():
        harm = layer_stats.get("D_harm_fit")
        high = layer_stats.get("D_correct_high_fit")
        probe = layer_stats.get("D_correct_low_relaxed_probe")
        if not (harm and high and probe):
            print(f"  Layer {layer_str}: incomplete probe stats, skipping summary.")
            continue

        probe_val = probe["proj_mean"]
        harm_val = harm["proj_mean"]
        high_val = high["proj_mean"]
        dist_to_harm = abs(probe_val - harm_val)
        dist_to_high = abs(probe_val - high_val)
        print(
            f"  Layer {layer_str}: probe={probe_val:.4f}, "
            f"D_harm={harm_val:.4f}, D_correct_high={high_val:.4f}, "
            f"|probe-harm|={dist_to_harm:.4f}, |probe-high|={dist_to_high:.4f}"
        )


# ---------------------------------------------------------------------------
# Applying bias projection to delta tensors
# ---------------------------------------------------------------------------

def apply_bias_projection_to_deltas(
    projected_task_vectors: dict,
    bias_directions: dict,
    n_bias_directions: int,
):
    positive_tallrec_layers = {9, 10, 11, 12, 28, 32, 33, 34, 35}
    bias_directions = {
        li: direction
        for li, direction in bias_directions.items()
        if li in positive_tallrec_layers
    }

    def _make_projectors(direction):
        if n_bias_directions == 1:
            d = direction  # [d_model]

            def project_input_weight(delta):
                # delta: [d_out, d_model]
                return delta - d.unsqueeze(0) * (delta @ d).unsqueeze(1) / (d @ d)

            def project_output_weight(delta):
                # delta: [d_model, d_other]
                return delta - d.unsqueeze(1) * (d @ delta).unsqueeze(0) / (d @ d)

        else:
            B = direction  # [d_model, k]
            k = B.shape[1]
            BtB = B.T @ B  # [k, k]
            BtB_inv = torch.linalg.solve(BtB, torch.eye(k, dtype=BtB.dtype, device=BtB.device))
            P = B @ BtB_inv @ B.T  # [d_model, d_model]

            def project_input_weight(delta):
                # delta: [d_out, d_model]; project along d_model (columns)
                return delta - delta @ P.T

            def project_output_weight(delta):
                # delta: [d_model, d_other]; project along d_model (rows)
                return delta - P @ delta

        return project_input_weight, project_output_weight

    total_layers_processed = 0

    for li, direction in bias_directions.items():
        direction_cpu = direction.cpu().float()
        proj_in, proj_out = _make_projectors(direction_cpu)

        layer_tensors_count = 0
        norm_reductions = []

        # --- QK ---
        qk_layer = projected_task_vectors.get("qk", {}).get(li, {})
        for head_idx, head_dict in qk_layer.items():
            for key in list(head_dict.keys()):
                delta = head_dict[key]
                orig_dtype = delta.dtype
                delta_f = delta.cpu().float()
                norm_before = delta_f.norm().item()
                # [d_model, head_dim]: transpose to [head_dim, d_model], apply proj_in, transpose back
                projected = proj_in(delta_f.T).T
                norm_after = projected.norm().item()
                if norm_before > 1e-12:
                    norm_reductions.append((norm_before - norm_after) / norm_before)
                head_dict[key] = projected.to(orig_dtype)
                layer_tensors_count += 1

        # --- VO ---
        vo_layer = projected_task_vectors.get("vo", {}).get(li, {})
        for head_idx, head_dict in vo_layer.items():
            for key in list(head_dict.keys()):
                delta = head_dict[key]
                orig_dtype = delta.dtype
                delta_f = delta.cpu().float()
                norm_before = delta_f.norm().item()

                is_output = "dO_proj" in key
                if is_output:
                    # O writes into hidden space, shape [d_model, head_dim]
                    projected = proj_out(delta_f)
                else:
                    # V reads from hidden space, shape [d_model, head_dim], d_model is dim 0
                    # same as Q/K: transpose to get [head_dim, d_model], apply proj_in, transpose back
                    projected = proj_in(delta_f.T).T

                norm_after = projected.norm().item()
                if norm_before > 1e-12:
                    norm_reductions.append((norm_before - norm_after) / norm_before)
                head_dict[key] = projected.to(orig_dtype)
                layer_tensors_count += 1

        # --- FFN ---
        ffn_layer = projected_task_vectors.get("ffn", {}).get(li, {})
        for key in list(ffn_layer.keys()):
            delta = ffn_layer[key]
            orig_dtype = delta.dtype
            delta_f = delta.cpu().float()
            norm_before = delta_f.norm().item()

            # Gate, Up, Down_T all have shape [d_ff, d_model] with d_model as dim 1
            projected = proj_in(delta_f)

            norm_after = projected.norm().item()
            if norm_before > 1e-12:
                norm_reductions.append((norm_before - norm_after) / norm_before)
            ffn_layer[key] = projected.to(orig_dtype)
            layer_tensors_count += 1

        mean_reduction = float(np.mean(norm_reductions)) if norm_reductions else 0.0
        print(
            f"  Layer {li}: projected {layer_tensors_count} tensors, "
            f"mean norm reduction={mean_reduction:.4f}"
        )
        total_layers_processed += 1

    print(f"Bias projection applied to {total_layers_processed} layers.")
    return projected_task_vectors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply rating-bias null-space projection to projected task vectors."
    )
    parser.add_argument("--output_pkl", required=True, help="Path to write debiased output pkl")
    parser.add_argument("--input_pkl", default="", help="Path to existing projected_task_vectors.pkl to skip rebuilding deltas from scratch")
    parser.add_argument("--base_model_path", default="", help="Path to base model directory (required when --input_pkl is not set)")
    parser.add_argument("--model_r_path", required=True, help="Path to RecZero model directory")
    parser.add_argument("--model_i_path", default="", help="Path to TALLRec model directory (required when --input_pkl is not set)")
    parser.add_argument("--merge_types", type=str, default="qkvof", help="Combination of q/k/v/o/f")
    parser.add_argument("--layers", type=str, default="", help="Comma-separated layer ids; overrides --layers_tail")
    parser.add_argument("--layers_tail", type=int, default=0, help="Use the last N layers when --layers is omitted")
    parser.add_argument("--heads", type=str, default="all", help="Heads to include: all or comma-separated")
    parser.add_argument("--reczero_predictions", required=True, help="Path to RecZero predictions jsonl")
    parser.add_argument("--tallrec_predictions", required=True, help="Path to TALLRec predictions jsonl")
    parser.add_argument("--output_dir", required=True, help="Directory to save diagnostics JSON files")
    parser.add_argument("--low_rating_threshold", type=float, default=2.5)
    parser.add_argument("--high_rating_threshold", type=float, default=4.0)
    parser.add_argument("--tallrec_high_threshold", type=float, default=3.5)
    parser.add_argument("--n_bias_directions", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--min_group_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of examples per forward pass batch. Default 4.")
    parser.add_argument("--chunk_size", type=int, default=2048,
                        help="Process sequences in chunks of this size using KV cache to reduce peak memory. Default 2048.")
    parser.add_argument("--skip_tallrec_validation", action="store_true",
                        help="Skip TALLRec validation (Step 5). Saves half the forward passes on diagnostic runs.")
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Build or load projected task vectors
    if args.input_pkl:
        print(f"Loading projected task vectors from {args.input_pkl} ...")
        with open(args.input_pkl, "rb") as f:
            data = pickle.load(f)
        projected_task_vectors = data["projected_task_vectors"]
        config = dict(data.get("config", {}))
        cfg_probe = AutoConfig.from_pretrained(args.model_r_path)
        num_layers = cfg_probe.num_hidden_layers
        n_heads = cfg_probe.num_attention_heads
        selected_layers = _parse_selected_layers(args.layers, args.layers_tail or num_layers, num_layers)
        selected_heads = _parse_selected_heads(args.heads, n_heads)
        print(f"Loaded task vectors. Selected layers={selected_layers}, heads={selected_heads}")
    else:
        if not args.base_model_path or not args.model_i_path:
            raise ValueError("--base_model_path and --model_i_path are required when --input_pkl is not set")
        cfg_probe = AutoConfig.from_pretrained(args.model_i_path)
        num_layers = cfg_probe.num_hidden_layers
        n_heads = cfg_probe.num_attention_heads
        selected_layers = _parse_selected_layers(args.layers, args.layers_tail or num_layers, num_layers)
        selected_heads = _parse_selected_heads(args.heads, n_heads)
        print(f"Building raw delta: layers={selected_layers}, heads={selected_heads}, merge_types={args.merge_types.upper()}")
        data = build_raw_delta_projected(
            base_model_path=args.base_model_path,
            instruct_model_path=args.model_i_path,
            merge_types=args.merge_types,
            selected_layers=selected_layers,
            selected_heads=selected_heads,
        )
        projected_task_vectors = data["projected_task_vectors"]
        config = dict(data["config"])
        print(f"Selected layers from raw delta: {selected_layers}")

    # Step 2: Load RecZero model only
    print(f"Loading RecZero tokenizer from {args.model_r_path}...")
    tokenizer_r = AutoTokenizer.from_pretrained(args.model_r_path, use_fast=True)

    print(f"Loading RecZero model from {args.model_r_path}...")
    model_r = AutoModelForCausalLM.from_pretrained(
        args.model_r_path,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model_r.eval()

    # Step 3: Load calibration examples
    print(f"Loading calibration data...")
    calibration_examples, correct_low_validation, calibration_group_stats = load_calibration_examples(
        reczero_jsonl_path=args.reczero_predictions,
        tallrec_jsonl_path=args.tallrec_predictions,
        low_rating_threshold=args.low_rating_threshold,
        high_rating_threshold=args.high_rating_threshold,
        tallrec_high_threshold=args.tallrec_high_threshold,
    )

    # Step 4: Compute bias directions from RecZero
    bias_directions = compute_bias_directions(
        model=model_r,
        tokenizer=tokenizer_r,
        calibration_examples=calibration_examples,
        selected_layers=selected_layers,
        device=device,
        n_bias_directions=args.n_bias_directions,
        min_group_size=args.min_group_size,
        batch_size=args.batch_size,
    )

    if not bias_directions:
        print("ERROR: No bias directions computed. Aborting.")
        sys.exit(1)

    reczero_probe_stats = compute_group_projection_stats(
        model=model_r,
        tokenizer=tokenizer_r,
        groups={
            "D_harm_fit": [e for e in calibration_examples if e["bin_label"] == "low"],
            "D_correct_high_fit": [e for e in calibration_examples if e["bin_label"] == "high"],
            "D_correct_low_relaxed_probe": correct_low_validation,
        },
        bias_directions=bias_directions,
        device=device,
        batch_size=args.batch_size,
    )
    print_probe_summary(reczero_probe_stats)

    del model_r
    torch.cuda.empty_cache()

    # Step 5: Validate with TALLRec (lazy-loaded after RecZero is freed)
    if args.input_pkl and not args.model_i_path:
        print("Skipping TALLRec validation (--input_pkl mode with no --model_i_path).")
        args.skip_tallrec_validation = True
    if args.skip_tallrec_validation:
        print("Skipping TALLRec validation (--skip_tallrec_validation set).")
        validation_stats = {}
    else:
        print(f"Loading TALLRec tokenizer from {args.model_i_path}...")
        tokenizer_i = AutoTokenizer.from_pretrained(args.model_i_path, use_fast=True)

        print(f"Loading TALLRec model from {args.model_i_path}...")
        model_i = AutoModelForCausalLM.from_pretrained(
            args.model_i_path,
            torch_dtype=dtype,
            device_map=args.device,
        )
        model_i.eval()

        validation_stats = validate_direction_with_tallrec(
            model_i=model_i,
            tokenizer_i=tokenizer_i,
            calibration_examples=calibration_examples,
            bias_directions=bias_directions,
            device=device,
            selected_layers=selected_layers,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

        del model_i
        torch.cuda.empty_cache()

    # Save diagnostics
    diag_path = os.path.join(args.output_dir, "bias_projection_diagnostics.json")
    direction_stats = {}
    for li, d in bias_directions.items():
        if d.dim() == 1:
            direction_stats[str(li)] = {"shape": list(d.shape), "norm": d.norm().item()}
        else:
            direction_stats[str(li)] = {"shape": list(d.shape)}
    diagnostics = {
        "calibration_group_stats": calibration_group_stats,
        "direction_stats": direction_stats,
        "reczero_probe_stats": reczero_probe_stats,
        "tallrec_validation_stats": validation_stats,
        "args": vars(args),
    }
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Diagnostics saved to {diag_path}")

    # Step 6: Apply bias projection to delta tensors
    print("Applying bias projection to delta tensors...")
    projected_task_vectors = apply_bias_projection_to_deltas(
        projected_task_vectors=projected_task_vectors,
        bias_directions=bias_directions,
        n_bias_directions=args.n_bias_directions,
    )

    # Step 7: Save output pkl
    config["bias_projection_applied"] = True
    config["bias_projection_n_directions"] = args.n_bias_directions
    config["bias_projection_low_threshold"] = args.low_rating_threshold
    config["bias_projection_high_threshold"] = args.high_rating_threshold
    config["bias_projection_tallrec_threshold"] = args.tallrec_high_threshold

    output_data = dict(data)
    output_data["projected_task_vectors"] = projected_task_vectors
    output_data["config"] = config

    print(f"Saving output pkl to {args.output_pkl}...")
    with open(args.output_pkl, "wb") as f:
        pickle.dump(output_data, f)

    size_mb = os.path.getsize(args.output_pkl) / (1024 * 1024)
    print(f"Output pkl saved. File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
