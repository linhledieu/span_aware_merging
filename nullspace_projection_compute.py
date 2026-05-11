# #!/usr/bin/env python3
# """
# Null-space projection script - Stage 1 (with integrated bias projection)
# Purpose: compute and save projected task vectors, then optionally apply rating-bias
# null-space projection in a single pass (previously a separate step0b_bias_projection.py step).
# Output: a file with projected task vectors to be used by later scaling.
# """

# import os
# import json
# import math
# import argparse
# import re
# import random
# import gc
# import pickle
# import copy
# import sys
# from collections import Counter
# from dataclasses import dataclass
# from typing import List, Dict, Tuple, Any, Optional
# from tqdm import tqdm
# import time

# import torch
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Import core functions
# from nullspace_merge_qkvo_ffn import (
#     ensure_dir, cleanup_memory, print_memory_status,
#     build_segmented_output, load_samples_data, extract_sample_fields, format_prompt_for_chat,
#     find_assistant_output_start,
#     resolve_model_device, resolve_compute_devices, load_materialized_model,
#     PreparedSample, prepare_samples_unified,
#     build_constraints_single_layer_unified, collect_layer_features_with_hooks,
#     task_vectors_single_layer_unified,
#     _sv_threshold_mask,
#     cg_single_head_batched, cg_v, cg_o, cg_ffn_down, cg_ffn_gate, cg_ffn_up,
#     # High-efficiency dense solvers
#     ffn_down_dense_project, ffn_gate_dense_project, ffn_up_dense_project,
#     q_dense_project, k_dense_project, v_dense_project, o_dense_project
# )


# ALL_SECTION_TAGS = [
#     "<analyze user>", "</analyze user>",
#     "<analyze item>", "</analyze item>",
#     "<match>", "</match>",
#     "<rate>", "</rate>",
# ]
# PROJECTION_BASE_KEYS = [
#     "dQ_proj", "dK_proj", "dV_proj", "dO_proj",
#     "dGate_proj", "dUp_proj", "dDown_T_proj",
# ]


# def _extract_rating_from_text(text: str) -> Optional[float]:
#     """Extract a numeric rating from <rate>...</rate> if present."""
#     if not isinstance(text, str) or not text:
#         return None

#     match = re.search(r"<rate>\s*([-+]?\d+(?:\.\d+)?)\s*</rate>", text, re.IGNORECASE | re.DOTALL)
#     if match:
#         try:
#             return float(match.group(1))
#         except ValueError:
#             return None

#     fallback = re.search(r"</rate>\s*([-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
#     if fallback:
#         try:
#             return float(fallback.group(1))
#         except ValueError:
#             return None

#     return None


# def _uniform_subsample_positions(positions: List[int], max_samples: int) -> List[int]:
#     """Uniformly subsample positions while preserving sorted order."""
#     if max_samples <= 0 or len(positions) <= max_samples:
#         return sorted(set(positions))

#     step = max(1, len(positions) // max_samples)
#     sampled = positions[::step]
#     if len(sampled) > max_samples:
#         sampled = sampled[:max_samples]
#     return sorted(set(sampled))


# def extract_section_positions_from_ids(
#     token_ids: List[int],
#     tokenizer,
#     max_interior_samples: int = 16,
# ) -> Dict[str, List[int]]:
#     """Extract boundary and interior token positions using exact tag token ids."""
#     tag_id_map = {
#         tag: tokenizer.encode(tag, add_special_tokens=False)
#         for tag in ALL_SECTION_TAGS
#     }
#     tag_bounds: Dict[str, Optional[Tuple[int, int]]] = {tag: None for tag in ALL_SECTION_TAGS}

#     for tag, tag_ids in tag_id_map.items():
#         if not tag_ids:
#             print(f"⚠️  Failed to encode tag: {tag}")
#             continue

#         width = len(tag_ids)
#         for pos in range(0, max(0, len(token_ids) - width + 1)):
#             if token_ids[pos:pos + width] == tag_ids:
#                 tag_bounds[tag] = (pos, pos + width - 1)

#     # Some tokenizers encode XML-ish tags differently depending on leading
#     # whitespace/newline context. Use decoded-text offsets as a robust fallback,
#     # and prefer the last occurrence so prompt format examples do not win over
#     # the assistant's actual tagged output.
#     decoded_text = tokenizer.decode(
#         token_ids,
#         skip_special_tokens=False,
#         clean_up_tokenization_spaces=False,
#     )
#     try:
#         encoded_with_offsets = tokenizer(
#             decoded_text,
#             add_special_tokens=False,
#             return_offsets_mapping=True,
#         )
#         offsets = encoded_with_offsets.get("offset_mapping", [])
#         offsets_match_tokens = len(offsets) == len(token_ids)
#     except Exception:
#         offsets = []
#         offsets_match_tokens = False

#     if offsets_match_tokens:
#         for tag in ALL_SECTION_TAGS:
#             char_start = decoded_text.rfind(tag)
#             if char_start < 0:
#                 continue
#             char_end = char_start + len(tag)
#             token_positions = [
#                 idx
#                 for idx, (start, end) in enumerate(offsets)
#                 if end > char_start and start < char_end
#             ]
#             if token_positions:
#                 tag_bounds[tag] = (token_positions[0], token_positions[-1])

#     for tag in ALL_SECTION_TAGS:
#         if tag_bounds[tag] is None:
#             print(f"⚠️  Missing tag while extracting section positions: {tag}")

#     def _interior_positions(open_tag: str, close_tag: str) -> List[int]:
#         open_bounds = tag_bounds.get(open_tag)
#         close_bounds = tag_bounds.get(close_tag)
#         if open_bounds is None or close_bounds is None:
#             return []
#         start = open_bounds[1] + 1
#         end = close_bounds[0] - 1
#         if end < start:
#             return []
#         return list(range(start, end + 1))

#     boundary = sorted({
#         position
#         for bounds in tag_bounds.values()
#         if bounds is not None
#         for position in range(bounds[0], bounds[1] + 1)
#     })

#     match_interior = _interior_positions("<match>", "</match>") + _interior_positions("<rate>", "</rate>")
#     analyze_interior = _interior_positions("<analyze user>", "</analyze user>") + _interior_positions("<analyze item>", "</analyze item>")

#     return {
#         "boundary": boundary,
#         "match_interior": _uniform_subsample_positions(match_interior, max_interior_samples),
#         "analyze_interior": _uniform_subsample_positions(analyze_interior, max_interior_samples),
#         "tag_bounds": {tag: bounds for tag, bounds in tag_bounds.items() if bounds is not None},
#     }


# def extract_all_section_positions(
#     full_texts: List[str],
#     tokenizer,
#     max_interior_samples: int,
# ) -> List[Dict[str, List[int]]]:
#     """Extract section positions for all texts and print a short summary."""
#     all_positions = []
#     full_count = 0
#     partial_count = 0
#     analyze_lengths = []

#     for text in full_texts:
#         token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
#         pos_dict = extract_section_positions_from_ids(token_ids, tokenizer, max_interior_samples=max_interior_samples)
#         all_positions.append(pos_dict)
#         analyze_lengths.append(len(pos_dict.get("analyze_interior", [])))
#         if len(pos_dict.get("tag_bounds", {})) == len(ALL_SECTION_TAGS):
#             full_count += 1
#         else:
#             partial_count += 1

#     mean_analyze_len = float(sum(analyze_lengths) / len(analyze_lengths)) if analyze_lengths else 0.0
#     print("📍 Section position summary:")
#     print(f"  All eight tags found: {full_count}")
#     print(f"  Partial tags: {partial_count}")
#     print(f"  Mean analyze interior length (sampled): {mean_analyze_len:.2f}")
#     return all_positions


# def compute_subspace_overlap(analyze_features: torch.Tensor, match_features: torch.Tensor) -> float:
#     """Compute a normalized overlap score between two row-spaces."""
#     if analyze_features is None or match_features is None:
#         return 0.0
#     if analyze_features.ndim != 2 or match_features.ndim != 2:
#         return 0.0
#     if analyze_features.numel() == 0 or match_features.numel() == 0:
#         return 0.0

#     analyze_norm = analyze_features / (analyze_features.norm(dim=1, keepdim=True) + 1e-8)
#     match_norm = match_features / (match_features.norm(dim=1, keepdim=True) + 1e-8)
#     numer = torch.linalg.norm(analyze_norm.T @ match_norm, ord="fro")
#     denom = torch.linalg.norm(analyze_norm, ord="fro") * torch.linalg.norm(match_norm, ord="fro")
#     if denom.item() <= 0.0:
#         return 0.0
#     return float((numer / denom).clamp(min=0.0, max=1.0).item())


# def _normalize_vector(vec: torch.Tensor) -> torch.Tensor:
#     return vec / (vec.norm() + 1e-8)


# def _get_layer_hidden_state(hidden_states, layer_idx: int) -> Optional[torch.Tensor]:
#     if hidden_states is None:
#         return None
#     target_idx = layer_idx + 1
#     if target_idx < 0 or target_idx >= len(hidden_states):
#         return None
#     state = hidden_states[target_idx]
#     if state.dim() == 3:
#         return state[0]
#     return state


# def _ground_truth_rating_from_sample(sample: Any) -> Optional[float]:
#     if isinstance(sample, dict):
#         for key in ["rating", "score", "label", "target_rating"]:
#             if key in sample and sample[key] is not None:
#                 try:
#                     return float(sample[key])
#                 except (TypeError, ValueError):
#                     pass

#     prompt, reasoning, response = extract_sample_fields(sample)
#     candidate_text = build_segmented_output(reasoning, response)
#     if not candidate_text:
#         candidate_text = prompt
#     return _extract_rating_from_text(candidate_text)


# @torch.no_grad()
# def get_reczero_predictions(
#     model,
#     tokenizer,
#     raw_samples: List[Any],
#     device: str,
#     max_new_tokens: int = 400,
# ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
#     """Get RecZero predictions, preferring cached dataset predictions when present."""
#     ground_truth_ratings: List[Optional[float]] = []
#     predicted_ratings: List[Optional[float]] = []
#     cached_count = 0
#     generated_count = 0

#     model.eval()
#     for sample in tqdm(raw_samples, desc="RecZero prediction pass"):
#         prompt, _, _ = extract_sample_fields(sample)
#         ground_truth = _ground_truth_rating_from_sample(sample)

#         predicted = None
#         if isinstance(sample, dict):
#             for key in ["pred_rating", "predicted_rating", "prediction", "reczero_pred_rating"]:
#                 if key in sample and sample[key] is not None:
#                     try:
#                         predicted = float(sample[key])
#                         break
#                     except (TypeError, ValueError):
#                         predicted = _extract_rating_from_text(str(sample[key]))
#                         if predicted is not None:
#                             break

#             if predicted is None:
#                 for key in ["raw_output_long_native", "raw_output", "output", "response"]:
#                     if key in sample and sample[key]:
#                         predicted = _extract_rating_from_text(str(sample[key]))
#                         if predicted is not None:
#                             break

#         if predicted is not None:
#             cached_count += 1
#         else:
#             formatted_prompt = format_prompt_for_chat(prompt, tokenizer)
#             enc = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
#             input_ids = enc["input_ids"].to(device)
#             attention_mask = enc["attention_mask"].to(device)

#             try:
#                 output_ids = model.generate(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     max_new_tokens=max_new_tokens,
#                     do_sample=False,
#                     temperature=1.0,
#                     top_p=1.0,
#                     top_k=50,
#                     pad_token_id=tokenizer.eos_token_id,
#                 )
#                 generated = tokenizer.decode(output_ids[0], skip_special_tokens=False)
#                 predicted = _extract_rating_from_text(generated)
#                 generated_count += 1
#             except Exception as exc:
#                 print(f"⚠️  Prediction generation failed for one sample: {exc}")

#         ground_truth_ratings.append(ground_truth)
#         predicted_ratings.append(predicted)

#     valid_pairs = sum(
#         1 for gt, pred in zip(ground_truth_ratings, predicted_ratings)
#         if gt is not None and pred is not None
#     )
#     print(f"📊 RecZero predictions parsed: {valid_pairs}/{len(raw_samples)} valid rating pairs")
#     print(f"   Cached predictions used: {cached_count}; generated predictions: {generated_count}")
#     return ground_truth_ratings, predicted_ratings


# def read_json_samples_robust(path: str, tokenizer, max_n: Optional[int] = None) -> List[str]:
#     """Read samples from JSON or JSONL and build full conversations."""
#     samples = load_samples_data(path)
    
#     full_prompts = []
#     for sample in samples:
#         if max_n is not None and len(full_prompts) >= max_n:
#             break

#         prompt, reasoning, response = extract_sample_fields(sample)
        
#         formatted_prompt = format_prompt_for_chat(prompt, tokenizer)
        
#         # Keep the segmented reasoning markup intact and append response text if present.
#         full_prompt = formatted_prompt + build_segmented_output(reasoning, response)
#         full_prompts.append(full_prompt)
    
#     return full_prompts


# def _char_pos_to_token_index(
#     offsets: List[Tuple[int, int]],
#     char_pos: int,
# ) -> Optional[int]:
#     """Map a character position to the token index whose span covers it."""
#     for idx, (start, end) in enumerate(offsets):
#         if start <= char_pos < end:
#             return idx
#     return None


# def _find_closing_tag_token_starts(
#     text: str,
#     tokenizer,
#     closing_tags: List[str],
# ) -> Dict[str, List[int]]:
#     """
#     Find the first token index for each closing-tag occurrence in context.

#     This is offset-based rather than token-sequence-based because the tokenizer
#     may merge surrounding whitespace/newlines into the first tag token.
#     """
#     search_start = find_assistant_output_start(text)
#     search_text = text[search_start:]

#     enc = tokenizer(
#         text,
#         add_special_tokens=False,
#         return_offsets_mapping=True,
#     )
#     offsets = enc["offset_mapping"]

#     tag_token_starts = {tag: [] for tag in closing_tags}
#     for tag in closing_tags:
#         for match in re.finditer(re.escape(tag), search_text):
#             char_pos = search_start + match.start()
#             tok_idx = _char_pos_to_token_index(offsets, char_pos)
#             if tok_idx is not None:
#                 tag_token_starts[tag].append(tok_idx)

#     return tag_token_starts


# def _chunk_list(items: List[Any], chunk_size: int):
#     """Yield fixed-size chunks from a list."""
#     if chunk_size <= 0:
#         raise ValueError("chunk_size must be positive")
#     for start in range(0, len(items), chunk_size):
#         yield items[start:start + chunk_size]


# def measure_baseline_closing_tag_margins(
#     model,
#     tokenizer,
#     full_texts: List[str],
#     device: str = "cuda:0",
#     batch_size: int = 4,
# ) -> Dict[str, Any]:
#     """
#     Measure first-token closing-tag margins at the only valid prediction position:
#     the token immediately before the closing tag begins.
#     """
#     closing_tags = [
#         "</analyze user>",
#         "</analyze item>",
#         "</match>",
#         "</rate>",
#     ]

#     all_margins = {tag: [] for tag in closing_tags}

#     valid_texts = []
#     for text in full_texts:
#         seq_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
#         if seq_len <= 6000:
#             valid_texts.append(text)

#     model.eval()
#     original_padding_side = tokenizer.padding_side
#     original_pad_token = tokenizer.pad_token
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     try:
#         total_batches = math.ceil(len(valid_texts) / batch_size) if valid_texts else 0
#         for batch_texts in tqdm(
#             _chunk_list(valid_texts, batch_size),
#             total=total_batches,
#             desc="Baseline margin measurement",
#         ):
#             enc = tokenizer(
#                 batch_texts,
#                 return_tensors="pt",
#                 add_special_tokens=False,
#                 padding=True,
#             )
#             input_ids = enc["input_ids"].to(device)
#             attention_mask = enc["attention_mask"].to(device)

#             with torch.no_grad():
#                 outputs = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     use_cache=False,
#                 )
#                 logits = outputs.logits

#             for row_idx, text in enumerate(batch_texts):
#                 seq_len = int(attention_mask[row_idx].sum().item())
#                 ids_list = input_ids[row_idx, :seq_len].tolist()
#                 tag_token_starts = _find_closing_tag_token_starts(text, tokenizer, closing_tags)

#                 for tag in closing_tags:
#                     for token_start in tag_token_starts[tag]:
#                         if token_start >= seq_len:
#                             continue

#                         pred_pos = token_start - 1
#                         if pred_pos < 0:
#                             continue

#                         first_tok = ids_list[token_start]
#                         pred_logits = logits[row_idx, pred_pos]
#                         closing_logit = pred_logits[first_tok].item()

#                         pred_logits_copy = pred_logits.clone()
#                         pred_logits_copy[first_tok] = float("-inf")
#                         max_other = pred_logits_copy.max().item()

#                         all_margins[tag].append(closing_logit - max_other)
#     finally:
#         tokenizer.padding_side = original_padding_side
#         tokenizer.pad_token = original_pad_token

#     result = {
#         "mean_margin_per_tag": {},
#         "min_margin_per_tag": {},
#         "all_margins_per_tag": all_margins,
#     }
#     for tag in closing_tags:
#         margins = all_margins[tag]
#         if margins:
#             result["mean_margin_per_tag"][tag] = float(sum(margins) / len(margins))
#             result["min_margin_per_tag"][tag] = float(min(margins))
#         else:
#             result["mean_margin_per_tag"][tag] = 0.0
#             result["min_margin_per_tag"][tag] = 0.0

#     return result


# def _stable_softmax_with_causal_mask(scores: torch.Tensor) -> torch.Tensor:
#     """Numerically stable causal softmax for [H, T, T] attention scores."""
#     _, seq_len, _ = scores.shape
#     mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool))
#     scores_masked = scores.clone()
#     scores_masked[~mask.unsqueeze(0).expand_as(scores_masked)] = torch.finfo(scores.dtype).min
#     row_max = scores_masked.max(dim=-1, keepdim=True).values
#     exp_scores = torch.exp(scores_masked - row_max) * mask.unsqueeze(0)
#     denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
#     return exp_scores / denom


# def _concat_cat_tensors(*values: torch.Tensor) -> torch.Tensor:
#     valid = [
#         value.to(torch.float32)
#         for value in values
#         if value is not None and value.numel() > 0
#     ]

#     if not valid:
#         return torch.empty(0, dtype=torch.float32)

#     max_ndim = max(value.ndim for value in valid)

#     fixed = []
#     for value in valid:
#         if value.ndim == max_ndim:
#             fixed.append(value)
#             continue

#         if value.ndim == 1 and max_ndim == 2:
#             # If the other tensor is [N, 1], this is a scale vector.
#             # Convert [N] -> [N, 1].
#             ref = next(v for v in valid if v.ndim == 2)
#             if ref.shape[1] == 1:
#                 value = value.unsqueeze(-1)
#             else:
#                 # Otherwise it is a feature vector.
#                 # Convert [D] -> [1, D].
#                 value = value.unsqueeze(0)

#         fixed.append(value)

#     return torch.cat(fixed, dim=0).contiguous()

# def _concat_stack_tensors(*values: torch.Tensor) -> torch.Tensor:
#     valid = [value for value in values if value is not None and value.numel() > 0]
#     if not valid:
#         return torch.empty(0, dtype=torch.float32)
#     return torch.cat(valid, dim=0).contiguous()


# def _normalize_row(vec: torch.Tensor) -> torch.Tensor:
#     return vec / (vec.norm() + 1e-8)


# @torch.no_grad()
# def build_match_interior_feature_matrix(
#     model,
#     tokenizer,
#     full_calibration_texts: List[str],
#     section_positions_per_sample: List[Dict[str, List[int]]],
#     layer_index: int,
#     selected_heads: Optional[List[int]],
#     merge_types: str,
#     device: str,
#     compute_dtype: torch.dtype,
#     constraint_sv_threshold: float = 1e-2,
#     bin_labels: Optional[List[str]] = None,
# ) -> Dict[str, Any]:
#     """
#     Build match-interior constraint rows and analyze/match feature matrices for one layer.
#     """
#     d_model = model.config.hidden_size
#     num_heads = model.config.num_attention_heads
#     head_dim = d_model // num_heads
#     num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
#     active_heads = selected_heads if selected_heads is not None else list(range(num_heads))

#     layer_obj = model.model.layers[layer_index]
#     w_o = layer_obj.self_attn.o_proj.weight.data.to(device=device, dtype=torch.float32)
#     w_d = layer_obj.mlp.down_proj.weight.data.to(device=device, dtype=torch.float32)

#     match_constraints = {
#         "qk": {},
#         "vo": {},
#         "ffn": {
#             "H": [], "c": [], "sc": [],
#             "X_gate": [], "c_gate": [], "sc_gate": [],
#             "X_up": [], "c_up": [], "sc_up": [],
#         },
#     }
#     match_features = {"qk": [], "vo": [], "ffn": []}
#     analyze_features = {"qk": [], "vo": [], "ffn": []}
#     analyze_effect_inputs = {
#         "qk": {},
#         "vo_v": {},
#         "vo_o": {},
#         "ffn_gate": [],
#         "ffn_up": [],
#         "ffn_down": [],
#     }

#     for head in active_heads:
#         match_constraints["qk"][head] = {
#             "Xi_q": [], "kj": [], "sc_q": [],
#             "Xj_k": [], "qi": [], "sc_k": [],
#         }
#         match_constraints["vo"][head] = {
#             "Xi_v": [], "rv": [], "sc_v": [],
#             "c_vec": [], "z_h": [], "sc_o": [],
#         }
#         analyze_effect_inputs["qk"][head] = []
#         analyze_effect_inputs["vo_v"][head] = []
#         analyze_effect_inputs["vo_o"][head] = []

#     # Per-sample bias accumulators (collected alongside existing forward passes)
#     bias_vecs_low: List[torch.Tensor] = []
#     bias_vecs_high: List[torch.Tensor] = []

#     model.eval()
#     for sample_idx, (text, pos_dict) in enumerate(tqdm(
#         zip(full_calibration_texts, section_positions_per_sample),
#         total=len(full_calibration_texts),
#         desc=f"Interior features layer {layer_index}",
#         leave=False,
#     )):
#         match_positions = pos_dict.get("match_interior", [])
#         analyze_positions = pos_dict.get("analyze_interior", [])
#         if not match_positions and not analyze_positions:
#             continue

#         enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
#         input_ids = enc["input_ids"].to(device)

#         layer_features = collect_layer_features_with_hooks(
#             model,
#             input_ids,
#             [layer_index],
#             merge_types="qkvof",
#         )
#         if layer_index not in layer_features or not layer_features[layer_index]:
#             continue

#         feat = layer_features[layer_index]
#         attn_input = feat["attn_input"].to(device=device, dtype=torch.float32)
#         ffn_input = feat["ffn_input"].to(device=device, dtype=torch.float32)
#         gate_out = feat["gate_output"][0].to(device=device, dtype=torch.float32)
#         up_out = feat["up_output"][0].to(device=device, dtype=torch.float32)
#         q_proj_out = feat["q_proj_out"][0].to(device=device, dtype=torch.float32)
#         k_proj_out = feat["k_proj_out"][0].to(device=device, dtype=torch.float32)
#         v_proj_out = feat["v_proj_out"][0].to(device=device, dtype=torch.float32)

#         seq_len = attn_input.shape[0]

#         # Collect hidden state at </match> position for bias direction computation
#         if bin_labels is not None and sample_idx < len(bin_labels):
#             match_close = pos_dict.get("tag_bounds", {}).get("</match>")
#             if match_close is not None:
#                 close_pos = match_close[0]
#                 if close_pos < seq_len:
#                     vec = attn_input[close_pos].detach().cpu().float()
#                     if bin_labels[sample_idx] == "low":
#                         bias_vecs_low.append(vec)
#                     elif bin_labels[sample_idx] == "high":
#                         bias_vecs_high.append(vec)

#         q = q_proj_out.view(seq_len, num_heads, head_dim).permute(1, 0, 2).contiguous()
#         k = k_proj_out.view(seq_len, num_kv_heads, head_dim).permute(1, 0, 2).contiguous()
#         v = v_proj_out.view(seq_len, num_kv_heads, head_dim).permute(1, 0, 2).contiguous()
#         if num_kv_heads < num_heads:
#             rep = num_heads // num_kv_heads
#             k = k.repeat_interleave(rep, dim=0)
#             v = v.repeat_interleave(rep, dim=0)
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
#         attn_weights = _stable_softmax_with_causal_mask(scores)
#         swiglu_hidden = torch.nn.functional.silu(gate_out) * up_out
#         down_out = swiglu_hidden @ w_d.T

#         valid_match_positions = [pos for pos in match_positions if pos < seq_len]
#         valid_analyze_positions = [pos for pos in analyze_positions if pos < seq_len]

#         if valid_match_positions:
#             match_features["qk"].append(attn_input[valid_match_positions].detach().cpu())
#             match_features["vo"].append(attn_input[valid_match_positions].detach().cpu())
#             match_features["ffn"].append(ffn_input[valid_match_positions].detach().cpu())

#         if valid_analyze_positions:
#             analyze_features["qk"].append(attn_input[valid_analyze_positions].detach().cpu())
#             analyze_features["vo"].append(attn_input[valid_analyze_positions].detach().cpu())
#             analyze_features["ffn"].append(ffn_input[valid_analyze_positions].detach().cpu())
#             analyze_effect_inputs["ffn_gate"].append(ffn_input[valid_analyze_positions].detach().cpu())
#             analyze_effect_inputs["ffn_up"].append(ffn_input[valid_analyze_positions].detach().cpu())
#             analyze_effect_inputs["ffn_down"].append(swiglu_hidden[valid_analyze_positions].detach().cpu())

#         for head in active_heads:
#             kv_head = head % num_kv_heads
#             o_block = w_o[:, head * head_dim:(head + 1) * head_dim]
#             z_analyze_rows = []

#             for pos in valid_match_positions:
#                 x_t = attn_input[pos]
#                 q_t = q[head, pos]
#                 k_t = k[head, pos]
#                 z_t = attn_weights[head, pos] @ v[head]
#                 o_current = o_block @ z_t
#                 c_vec = _normalize_row(o_current)
#                 rv = o_block.T @ c_vec

#                 match_constraints["qk"][head]["Xi_q"].append(x_t.detach().cpu())
#                 match_constraints["qk"][head]["kj"].append(k_t.detach().cpu())
#                 match_constraints["qk"][head]["sc_q"].append(torch.tensor([1.0 / math.sqrt(head_dim)], dtype=torch.float32))

#                 match_constraints["qk"][head]["Xj_k"].append(x_t.detach().cpu())
#                 match_constraints["qk"][head]["qi"].append(q_t.detach().cpu())
#                 match_constraints["qk"][head]["sc_k"].append(torch.tensor([1.0 / math.sqrt(head_dim)], dtype=torch.float32))

#                 match_constraints["vo"][head]["Xi_v"].append(x_t.detach().cpu())
#                 match_constraints["vo"][head]["rv"].append(rv.detach().cpu())
#                 match_constraints["vo"][head]["sc_v"].append(torch.tensor([1.0 / math.sqrt(head_dim)], dtype=torch.float32))

#                 match_constraints["vo"][head]["c_vec"].append(c_vec.detach().cpu())
#                 match_constraints["vo"][head]["z_h"].append(z_t.detach().cpu())
#                 match_constraints["vo"][head]["sc_o"].append(torch.tensor([1.0 / math.sqrt(head_dim)], dtype=torch.float32))

#             for pos in valid_analyze_positions:
#                 analyze_effect_inputs["qk"][head].append(attn_input[pos].detach().cpu())
#                 analyze_effect_inputs["vo_v"][head].append(attn_input[pos].detach().cpu())
#                 z_analyze_rows.append((attn_weights[head, pos] @ v[head]).detach().cpu())

#             if z_analyze_rows:
#                 analyze_effect_inputs["vo_o"][head].append(torch.stack(z_analyze_rows, dim=0))

#         for pos in valid_match_positions:
#             x_t = ffn_input[pos]
#             gate_t = gate_out[pos]
#             up_t = up_out[pos]
#             hidden_t = swiglu_hidden[pos]
#             down_t = down_out[pos]

#             match_constraints["ffn"]["X_gate"].append(x_t.detach().cpu())
#             match_constraints["ffn"]["c_gate"].append(_normalize_row(gate_t).detach().cpu())
#             match_constraints["ffn"]["sc_gate"].append(torch.tensor([1.0 / math.sqrt(max(1, gate_t.numel()))], dtype=torch.float32))

#             match_constraints["ffn"]["X_up"].append(x_t.detach().cpu())
#             match_constraints["ffn"]["c_up"].append(_normalize_row(up_t).detach().cpu())
#             match_constraints["ffn"]["sc_up"].append(torch.tensor([1.0 / math.sqrt(max(1, up_t.numel()))], dtype=torch.float32))

#             match_constraints["ffn"]["H"].append(hidden_t.detach().cpu())
#             match_constraints["ffn"]["c"].append(_normalize_row(down_t).detach().cpu())
#             match_constraints["ffn"]["sc"].append(torch.tensor([1.0 / math.sqrt(max(1, hidden_t.numel()))], dtype=torch.float32))

#     for head in active_heads:
#         cons_qk = match_constraints["qk"][head]
#         for key in ["Xi_q", "kj", "Xj_k", "qi"]:
#             cons_qk[key] = torch.stack(cons_qk[key], dim=0).contiguous() if cons_qk[key] else torch.empty(0, dtype=torch.float32)
#         for key in ["sc_q", "sc_k"]:
#             cons_qk[key] = torch.stack(cons_qk[key], dim=0).contiguous() if cons_qk[key] else torch.empty(0, 1, dtype=torch.float32)

#         cons_vo = match_constraints["vo"][head]
#         for key in ["Xi_v", "rv", "c_vec", "z_h"]:
#             cons_vo[key] = torch.stack(cons_vo[key], dim=0).contiguous() if cons_vo[key] else torch.empty(0, dtype=torch.float32)
#         for key in ["sc_v", "sc_o"]:
#             cons_vo[key] = torch.stack(cons_vo[key], dim=0).contiguous() if cons_vo[key] else torch.empty(0, 1, dtype=torch.float32)

#         analyze_effect_inputs["qk"][head] = torch.stack(analyze_effect_inputs["qk"][head], dim=0) if analyze_effect_inputs["qk"][head] else torch.empty((0, d_model), dtype=torch.float32)
#         analyze_effect_inputs["vo_v"][head] = torch.stack(analyze_effect_inputs["vo_v"][head], dim=0) if analyze_effect_inputs["vo_v"][head] else torch.empty((0, d_model), dtype=torch.float32)
#         analyze_effect_inputs["vo_o"][head] = torch.cat(analyze_effect_inputs["vo_o"][head], dim=0) if analyze_effect_inputs["vo_o"][head] else torch.empty((0, head_dim), dtype=torch.float32)

#     for key in ["H", "c", "X_gate", "c_gate", "X_up", "c_up"]:
#         match_constraints["ffn"][key] = torch.stack(match_constraints["ffn"][key], dim=0).contiguous() if match_constraints["ffn"][key] else torch.empty(0, dtype=torch.float32)
#     for key in ["sc", "sc_gate", "sc_up"]:
#         match_constraints["ffn"][key] = torch.stack(match_constraints["ffn"][key], dim=0).contiguous() if match_constraints["ffn"][key] else torch.empty(0, 1, dtype=torch.float32)

#     # ---- Correctness: verify analyze positions never contaminate match_constraints ----
#     # (match_constraints is only populated inside the valid_match_positions loop above)
#     # The analyze_effect_inputs dicts are populated exclusively from valid_analyze_positions.
#     # These assertions are a safety check — they would only fail if the loop logic changed.
#     for head in active_heads:
#         qk_rows = match_constraints["qk"][head]["Xi_q"].shape[0] if match_constraints["qk"][head]["Xi_q"].numel() > 0 else 0
#         qk_rows_k = match_constraints["qk"][head]["Xj_k"].shape[0] if match_constraints["qk"][head]["Xj_k"].numel() > 0 else 0
#         analyze_rows = analyze_effect_inputs["qk"][head].shape[0] if analyze_effect_inputs["qk"][head].numel() > 0 else 0
#         # The two pools must not overlap: match rows come only from valid_match_positions,
#         # analyze rows only from valid_analyze_positions — they are disjoint by construction.
#         assert qk_rows == qk_rows_k or qk_rows == 0 or qk_rows_k == 0, (
#             f"Layer {layer_index} head {head}: Xi_q rows ({qk_rows}) != Xj_k rows ({qk_rows_k}); "
#             "mixed-source contamination suspected"
#         )
#         # No assertion on analyze_rows vs qk_rows needed here because they are stored
#         # in separate dicts; the assertion above checks internal qk consistency.

#     # ---- SV threshold filtering on match interior constraints ----
#     interior_retention = {}

#     for head in active_heads:
#         ch_qk = match_constraints["qk"][head]
#         ch_vo = match_constraints["vo"][head]
#         ret_q = ret_k = ret_v = ret_o = 1.0
#         if ch_qk["Xi_q"].numel() > 0 and ch_qk["Xi_q"].ndim >= 2:
#             mask = _sv_threshold_mask(ch_qk["Xi_q"], constraint_sv_threshold)
#             before = mask.shape[0]
#             ch_qk["Xi_q"] = ch_qk["Xi_q"][mask]
#             ch_qk["kj"] = ch_qk["kj"][mask]
#             ch_qk["sc_q"] = ch_qk["sc_q"][mask] if ch_qk["sc_q"].ndim > 0 and ch_qk["sc_q"].shape[0] == before else ch_qk["sc_q"]
#             ret_q = mask.float().mean().item()
#         if ch_qk["Xj_k"].numel() > 0 and ch_qk["Xj_k"].ndim >= 2:
#             mask = _sv_threshold_mask(ch_qk["Xj_k"], constraint_sv_threshold)
#             before = mask.shape[0]
#             ch_qk["Xj_k"] = ch_qk["Xj_k"][mask]
#             ch_qk["qi"] = ch_qk["qi"][mask]
#             ch_qk["sc_k"] = ch_qk["sc_k"][mask] if ch_qk["sc_k"].ndim > 0 and ch_qk["sc_k"].shape[0] == before else ch_qk["sc_k"]
#             ret_k = mask.float().mean().item()
#         if ch_vo["Xi_v"].numel() > 0 and ch_vo["Xi_v"].ndim >= 2:
#             mask = _sv_threshold_mask(ch_vo["Xi_v"], constraint_sv_threshold)
#             before = mask.shape[0]
#             ch_vo["Xi_v"] = ch_vo["Xi_v"][mask]
#             ch_vo["rv"] = ch_vo["rv"][mask]
#             ch_vo["sc_v"] = ch_vo["sc_v"][mask] if ch_vo["sc_v"].ndim > 0 and ch_vo["sc_v"].shape[0] == before else ch_vo["sc_v"]
#             ret_v = mask.float().mean().item()
#         if ch_vo["c_vec"].numel() > 0 and ch_vo["c_vec"].ndim >= 2:
#             mask = _sv_threshold_mask(ch_vo["c_vec"], constraint_sv_threshold)
#             before = mask.shape[0]
#             ch_vo["c_vec"] = ch_vo["c_vec"][mask]
#             ch_vo["z_h"] = ch_vo["z_h"][mask]
#             ch_vo["sc_o"] = ch_vo["sc_o"][mask] if ch_vo["sc_o"].ndim > 0 and ch_vo["sc_o"].shape[0] == before else ch_vo["sc_o"]
#             ret_o = mask.float().mean().item()
#         interior_retention[head] = {"q": ret_q, "k": ret_k, "v": ret_v, "o": ret_o}

#     ffn_ret_gate = ffn_ret_up = ffn_ret_down = 1.0
#     fc = match_constraints["ffn"]
#     if fc["X_gate"].numel() > 0 and fc["X_gate"].ndim >= 2:
#         mask = _sv_threshold_mask(fc["X_gate"], constraint_sv_threshold)
#         before = mask.shape[0]
#         fc["X_gate"] = fc["X_gate"][mask]
#         fc["c_gate"] = fc["c_gate"][mask]
#         fc["sc_gate"] = fc["sc_gate"][mask] if fc["sc_gate"].ndim > 0 and fc["sc_gate"].shape[0] == before else fc["sc_gate"]
#         ffn_ret_gate = mask.float().mean().item()
#     if fc["X_up"].numel() > 0 and fc["X_up"].ndim >= 2:
#         mask = _sv_threshold_mask(fc["X_up"], constraint_sv_threshold)
#         before = mask.shape[0]
#         fc["X_up"] = fc["X_up"][mask]
#         fc["c_up"] = fc["c_up"][mask]
#         fc["sc_up"] = fc["sc_up"][mask] if fc["sc_up"].ndim > 0 and fc["sc_up"].shape[0] == before else fc["sc_up"]
#         ffn_ret_up = mask.float().mean().item()
#     if fc["H"].numel() > 0 and fc["H"].ndim >= 2:
#         mask = _sv_threshold_mask(fc["H"], constraint_sv_threshold)
#         before = mask.shape[0]
#         fc["H"] = fc["H"][mask]
#         fc["c"] = fc["c"][mask]
#         fc["sc"] = fc["sc"][mask] if fc["sc"].ndim > 0 and fc["sc"].shape[0] == before else fc["sc"]
#         ffn_ret_down = mask.float().mean().item()

#     return {
#         "match_constraints": match_constraints,
#         "match_feature_matrix": {
#             key: torch.cat(rows, dim=0) if rows else torch.empty((0, d_model), dtype=torch.float32)
#             for key, rows in match_features.items()
#         },
#         "analyze_feature_matrix": {
#             key: torch.cat(rows, dim=0) if rows else torch.empty((0, d_model), dtype=torch.float32)
#             for key, rows in analyze_features.items()
#         },
#         "analyze_effect_inputs": {
#             "qk": analyze_effect_inputs["qk"],
#             "vo_v": analyze_effect_inputs["vo_v"],
#             "vo_o": analyze_effect_inputs["vo_o"],
#             "ffn_gate": torch.cat(analyze_effect_inputs["ffn_gate"], dim=0) if analyze_effect_inputs["ffn_gate"] else torch.empty((0, d_model), dtype=torch.float32),
#             "ffn_up": torch.cat(analyze_effect_inputs["ffn_up"], dim=0) if analyze_effect_inputs["ffn_up"] else torch.empty((0, d_model), dtype=torch.float32),
#             "ffn_down": torch.cat(analyze_effect_inputs["ffn_down"], dim=0) if analyze_effect_inputs["ffn_down"] else torch.empty((0, w_d.shape[1]), dtype=torch.float32),
#         },
#         "interior_retention": {
#             "qk_vo": interior_retention,
#             "ffn": {"gate": ffn_ret_gate, "up": ffn_ret_up, "down": ffn_ret_down},
#         },
#         "bias_vecs_low": bias_vecs_low,
#         "bias_vecs_high": bias_vecs_high,
#     }


# def _combine_qk_constraints(boundary_cons: Dict[str, torch.Tensor], match_cons: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#     return {
#         "Xi_q": _concat_cat_tensors(boundary_cons.get("Xi_q"), match_cons.get("Xi_q")),
#         "kj": _concat_cat_tensors(boundary_cons.get("kj"), match_cons.get("kj")),
#         "sc_q": _concat_cat_tensors(boundary_cons.get("sc_q"), match_cons.get("sc_q")),
#         "Xj_k": _concat_cat_tensors(boundary_cons.get("Xj_k"), match_cons.get("Xj_k")),
#         "qi": _concat_cat_tensors(boundary_cons.get("qi"), match_cons.get("qi")),
#         "sc_k": _concat_cat_tensors(boundary_cons.get("sc_k"), match_cons.get("sc_k")),
#     }


# def _combine_vo_constraints(boundary_cons: Dict[str, torch.Tensor], match_cons: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#     return {
#         "Xi_v": _concat_stack_tensors(boundary_cons.get("Xi_v"), match_cons.get("Xi_v")),
#         "rv": _concat_stack_tensors(boundary_cons.get("rv"), match_cons.get("rv")),
#         "sc_v": _concat_cat_tensors(boundary_cons.get("sc_v"), match_cons.get("sc_v")),
#         "c_vec": _concat_stack_tensors(boundary_cons.get("c_vec"), match_cons.get("c_vec")),
#         "z_h": _concat_stack_tensors(boundary_cons.get("z_h"), match_cons.get("z_h")),
#         "sc_o": _concat_cat_tensors(boundary_cons.get("sc_o"), match_cons.get("sc_o")),
#     }


# def _combine_ffn_constraints(boundary_cons: Dict[str, torch.Tensor], match_cons: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#     return {
#         "H": _concat_stack_tensors(boundary_cons.get("H"), match_cons.get("H")),
#         "c": _concat_stack_tensors(boundary_cons.get("c"), match_cons.get("c")),
#         "sc": _concat_cat_tensors(boundary_cons.get("sc"), match_cons.get("sc")),
#         "X_gate": _concat_stack_tensors(boundary_cons.get("X_gate"), match_cons.get("X_gate")),
#         "c_gate": _concat_stack_tensors(boundary_cons.get("c_gate"), match_cons.get("c_gate")),
#         "sc_gate": _concat_cat_tensors(boundary_cons.get("sc_gate"), match_cons.get("sc_gate")),
#         "X_up": _concat_stack_tensors(boundary_cons.get("X_up"), match_cons.get("X_up")),
#         "c_up": _concat_stack_tensors(boundary_cons.get("c_up"), match_cons.get("c_up")),
#         "sc_up": _concat_cat_tensors(boundary_cons.get("sc_up"), match_cons.get("sc_up")),
#     }


# def _solve_projection(
#     kind: str,
#     cons: Dict[str, torch.Tensor],
#     task_delta: torch.Tensor,
#     lambda_ridge: float,
#     cg_maxit: int,
#     cg_tol: float,
#     device: str,
#     compute_dtype: torch.dtype,
# ) -> Tuple[torch.Tensor, Dict[str, Any]]:
#     if task_delta is None:
#         return None, {"residual_norm": 0.0, "iterations": 0, "solver": "skip_none"}

#     if kind == "q":
#         return q_dense_project(cons, task_delta, lam=lambda_ridge, device=device, compute_dtype=compute_dtype)
#     if kind == "k":
#         return k_dense_project(cons, task_delta, lam=lambda_ridge, device=device, compute_dtype=compute_dtype)
#     if kind == "v":
#         if cons.get("Xi_v", torch.empty(0)).shape[0] <= 4000:
#             return v_dense_project(cons, task_delta, lam=lambda_ridge, device=device, compute_dtype=compute_dtype)
#         return cg_v(cons, task_delta, lambda_ridge, cg_maxit, cg_tol, device=device, compute_dtype=compute_dtype)
#     if kind == "o":
#         if cons.get("c_vec", torch.empty(0)).shape[0] <= 4000:
#             return o_dense_project(cons, task_delta, lam=lambda_ridge, device=device, compute_dtype=compute_dtype)
#         return cg_o(cons, task_delta, lambda_ridge, cg_maxit, cg_tol, device=device, compute_dtype=compute_dtype)
#     if kind == "ffn_gate":
#         if cons.get("X_gate", torch.empty(0)).shape[0] <= 4000:
#             return ffn_gate_dense_project(cons, task_delta, lam=lambda_ridge, device=device, compute_dtype=compute_dtype)
#         return cg_ffn_gate(cons, task_delta, lambda_ridge, cg_maxit, cg_tol, device=device, compute_dtype=compute_dtype)
#     if kind == "ffn_up":
#         if cons.get("X_up", torch.empty(0)).shape[0] <= 4000:
#             return ffn_up_dense_project(cons, task_delta, lam=lambda_ridge, device=device, compute_dtype=compute_dtype)
#         return cg_ffn_up(cons, task_delta, lambda_ridge, cg_maxit, cg_tol, device=device, compute_dtype=compute_dtype)
#     if kind == "ffn_down":
#         if cons.get("H", torch.empty(0)).shape[0] <= 4000:
#             return ffn_down_dense_project(cons, task_delta, lam=lambda_ridge, device=device, compute_dtype=compute_dtype)
#         return cg_ffn_down(cons, task_delta, lambda_ridge, cg_maxit, cg_tol, device=device, compute_dtype=compute_dtype)
#     raise ValueError(f"Unsupported projection kind: {kind}")


# def _log_retention(layer_idx: int, source: str, retention: Dict[str, Any], selected_heads) -> None:
#     """Print a compact retention-rate summary for one layer and one constraint source."""
#     if not retention:
#         return
#     parts = []
#     # QK per-head (average across heads)
#     if "qk" in retention:
#         qk_data = retention["qk"]
#         avg_q = sum(v.get("q", 1.0) for v in qk_data.values()) / max(len(qk_data), 1)
#         avg_k = sum(v.get("k", 1.0) for v in qk_data.values()) / max(len(qk_data), 1)
#         parts.append(f"Q={avg_q:.2%} K={avg_k:.2%}")
#     # VO per-head (average across heads)
#     if "vo" in retention:
#         vo_data = retention["vo"]
#         avg_v = sum(v.get("v", 1.0) for v in vo_data.values()) / max(len(vo_data), 1)
#         avg_o = sum(v.get("o", 1.0) for v in vo_data.values()) / max(len(vo_data), 1)
#         parts.append(f"V={avg_v:.2%} O={avg_o:.2%}")
#     # FFN flat
#     if "ffn" in retention:
#         ffn = retention["ffn"]
#         parts.append(f"Gate={ffn.get('gate', 1.0):.2%} Up={ffn.get('up', 1.0):.2%} Down={ffn.get('down', 1.0):.2%}")
#     # Interior retention format (from build_match_interior_feature_matrix)
#     if "qk_vo" in retention:
#         qk_vo = retention["qk_vo"]
#         avg_q = sum(v.get("q", 1.0) for v in qk_vo.values()) / max(len(qk_vo), 1)
#         avg_k = sum(v.get("k", 1.0) for v in qk_vo.values()) / max(len(qk_vo), 1)
#         avg_v = sum(v.get("v", 1.0) for v in qk_vo.values()) / max(len(qk_vo), 1)
#         avg_o = sum(v.get("o", 1.0) for v in qk_vo.values()) / max(len(qk_vo), 1)
#         parts.append(f"Q={avg_q:.2%} K={avg_k:.2%} V={avg_v:.2%} O={avg_o:.2%}")
#     if "ffn" in retention and isinstance(retention["ffn"], dict) and "gate" in retention["ffn"]:
#         ffn = retention["ffn"]
#         parts.append(f"Gate={ffn.get('gate', 1.0):.2%} Up={ffn.get('up', 1.0):.2%} Down={ffn.get('down', 1.0):.2%}")
#     summary = " | ".join(parts) if parts else "(no data)"
#     print(f"    SV-filter [{source}] layer {layer_idx}: {summary}")


# def compute_nullspace_projections_original_behavior(
#     model_base, model_instruct, model_target,
#     texts_R: List[str], tokenizer,
#     selected_layers: List[int], selected_heads: List[int],
#     neigh_radius: int, lambda_ridge: float, cg_maxit: int, cg_tol: float,
#     compute_dtype: torch.dtype = torch.float32,
#     merge_types: str = "qk",
#     q_rows_per_text: int = 8, k_rows_per_text: int = 8, w_q: float = 1.0, w_k: float = 1.0,
#     v_rows_per_text: int = 4, o_rows_per_text: int = 4, w_v: float = 1.0, w_o: float = 1.0,
#     ffn_rows_per_text: int = 4, w_ffn: float = 1.0, readout_dirs: int = 2,
#     seed: int = 42,
#     qk_device: str = "auto", vo_device: str = "auto", ffn_device: str = "auto",
#     use_hooks: bool = True,
#     max_seq_len: int = 7168,
#     constraint_sv_threshold: float = 1e-2,
#     # Bias projection — can use a larger sample set than constraints
#     texts_bias: Optional[List[str]] = None,  # if None, falls back to texts_R
#     raw_samples: Optional[List[Any]] = None,
#     skip_bias_projection: bool = False,
#     bias_low_threshold: float = 2.5,
#     bias_high_threshold: float = 4.0,
#     bias_direct_high_threshold: float = 3.5,
#     bias_n_directions: int = 1,
#     bias_min_group_size: int = 5,
#     bias_positive_layers: Optional[set] = None,
# ) -> Dict[str, Any]:
#     """Original RAIN Stage-1 behavior with the new multi-tag segment boundaries."""
#     print("🚀 Starting original-behavior null-space projected task vector computation...")
#     rng = random.Random(seed)

#     d_model = model_target.config.hidden_size
#     n_heads = model_target.config.num_attention_heads
#     head_dim = d_model // n_heads
#     kv_heads = getattr(model_target.config, "num_key_value_heads", n_heads)

#     print(f"📋 Config: d_model={d_model}, n_heads={n_heads}, kv_heads={kv_heads}")
#     print(f"Feature extraction: {'Hook-based (recommended)' if use_hooks else 'Original method'}")

#     prepped_samples = prepare_samples_unified(
#         texts_R, tokenizer, neigh_radius, merge_types,
#         q_rows_per_text, k_rows_per_text, v_rows_per_text, o_rows_per_text, ffn_rows_per_text, rng
#     )

#     qk_device, vo_device, ffn_device = resolve_compute_devices(
#         qk_device, vo_device, ffn_device
#     )
#     print(f"🔧 Device mapping: QK={qk_device}, VO={vo_device}, FFN={ffn_device}")

#     merge_q = "q" in merge_types.lower()
#     merge_k = "k" in merge_types.lower()
#     merge_v = "v" in merge_types.lower()
#     merge_o = "o" in merge_types.lower()
#     merge_f = "f" in merge_types.lower()
#     print(f"🎯 Merge types: {merge_types.upper()} (Q={merge_q}, K={merge_k}, V={merge_v}, O={merge_o}, F={merge_f})")

#     print("\n🎯 Extracting raw task vectors...")
#     all_layer_task_vectors_raw = {}
#     for li in tqdm(selected_layers, desc="Extract task vectors for all layers"):
#         all_layer_task_vectors_raw[li] = task_vectors_single_layer_unified(
#             model_base, model_instruct, li, selected_heads, merge_types, scaling_factor=1.0
#         )

#     print("\n🔬 Computing null-space projection...")
#     projected_task_vectors = {"qk": {}, "vo": {}, "ffn": {}}
#     projection_stats = {
#         "total_cg_iterations": 0,
#         "total_constraint_residual": 0.0,
#         "layer_stats": {},
#     }
#     constraint_retention_per_layer: Dict[str, Dict[str, Any]] = {}

#     model_device = resolve_model_device(qk_device, vo_device, ffn_device)
#     print(f"🔧 Loading model for constraint construction on {model_device}...")
#     model_R_shared = load_materialized_model(
#         model_target.config._name_or_path,
#         torch.bfloat16,
#         model_device,
#     )

#     # texts_bias is the (possibly larger) set used for bias forward passes
#     _texts_for_bias = texts_bias if texts_bias is not None else texts_R

#     # Derive per-sample bias bin labels from raw_samples
#     _bias_bin_labels: Optional[List[str]] = None
#     _bias_positive_layers = bias_positive_layers if bias_positive_layers is not None else {9, 10, 11, 12, 28, 32, 33, 34, 35}
#     _section_positions_for_bias: Optional[List[Dict]] = None
#     if not skip_bias_projection and raw_samples is not None:
#         _bias_bin_labels = []
#         for s in raw_samples:
#             if not isinstance(s, dict):
#                 _bias_bin_labels.append("mid")
#                 continue
#             try:
#                 gt = float(s["gt"])
#                 dr = float(s["direct_rating"])
#             except (KeyError, TypeError, ValueError):
#                 _bias_bin_labels.append("mid")
#                 continue
#             if gt <= bias_low_threshold and dr >= bias_direct_high_threshold:
#                 _bias_bin_labels.append("low")
#             elif gt >= bias_high_threshold and dr >= bias_direct_high_threshold:
#                 _bias_bin_labels.append("high")
#             else:
#                 _bias_bin_labels.append("mid")
#         print(f"ℹ️  Bias calibration: D_low={_bias_bin_labels.count('low')}, D_high={_bias_bin_labels.count('high')}, D_mid={_bias_bin_labels.count('mid')}")
#         # Extract section positions once using the bias text set
#         _section_positions_for_bias = extract_all_section_positions(_texts_for_bias, tokenizer, max_interior_samples=1)

#     for li_idx, li in enumerate(tqdm(selected_layers, desc="Projection per layer")):
#         print(f"\n🔄 Processing layer {li} ({li_idx+1}/{len(selected_layers)})")
#         print(f"  📐 Building constraints for layer {li}...")
#         layer_cons = build_constraints_single_layer_unified(
#             model_R_shared, prepped_samples, li, selected_heads, merge_types,
#             w_q, w_k, q_rows_per_text, k_rows_per_text,
#             w_v, w_o, v_rows_per_text, o_rows_per_text,
#             w_ffn, ffn_rows_per_text, readout_dirs,
#             qk_device, vo_device, ffn_device, compute_dtype, use_hooks, max_seq_len,
#             constraint_sv_threshold=constraint_sv_threshold,
#         )

#         boundary_ret = layer_cons.get("_boundary_retention", {})
#         _log_retention(li, "boundary", boundary_ret, selected_heads)
#         constraint_retention_per_layer[str(li)] = {"boundary": boundary_ret}

#         if li not in all_layer_task_vectors_raw:
#             continue

#         layer_task_raw = all_layer_task_vectors_raw[li]
#         layer_stats = {"heads": {}}

#         if merge_q or merge_k:
#             projected_task_vectors["qk"][li] = {}
#         if merge_v or merge_o:
#             projected_task_vectors["vo"][li] = {}
#         if merge_f:
#             projected_task_vectors["ffn"][li] = {}

#         for h in tqdm(selected_heads, desc=f"Per-head projection for layer {li}", leave=False):
#             head_stat = {
#                 "constraints_qk": 0, "constraints_v": 0, "constraints_o": 0,
#                 "residual_norm_qk": 0.0, "residual_norm_v": 0.0, "residual_norm_o": 0.0,
#                 "cg_iterations": 0,
#             }

#             if (merge_q or merge_k) and h in layer_task_raw.get("qk", {}):
#                 cons_qk = layer_cons.get("qk", {}).get(h, {})
#                 task_qk = layer_task_raw["qk"][h]
#                 projected_task_vectors["qk"][li][h] = {}

#                 if merge_q and "dQ" in task_qk:
#                     dQ_proj, info_q = _solve_projection("q", cons_qk, task_qk["dQ"], lambda_ridge, cg_maxit, cg_tol, qk_device, compute_dtype)
#                     projected_task_vectors["qk"][li][h]["dQ_proj"] = dQ_proj.cpu()
#                     head_stat["residual_norm_qk"] += info_q.get("residual_norm", 0.0)
#                     head_stat["cg_iterations"] += info_q.get("iterations", 0)

#                 if merge_k and "dK" in task_qk:
#                     dK_proj, info_k = _solve_projection("k", cons_qk, task_qk["dK"], lambda_ridge, cg_maxit, cg_tol, qk_device, compute_dtype)
#                     projected_task_vectors["qk"][li][h]["dK_proj"] = dK_proj.cpu()
#                     head_stat["residual_norm_qk"] += info_k.get("residual_norm", 0.0)
#                     head_stat["cg_iterations"] += info_k.get("iterations", 0)

#                 head_stat["constraints_qk"] = int(
#                     cons_qk.get("Xi_q", torch.empty(0)).shape[0] +
#                     cons_qk.get("Xj_k", torch.empty(0)).shape[0]
#                 )

#             if (merge_v or merge_o) and h in layer_task_raw.get("vo", {}):
#                 cons_vo = layer_cons.get("vo", {}).get(h, {})
#                 task_vo = layer_task_raw["vo"][h]
#                 projected_task_vectors["vo"][li][h] = {}

#                 if merge_v and "dV" in task_vo:
#                     dV_proj, info_v = _solve_projection("v", cons_vo, task_vo["dV"], lambda_ridge, cg_maxit, cg_tol, vo_device, compute_dtype)
#                     projected_task_vectors["vo"][li][h]["dV_proj"] = dV_proj.cpu()
#                     head_stat["residual_norm_v"] += info_v.get("residual_norm", 0.0)
#                     head_stat["cg_iterations"] += info_v.get("iterations", 0)
#                     head_stat["constraints_v"] = int(cons_vo.get("Xi_v", torch.empty(0)).shape[0])

#                 if merge_o and "dO" in task_vo:
#                     dO_proj, info_o = _solve_projection("o", cons_vo, task_vo["dO"], lambda_ridge, cg_maxit, cg_tol, vo_device, compute_dtype)
#                     projected_task_vectors["vo"][li][h]["dO_proj"] = dO_proj.cpu()
#                     head_stat["residual_norm_o"] += info_o.get("residual_norm", 0.0)
#                     head_stat["cg_iterations"] += info_o.get("iterations", 0)
#                     head_stat["constraints_o"] = int(cons_vo.get("c_vec", torch.empty(0)).shape[0])

#             layer_stats["heads"][h] = head_stat
#             projection_stats["total_cg_iterations"] += head_stat["cg_iterations"]
#             projection_stats["total_constraint_residual"] += (
#                 head_stat["residual_norm_qk"] +
#                 head_stat["residual_norm_v"] +
#                 head_stat["residual_norm_o"]
#             )

#         if merge_f and "ffn" in layer_task_raw:
#             cons_ffn = layer_cons.get("ffn", {})
#             task_ffn = layer_task_raw["ffn"]
#             layer_stats["ffn"] = {
#                 "constraints_gate": int(cons_ffn.get("X_gate", torch.empty(0)).shape[0]),
#                 "constraints_up": int(cons_ffn.get("X_up", torch.empty(0)).shape[0]),
#                 "constraints_down": int(cons_ffn.get("H", torch.empty(0)).shape[0]),
#                 "residual_norm_gate": 0.0,
#                 "residual_norm_up": 0.0,
#                 "residual_norm_down": 0.0,
#                 "cg_iterations": 0,
#             }

#             if "dGate" in task_ffn:
#                 dGate_proj, info_gate = _solve_projection("ffn_gate", cons_ffn, task_ffn["dGate"], lambda_ridge, cg_maxit, cg_tol, ffn_device, compute_dtype)
#                 projected_task_vectors["ffn"][li]["dGate_proj"] = dGate_proj.cpu()
#                 layer_stats["ffn"]["residual_norm_gate"] = info_gate.get("residual_norm", 0.0)
#                 layer_stats["ffn"]["cg_iterations"] += info_gate.get("iterations", 0)

#             if "dUp" in task_ffn:
#                 dUp_proj, info_up = _solve_projection("ffn_up", cons_ffn, task_ffn["dUp"], lambda_ridge, cg_maxit, cg_tol, ffn_device, compute_dtype)
#                 projected_task_vectors["ffn"][li]["dUp_proj"] = dUp_proj.cpu()
#                 layer_stats["ffn"]["residual_norm_up"] = info_up.get("residual_norm", 0.0)
#                 layer_stats["ffn"]["cg_iterations"] += info_up.get("iterations", 0)

#             if "dDown_T" in task_ffn:
#                 dDown_proj, info_down = _solve_projection("ffn_down", cons_ffn, task_ffn["dDown_T"], lambda_ridge, cg_maxit, cg_tol, ffn_device, compute_dtype)
#                 projected_task_vectors["ffn"][li]["dDown_T_proj"] = dDown_proj.cpu()
#                 layer_stats["ffn"]["residual_norm_down"] = info_down.get("residual_norm", 0.0)
#                 layer_stats["ffn"]["cg_iterations"] += info_down.get("iterations", 0)

#             projection_stats["total_constraint_residual"] += (
#                 layer_stats["ffn"]["residual_norm_gate"] +
#                 layer_stats["ffn"]["residual_norm_up"] +
#                 layer_stats["ffn"]["residual_norm_down"]
#             )
#             projection_stats["total_cg_iterations"] += layer_stats["ffn"]["cg_iterations"]

#         projection_stats["layer_stats"][li] = layer_stats

#         # Bias projection — collect hidden states at </match> during a single lightweight pass
#         if _bias_bin_labels is not None and li in _bias_positive_layers and _section_positions_for_bias is not None:
#             interior_bias = build_match_interior_feature_matrix(
#                 model=model_R_shared,
#                 tokenizer=tokenizer,
#                 full_calibration_texts=_texts_for_bias,
#                 section_positions_per_sample=_section_positions_for_bias,
#                 layer_index=li,
#                 selected_heads=selected_heads,
#                 merge_types=merge_types,
#                 device=model_device,
#                 compute_dtype=compute_dtype,
#                 constraint_sv_threshold=constraint_sv_threshold,
#                 bin_labels=_bias_bin_labels,
#             )
#             bias_direction = _bias_direction_from_vecs(
#                 low_vecs=interior_bias.get("bias_vecs_low", []),
#                 high_vecs=interior_bias.get("bias_vecs_high", []),
#                 n_directions=bias_n_directions,
#                 min_group_size=bias_min_group_size,
#                 layer_idx=li,
#             )
#             if bias_direction is not None:
#                 _bias_apply_to_deltas(
#                     projected_task_vectors=projected_task_vectors,
#                     bias_directions={li: bias_direction},
#                     n_bias_directions=bias_n_directions,
#                     positive_tallrec_layers={li},
#                 )

#         del layer_cons
#         cleanup_memory()
#         print(f"  🧹 Cleared constraints for layer {li}, freed VRAM")

#     del model_R_shared
#     cleanup_memory()
#     print("🧹 Released shared constraint model")

#     print("\n✅ Null-space projection finished!")
#     print("  📊 Totals:")
#     print(f"     - Total CG iterations: {projection_stats['total_cg_iterations']}")
#     print(f"     - Sum of constraint residuals: {projection_stats['total_constraint_residual']:.6f}")

#     return {
#         "projected_task_vectors": projected_task_vectors,
#         "projection_stats": projection_stats,
#         "config": {
#             "merge_types": merge_types,
#             "selected_layers": selected_layers,
#             "selected_heads": selected_heads,
#             "d_model": d_model,
#             "n_heads": n_heads,
#             "head_dim": head_dim,
#             "kv_heads": kv_heads,
#             "compute_dtype": str(compute_dtype),
#             "lambda_ridge": lambda_ridge,
#             "cg_maxit": cg_maxit,
#             "cg_tol": cg_tol,
#             "projection_format": "legacy_original",
#             "constraint_sv_threshold": constraint_sv_threshold,
#             "constraint_retention_per_layer": constraint_retention_per_layer,
#             "layer_lambda_max": {str(layer): 1.0 for layer in selected_layers},
#             "active_layers": selected_layers,
#             "lambda_search_done": False,
#             "lambda_search_skipped_reason": "restored_original_rain_behavior",
#             "gamma_per_layer": {str(layer): 0.0 for layer in selected_layers},
#             "gamma_search_skipped_reason": "restored_original_rain_behavior",
#         },
#     }


# # ---------------------------------------------------------------------------
# # Raw task vectors + bias projection (no nullspace mode)
# # ---------------------------------------------------------------------------

# def compute_raw_task_vectors_with_bias(
#     model_base, model_instruct, model_target,
#     texts_R: List[str], tokenizer,
#     selected_layers: List[int], selected_heads: List[int],
#     merge_types: str = "qkvof",
#     compute_dtype: torch.dtype = torch.float32,
#     qk_device: str = "auto", vo_device: str = "auto", ffn_device: str = "auto",
#     max_seq_len: int = 7168,
#     constraint_sv_threshold: float = 1e-2,
#     raw_samples: Optional[List[Any]] = None,
#     skip_bias_projection: bool = False,
#     bias_low_threshold: float = 2.5,
#     bias_high_threshold: float = 4.0,
#     bias_direct_high_threshold: float = 3.5,
#     bias_n_directions: int = 1,
#     bias_min_group_size: int = 5,
#     bias_positive_layers: Optional[set] = None,
# ) -> Dict[str, Any]:
#     """Store raw task vectors (instruct - base) then apply bias projection. No nullspace solve."""
#     print("🚀 Starting raw task vector computation (no nullspace projection)...")

#     d_model = model_target.config.hidden_size
#     n_heads = model_target.config.num_attention_heads
#     head_dim = d_model // n_heads
#     kv_heads = getattr(model_target.config, "num_key_value_heads", n_heads)
#     print(f"📋 Config: d_model={d_model}, n_heads={n_heads}, kv_heads={kv_heads}")

#     merge_q = "q" in merge_types.lower()
#     merge_k = "k" in merge_types.lower()
#     merge_v = "v" in merge_types.lower()
#     merge_o = "o" in merge_types.lower()
#     merge_f = "f" in merge_types.lower()

#     qk_device, vo_device, ffn_device = resolve_compute_devices(qk_device, vo_device, ffn_device)
#     model_device = resolve_model_device(qk_device, vo_device, ffn_device)

#     print("\n🎯 Extracting raw task vectors (instruct - base)...")
#     projected_task_vectors = {"qk": {}, "vo": {}, "ffn": {}}

#     for li in tqdm(selected_layers, desc="Extract task vectors"):
#         tv = task_vectors_single_layer_unified(
#             model_base, model_instruct, li, selected_heads, merge_types, scaling_factor=1.0
#         )
#         if merge_q or merge_k:
#             projected_task_vectors["qk"][li] = {}
#         if merge_v or merge_o:
#             projected_task_vectors["vo"][li] = {}
#         if merge_f:
#             projected_task_vectors["ffn"][li] = {}

#         for h in selected_heads:
#             if (merge_q or merge_k) and h in tv.get("qk", {}):
#                 projected_task_vectors["qk"][li][h] = {}
#                 if merge_q and "dQ" in tv["qk"][h]:
#                     projected_task_vectors["qk"][li][h]["dQ_proj"] = tv["qk"][h]["dQ"].cpu()
#                 if merge_k and "dK" in tv["qk"][h]:
#                     projected_task_vectors["qk"][li][h]["dK_proj"] = tv["qk"][h]["dK"].cpu()
#             if (merge_v or merge_o) and h in tv.get("vo", {}):
#                 projected_task_vectors["vo"][li][h] = {}
#                 if merge_v and "dV" in tv["vo"][h]:
#                     projected_task_vectors["vo"][li][h]["dV_proj"] = tv["vo"][h]["dV"].cpu()
#                 if merge_o and "dO" in tv["vo"][h]:
#                     projected_task_vectors["vo"][li][h]["dO_proj"] = tv["vo"][h]["dO"].cpu()
#         if merge_f and "ffn" in tv:
#             if "dGate" in tv["ffn"]:
#                 projected_task_vectors["ffn"][li]["dGate_proj"] = tv["ffn"]["dGate"].cpu()
#             if "dUp" in tv["ffn"]:
#                 projected_task_vectors["ffn"][li]["dUp_proj"] = tv["ffn"]["dUp"].cpu()
#             if "dDown_T" in tv["ffn"]:
#                 projected_task_vectors["ffn"][li]["dDown_T_proj"] = tv["ffn"]["dDown_T"].cpu()

#     # Bias projection — same logic as in nullspace mode
#     _bias_positive_layers = bias_positive_layers if bias_positive_layers is not None else {9, 10, 11, 12, 28, 32, 33, 34, 35}
#     if not skip_bias_projection and raw_samples is not None:
#         _bias_bin_labels: List[str] = []
#         for s in raw_samples:
#             if not isinstance(s, dict):
#                 _bias_bin_labels.append("mid")
#                 continue
#             try:
#                 gt = float(s["gt"])
#                 dr = float(s["direct_rating"])
#             except (KeyError, TypeError, ValueError):
#                 _bias_bin_labels.append("mid")
#                 continue
#             if gt <= bias_low_threshold and dr >= bias_direct_high_threshold:
#                 _bias_bin_labels.append("low")
#             elif gt >= bias_high_threshold and dr >= bias_direct_high_threshold:
#                 _bias_bin_labels.append("high")
#             else:
#                 _bias_bin_labels.append("mid")
#         print(f"ℹ️  Bias calibration: D_low={_bias_bin_labels.count('low')}, D_high={_bias_bin_labels.count('high')}, D_mid={_bias_bin_labels.count('mid')}")

#         section_positions = extract_all_section_positions(texts_R, tokenizer, max_interior_samples=1)

#         print(f"🔧 Loading model for bias forward passes on {model_device}...")
#         model_R_shared = load_materialized_model(
#             model_target.config._name_or_path, torch.bfloat16, model_device,
#         )
#         for li in tqdm(selected_layers, desc="Bias projection per layer"):
#             if li not in _bias_positive_layers:
#                 continue
#             interior_bias = build_match_interior_feature_matrix(
#                 model=model_R_shared, tokenizer=tokenizer,
#                 full_calibration_texts=texts_R,
#                 section_positions_per_sample=section_positions,
#                 layer_index=li, selected_heads=selected_heads,
#                 merge_types=merge_types, device=model_device,
#                 compute_dtype=compute_dtype,
#                 constraint_sv_threshold=constraint_sv_threshold,
#                 bin_labels=_bias_bin_labels,
#             )
#             bias_direction = _bias_direction_from_vecs(
#                 low_vecs=interior_bias.get("bias_vecs_low", []),
#                 high_vecs=interior_bias.get("bias_vecs_high", []),
#                 n_directions=bias_n_directions,
#                 min_group_size=bias_min_group_size,
#                 layer_idx=li,
#             )
#             if bias_direction is not None:
#                 _bias_apply_to_deltas(
#                     projected_task_vectors=projected_task_vectors,
#                     bias_directions={li: bias_direction},
#                     n_bias_directions=bias_n_directions,
#                     positive_tallrec_layers={li},
#                 )
#             cleanup_memory()

#         del model_R_shared
#         cleanup_memory()
#         print("🧹 Released bias model")

#     print("\n✅ Raw task vectors ready!")
#     return {
#         "projected_task_vectors": projected_task_vectors,
#         "projection_stats": {"total_cg_iterations": 0, "total_constraint_residual": 0.0, "layer_stats": {}},
#         "config": {
#             "merge_types": merge_types,
#             "selected_layers": selected_layers,
#             "selected_heads": selected_heads,
#             "d_model": d_model,
#             "n_heads": n_heads,
#             "head_dim": head_dim,
#             "kv_heads": kv_heads,
#             "compute_dtype": str(compute_dtype),
#             "projection_format": "raw_no_nullspace",
#             "layer_lambda_max": {str(layer): 1.0 for layer in selected_layers},
#             "active_layers": selected_layers,
#         },
#     }


# # ---------------------------------------------------------------------------
# # Bias projection (integrated from step0b_bias_projection.py)
# # ---------------------------------------------------------------------------

# def _bias_direction_from_vecs(
#     low_vecs: List[torch.Tensor],
#     high_vecs: List[torch.Tensor],
#     n_directions: int,
#     min_group_size: int,
#     layer_idx: int,
# ) -> Optional[torch.Tensor]:
#     """Compute bias direction from accumulated hidden-state vectors. No model needed."""
#     if len(low_vecs) < min_group_size or len(high_vecs) < min_group_size:
#         print(f"  Layer {layer_idx}: bias skip — D_low={len(low_vecs)}, D_high={len(high_vecs)} < min={min_group_size}")
#         return None

#     L = torch.stack(low_vecs)
#     H = torch.stack(high_vecs)

#     if n_directions == 1:
#         raw_dir = H.mean(dim=0) - L.mean(dim=0)
#         norm = raw_dir.norm()
#         d_hat = raw_dir / norm.clamp(min=1e-8)
#         gap = (H @ d_hat).mean().item() - (L @ d_hat).mean().item()
#         print(f"  Layer {layer_idx}: bias dir_norm={norm:.4f}, proj_gap={gap:.4f} (D_low={len(low_vecs)}, D_high={len(high_vecs)})")
#         if gap < 0.05:
#             print(f"  Layer {layer_idx}: WARNING — bias projection gap {gap:.4f} < 0.05")
#         return d_hat
#     else:
#         k = n_directions
#         D_mat = H - L.mean(dim=0, keepdim=True)
#         _, S, Vh = torch.linalg.svd(D_mat, full_matrices=False)
#         B = Vh[:k].T
#         gap = (H @ B).norm(dim=1).mean().item() - (L @ B).norm(dim=1).mean().item()
#         print(f"  Layer {layer_idx}: bias top-{k} SVs={S[:k].tolist()}, proj_gap={gap:.4f}")
#         return B


# def save_projected_task_vectors(projected_data: Dict[str, Any], output_path: str):
#     """Save the projected task vectors to file"""
#     print(f"💾 Saving projections to: {output_path}")
    
#     # Ensure directory exists
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     # Use pickle (supports torch tensors)
#     with open(output_path, 'wb') as f:
#         pickle.dump(projected_data, f)
    
#     # Also save a JSON config for quick inspection
#     def _json_safe(value):
#         if isinstance(value, torch.Tensor):
#             return {
#                 "__tensor__": True,
#                 "shape": list(value.shape),
#                 "dtype": str(value.dtype),
#             }
#         if isinstance(value, dict):
#             return {str(k): _json_safe(v) for k, v in value.items()}
#         if isinstance(value, list):
#             return [_json_safe(v) for v in value]
#         return value

#     config_path = output_path.replace('.pkl', '_config.json')
#     with open(config_path, 'w', encoding='utf-8') as f:
#         config_data = projected_data["config"].copy()
#         config_data["stats"] = {
#             "total_cg_iterations": projected_data["projection_stats"]["total_cg_iterations"],
#             "total_constraint_residual": projected_data["projection_stats"]["total_constraint_residual"]
#         }
#         json.dump(_json_safe(config_data), f, ensure_ascii=False, indent=2)
    
#     # Print file size
#     file_size = os.path.getsize(output_path) / 1024 / 1024
#     print(f"✅ Saved: {output_path} ({file_size:.1f} MB)")
#     print(f"📋 Config info: {config_path}")


# def _stratified_sample(
#     samples: List[Any],
#     max_n: int,
#     low_threshold: float,
#     high_threshold: float,
#     direct_high_threshold: float,
#     seed: int = 42,
# ) -> List[Any]:
#     """Return exactly max_n low + max_n high samples (no mid). Shuffled."""
#     low, high = [], []
#     for s in samples:
#         if not isinstance(s, dict):
#             continue
#         try:
#             gt = float(s["gt"])
#             dr = float(s["direct_rating"])
#         except (KeyError, TypeError, ValueError):
#             continue
#         if gt <= low_threshold and dr >= direct_high_threshold:
#             low.append(s)
#         elif gt >= high_threshold and dr >= direct_high_threshold:
#             high.append(s)

#     rng = random.Random(seed)
#     rng.shuffle(low)
#     rng.shuffle(high)

#     low = low[:max_n]
#     high = high[:max_n]

#     if len(low) < max_n:
#         print(f"⚠️  Only {len(low)} low-bin samples available (requested {max_n})")
#     if len(high) < max_n:
#         print(f"⚠️  Only {len(high)} high-bin samples available (requested {max_n})")

#     selected = low + high
#     rng.shuffle(selected)
#     print(f"📊 Stratified sample: low={len(low)}, high={len(high)}, total={len(selected)}")
#     return selected


# def main():
#     parser = argparse.ArgumentParser(description="Stage 1: compute and save projected task vectors")
    
#     # Mode
#     parser.add_argument(
#         "--mode", type=str, choices=["nullspace", "raw"], default="nullspace",
#         help="nullspace: full null-space projection + bias (default); raw: raw task vectors + bias only (no CG solve)",
#     )

#     # Base paths
#     parser.add_argument("--base", type=str,
#                        default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B")
#     parser.add_argument("--instruct", type=str,
#                        default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B-Instruct")
#     parser.add_argument("--target", type=str,
#                        default="/opt/data/private/hzhcode/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
#     # Data & constraint params
#     parser.add_argument("--texts_r", type=str, required=True, help="Path to JSON sample file")
#     parser.add_argument("--max_samples_r", type=int, default=10, help="Max samples per bin for bias projection (low + high)")
#     parser.add_argument("--constraint_samples", type=int, default=None,
#                         help="Max samples to use for nullspace constraint construction (default: same as max_samples_r)")
#     parser.add_argument("--neigh_radius", type=int, default=5, help="Boundary neighborhood radius")
    
#     # Layer & head config
#     parser.add_argument("--layers_tail", type=int, default=2, help="Process the last N layers")
#     parser.add_argument("--heads", type=str, default="all", help="Heads to process ('all' or comma-separated indices)")
    
#     # Weights & solver params
#     parser.add_argument("--lambda_ridge", type=float, default=1e-4, help="Ridge parameter (λ)")
#     parser.add_argument("--cg_maxit", type=int, default=100, help="Max CG iterations")
#     parser.add_argument("--cg_tol", type=float, default=1e-5, help="CG convergence tolerance")
    
#     # Compute config
#     parser.add_argument("--compute_precision", type=str, choices=["fp32", "fp64"], default="fp32",
#                        help="Computation precision")
    
#     # Merge types
#     parser.add_argument("--merge_types", type=str, default="qk", 
#                        help="Merge types: a combination of q/k/v/o/f")
    
#     # QK params
#     parser.add_argument("--q_rows_per_text", type=int, default=8, help="Q constraint rows per text")
#     parser.add_argument("--k_rows_per_text", type=int, default=8, help="K constraint rows per text")
#     parser.add_argument("--w_q", type=float, default=1.0, help="Weight for Q constraints")
#     parser.add_argument("--w_k", type=float, default=1.0, help="Weight for K constraints")
    
#     # VO params
#     parser.add_argument("--v_rows_per_text", type=int, default=4, help="V constraint target positions per text")
#     parser.add_argument("--o_rows_per_text", type=int, default=4, help="O constraint target positions per text")
#     parser.add_argument("--w_v", type=float, default=1.0, help="Weight for V constraints")
#     parser.add_argument("--w_o", type=float, default=1.0, help="Weight for O constraints")
    
#     # FFN params
#     parser.add_argument("--ffn_rows_per_text", type=int, default=4, help="FFN-Down constraint target positions per text")
#     parser.add_argument("--readout_dirs", type=int, default=2, help="Number of output readout directions c per head/layer")
#     parser.add_argument("--w_ffn", type=float, default=1.0, help="Weight for FFN-Down constraints")
    
#     # Multi-device config
#     parser.add_argument("--qk_device", type=str, default="auto",
#                        help="Computation device for QK constraints")
#     parser.add_argument("--vo_device", type=str, default="auto",
#                        help="Computation device for VO constraints")
#     parser.add_argument("--ffn_device", type=str, default="auto",
#                        help="Computation device for FFN constraints")
    
#     # Hook config
#     parser.add_argument("--use_hooks", action="store_true", default=True,
#                        help="Use hooks to capture exact layer internals (recommended)")
#     parser.add_argument("--no_hooks", action="store_true",
#                        help="Disable hooks and use the legacy extraction method")
    
#     # Sequence length limit
#     parser.add_argument("--max_seq_len", type=int, default=7168,
#                        help="Max sequence length (BF16-optimized attention, default 7168; BF16 halves memory usage)")

#     parser.add_argument(
#         "--skip_lambda_search",
#         action="store_true",
#         help="Skip lambda search; use lambda=1.0 for all layers"
#     )
#     parser.add_argument(
#         "--lambda_candidates",
#         type=str,
#         default="1.5,1.2,1.0,0.8,0.5,0.3",
#         help="Comma-separated lambda candidates, tried high to low"
#     )
#     parser.add_argument(
#         "--margin_retention_threshold",
#         type=float,
#         default=0.80,
#         help="Min fraction of baseline closing-tag margin to retain"
#     )
#     parser.add_argument(
#         "--lambda_search_max_new_tokens",
#         type=int,
#         default=512,
#         help="Max new tokens for generation in lambda search"
#     )
#     parser.add_argument(
#         "--lambda_generation_batch_size",
#         type=int,
#         default=4,
#         help="Batch size for lambda-search generation validation"
#     )
#     parser.add_argument(
#         "--lambda_margin_batch_size",
#         type=int,
#         default=4,
#         help="Batch size for lambda-search margin measurement"
#     )
#     # Constraint SV threshold
#     parser.add_argument(
#         "--constraint_sv_threshold",
#         type=float,
#         default=1e-2,
#         help="Singular value threshold tau for constraint row filtering (AlphaEdit-style). "
#              "Rows with sigma_i/sigma_max <= tau are discarded. Default: 1e-2",
#     )

#     # Bias projection — runs on the same --texts_r samples, no extra files needed
#     parser.add_argument("--skip_bias_projection", action="store_true",
#                         help="Skip bias projection step (runs by default)")
#     parser.add_argument("--bias_low_rating_threshold", type=float, default=2.5,
#                         help="gt <= this → 'low' bin for bias direction (default: 2.5)")
#     parser.add_argument("--bias_high_rating_threshold", type=float, default=4.0,
#                         help="gt >= this → 'high' bin for bias direction (default: 4.0)")
#     parser.add_argument("--bias_direct_high_threshold", type=float, default=3.5,
#                         help="direct_rating >= this required for low/high binning (default: 3.5)")
#     parser.add_argument("--bias_n_directions", type=int, default=1,
#                         help="Number of bias directions to remove per layer (default: 1)")
#     parser.add_argument("--bias_min_group_size", type=int, default=5,
#                         help="Min examples per bin to compute a direction (default: 5)")
#     parser.add_argument("--bias_positive_layers", type=int, nargs="*", default=None,
#                         help="Layers to apply bias projection to (default: {9,10,11,12,28,32,33,34,35})")

#     # Output config
#     parser.add_argument("--output_file", type=str, required=True,
#                        help="Output file path (*.pkl)")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
#     args = parser.parse_args()

#     random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     # Precision
#     compute_dtype = torch.float64 if args.compute_precision == "fp64" else torch.float32
    
#     print("🚀 Null-space projection - Stage 1")
#     print("=" * 70)
#     print(f"Base: {args.base}")
#     print(f"Instruct: {args.instruct}")
#     print(f"Target: {args.target}")
#     print(f"Output file: {args.output_file}")
#     print(f"Precision: {args.compute_precision.upper()}")
#     print(f"Merge types: {args.merge_types.upper()}")

#     # Hook method selection
#     use_hooks = args.use_hooks and not args.no_hooks
#     print(f"Feature extraction method: {'Hook-based (recommended)' if use_hooks else 'Legacy method'}")

#     start_time = time.time()

#     # Load models (on CPU)
#     print("\n📥 Loading models on CPU...")
#     model_base = AutoModelForCausalLM.from_pretrained(
#         args.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
#     ).eval()
    
#     model_instruct = AutoModelForCausalLM.from_pretrained(
#         args.instruct, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
#     ).eval()
    
#     model_target = AutoModelForCausalLM.from_pretrained(
#         args.target, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
#     ).eval()
    
#     tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True, trust_remote_code=True)

#     # Config
#     num_layers = model_target.config.num_hidden_layers
#     n_heads = model_target.config.num_attention_heads

#     selected_layers = list(range(num_layers - args.layers_tail, num_layers))
#     if args.heads == "all":
#         selected_heads = list(range(n_heads))
#     else:
#         selected_heads = [int(x) for x in args.heads.split(",")]

#     print(f"📋 Run config:")
#     print(f"  Layers: {selected_layers}")
#     print(f"  Heads: {len(selected_heads)}/{n_heads}")

#     # Read data — two stratified samples:
#     #   raw_samples / texts_R      → bias projection  (max_samples_r per bin)
#     #   constraint_samples / texts_constraint → nullspace constraints (constraint_samples per bin, faster)
#     _all_samples = load_samples_data(args.texts_r)

#     n_bias = args.max_samples_r
#     n_constraint = args.constraint_samples if args.constraint_samples is not None else n_bias

#     def _build_texts(samples):
#         id_to_text = {}
#         for s in _all_samples:
#             if id(s) in {id(x) for x in samples}:
#                 prompt, reasoning, response = extract_sample_fields(s)
#                 id_to_text[id(s)] = format_prompt_for_chat(prompt, tokenizer) + build_segmented_output(reasoning, response)
#         return [id_to_text[id(s)] for s in samples]

#     print(f"📊 Bias sample pool  (max_samples_r={n_bias} per bin):")
#     raw_samples = _stratified_sample(
#         _all_samples, n_bias,
#         args.bias_low_rating_threshold, args.bias_high_rating_threshold,
#         args.bias_direct_high_threshold, args.seed,
#     )
#     texts_R = _build_texts(raw_samples)

#     if n_constraint != n_bias:
#         print(f"📊 Constraint sample pool (constraint_samples={n_constraint} per bin):")
#         constraint_raw = _stratified_sample(
#             _all_samples, n_constraint,
#             args.bias_low_rating_threshold, args.bias_high_rating_threshold,
#             args.bias_direct_high_threshold, args.seed,
#         )
#         texts_constraint = _build_texts(constraint_raw)
#     else:
#         constraint_raw = raw_samples
#         texts_constraint = texts_R

#     print(f"📊 Bias samples: {len(texts_R)}, Constraint samples: {len(texts_constraint)}")

#     if not args.skip_lambda_search:
#         print("ℹ️  Lambda search is disabled in original behavior.")

#     bias_kwargs = dict(
#         raw_samples=raw_samples,
#         skip_bias_projection=args.skip_bias_projection,
#         bias_low_threshold=args.bias_low_rating_threshold,
#         bias_high_threshold=args.bias_high_rating_threshold,
#         bias_direct_high_threshold=args.bias_direct_high_threshold,
#         bias_n_directions=args.bias_n_directions,
#         bias_min_group_size=args.bias_min_group_size,
#         bias_positive_layers=set(args.bias_positive_layers) if args.bias_positive_layers else None,
#     )

#     if args.mode == "raw":
#         print("\n🔬 Mode: raw task vectors + bias projection (no null-space solve)...")
#         projected_data = compute_raw_task_vectors_with_bias(
#             model_base, model_instruct, model_target,
#             texts_R, tokenizer,
#             selected_layers, selected_heads,
#             args.merge_types, compute_dtype,
#             args.qk_device, args.vo_device, args.ffn_device,
#             args.max_seq_len, args.constraint_sv_threshold,
#             **bias_kwargs,
#         )
#     else:
#         print(f"\n🔬 Mode: null-space projection + bias projection "
#               f"(constraint={len(texts_constraint)} samples, bias={len(texts_R)} samples)...")
#         projected_data = compute_nullspace_projections_original_behavior(
#             model_base, model_instruct, model_target,
#             texts_constraint, tokenizer,          # constraint build uses smaller set
#             selected_layers, selected_heads,
#             args.neigh_radius, args.lambda_ridge, args.cg_maxit, args.cg_tol,
#             compute_dtype, args.merge_types,
#             args.q_rows_per_text, args.k_rows_per_text, args.w_q, args.w_k,
#             args.v_rows_per_text, args.o_rows_per_text, args.w_v, args.w_o,
#             args.ffn_rows_per_text, args.w_ffn, args.readout_dirs, args.seed,
#             args.qk_device, args.vo_device, args.ffn_device,
#             use_hooks, args.max_seq_len, args.constraint_sv_threshold,
#             texts_bias=texts_R,                   # bias uses larger set
#             **bias_kwargs,
#         )

#     # Save results
#     end_time = time.time()

#     # Attach runtime info
#     projected_data["runtime_info"] = {
#         "runtime_seconds": end_time - start_time,
#         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "args": vars(args)
#     }
    
#     save_projected_task_vectors(projected_data, args.output_file)

#     print(f"\n✅ Null-space projection finished! Elapsed: {end_time - start_time:.1f}s")
#     print(f"📁 Output file: {args.output_file}")
#     print(f"🚀 Next: use scaling_model_merge.py to apply different scaling factors for model merging")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Null-space projection script - Stage 1
Purpose: compute and save projected task vectors without applying the scaling factor
Output: a file with projected task vectors to be used by later scaling
"""

import os
import json
import math
import argparse
import re
import random
import gc
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import core functions
from nullspace_merge_qkvo_ffn import (
    ensure_dir, cleanup_memory, print_memory_status,
    PreparedSample, prepare_samples_unified,
    build_constraints_single_layer_unified,
    task_vectors_single_layer_unified,
    cg_single_head_batched, cg_v, cg_o, cg_ffn_down, cg_ffn_gate, cg_ffn_up,
    # High-efficiency dense solvers
    ffn_down_dense_project, ffn_gate_dense_project, ffn_up_dense_project
)


def read_json_samples_robust(path: str, tokenizer, max_n: Optional[int] = None) -> List[str]:
    """Read samples and build full conversations. Supports reczero calibration format
    (prompt_long_native / raw_output_long_native) and generic prompt/reasoning/response dicts."""
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    # Support both JSON array and JSONL
    try:
        data = json.loads(raw_text)
        samples = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        samples = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    full_prompts = []
    for sample in samples:
        if max_n is not None and len(full_prompts) >= max_n:
            break

        if isinstance(sample, str):
            full_prompts.append(sample)
            continue

        if not isinstance(sample, dict):
            continue

        # Reczero calibration format: prompt already chat-templated, output pre-tagged
        if "prompt_long_native" in sample and "raw_output_long_native" in sample:
            prompt_part = sample["prompt_long_native"]
            output_part = sample["raw_output_long_native"]
            # Strip leading "markdown\n" artifact if present
            if output_part.startswith("markdown\n"):
                output_part = output_part[len("markdown\n"):]
            full_prompts.append(prompt_part + output_part)
            continue

        # Generic format: build chat template from prompt field
        prompt = sample.get("prompt", sample.get("text", ""))
        output = sample.get("raw_output_long_native", sample.get("reasoning", ""))
        response = sample.get("response", "")

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_prompts.append(formatted_prompt + output + response)

    return full_prompts


def compute_nullspace_projections(
    model_base, model_instruct, model_target,
    texts_R: List[str], tokenizer,
    selected_layers: List[int], selected_heads: List[int],
    neigh_radius: int, lambda_ridge: float, cg_maxit: int, cg_tol: float, 
    compute_dtype: torch.dtype = torch.float32,
    merge_types: str = "qk",
    # QK params
    q_rows_per_text: int = 8, k_rows_per_text: int = 8, w_q: float = 1.0, w_k: float = 1.0,
    # VO params
    v_rows_per_text: int = 4, o_rows_per_text: int = 4, w_v: float = 1.0, w_o: float = 1.0,
    # FFN params
    ffn_rows_per_text: int = 4, w_ffn: float = 1.0, readout_dirs: int = 2,
    seed: int = 42,
    # Multi-device config
    qk_device: str = "auto", vo_device: str = "auto", ffn_device: str = "auto",
    # Hook config
    use_hooks: bool = True,
    # Sequence length limit
    max_seq_len: int = 7168
) -> Dict[str, Any]:
    """Compute null-space projected task vectors (without applying the scaling factor)"""
    
    print("🚀 Starting computation of null-space projected task vectors...")
    rng = random.Random(seed)
    
    d_model = model_target.config.hidden_size
    n_heads = model_target.config.num_attention_heads
    head_dim = d_model // n_heads
    kv_heads = getattr(model_target.config, 'num_key_value_heads', n_heads)
    
    print(f"📋 Config: d_model={d_model}, n_heads={n_heads}, kv_heads={kv_heads}")
    print(f"Feature extraction: {'Hook-based (recommended)' if use_hooks else 'Original method'}")
    
    # 1) Preprocess samples
    prepped_samples = prepare_samples_unified(
        texts_R, tokenizer, neigh_radius, merge_types,
        q_rows_per_text, k_rows_per_text, v_rows_per_text, o_rows_per_text, ffn_rows_per_text, rng
    )
    
    # Device assignment
    if qk_device == "auto":
        qk_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if vo_device == "auto":
        vo_device = "cuda:1" if torch.cuda.device_count() > 1 else qk_device
    if ffn_device == "auto":
        ffn_device = "cuda:2" if torch.cuda.device_count() > 2 else vo_device
    
    print(f"🔧 Device mapping: QK={qk_device}, VO={vo_device}, FFN={ffn_device}")
    
    # Parse merge types
    merge_q = 'q' in merge_types.lower()
    merge_k = 'k' in merge_types.lower()
    merge_v = 'v' in merge_types.lower()
    merge_o = 'o' in merge_types.lower()
    merge_f = 'f' in merge_types.lower()
    
    print(f"🎯 Merge types: {merge_types.upper()} (Q={merge_q}, K={merge_k}, V={merge_v}, O={merge_o}, F={merge_f})")
    
    # 2) Extract raw task vectors (no scaling factor applied)
    print("\n🎯 Extracting raw task vectors...")
    all_layer_task_vectors_raw = {}
    for li in tqdm(selected_layers, desc="Extract task vectors for all layers"):
        layer_task_vectors = task_vectors_single_layer_unified(
            model_base, model_instruct, li, selected_heads, merge_types, scaling_factor=1.0
        )
        all_layer_task_vectors_raw[li] = layer_task_vectors
    
    # 3) Compute projected task vectors (process layer-by-layer to save VRAM)
    print("\n🔬 Computing null-space projection...")
    projected_task_vectors = {
        "qk": {},  # {layer: {head: {"dQ_proj": tensor, "dK_proj": tensor}}}
        "vo": {},  # {layer: {head: {"dV_proj": tensor, "dO_proj": tensor}}}
        "ffn": {}  # {layer: {"dGate_proj": tensor, "dUp_proj": tensor, "dDown_T_proj": tensor}}
    }
    
    projection_stats = {
        "total_cg_iterations": 0,
        "total_constraint_residual": 0.0,
        "layer_stats": {}
    }
    # Load the model once for all layers (avoids repeated disk I/O)
    print("🔧 Loading model for constraint construction (shared across all layers)...")
    model_R_shared = AutoModelForCausalLM.from_pretrained(
        model_target.config._name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    ).eval()

    for li_idx, li in enumerate(tqdm(selected_layers, desc="Projection per layer")):
        print(f"\n🔄 Processing layer {li} ({li_idx+1}/{len(selected_layers)})")

        # Build constraints for the current layer
        print(f"  📐 Building constraints for layer {li}...")
        layer_cons = build_constraints_single_layer_unified(
            model_R_shared, prepped_samples, li, selected_heads, merge_types,
            w_q, w_k, q_rows_per_text, k_rows_per_text,
            w_v, w_o, v_rows_per_text, o_rows_per_text,
            w_ffn, ffn_rows_per_text, readout_dirs,
            qk_device, vo_device, ffn_device, compute_dtype, use_hooks, max_seq_len
        )
        
        layer_stats = {"heads": {}}
        
        # Get task vectors for this layer
        if li not in all_layer_task_vectors_raw:
            continue
            
        layer_task_raw = all_layer_task_vectors_raw[li]
        
        # Initialize storage for this layer
        if merge_q or merge_k:
            projected_task_vectors["qk"][li] = {}
        if merge_v or merge_o:
            projected_task_vectors["vo"][li] = {}
        
        # Per-head CG projection
        for h in tqdm(selected_heads, desc=f"Per-head projection for layer {li}", leave=False):
            head_stat = {
                "constraints_qk": 0, "constraints_v": 0, "constraints_o": 0,
                "residual_norm_qk": 0.0, "residual_norm_v": 0.0, "residual_norm_o": 0.0,
                "cg_iterations": 0
            }
            
            # QK projection
            if (merge_q or merge_k) and "qk" in layer_cons and h in layer_cons["qk"]:
                cons_h_qk = layer_cons["qk"][h]
                total_constraints_qk = cons_h_qk["Xi_q"].shape[0] + cons_h_qk["Xj_k"].shape[0]
                head_stat["constraints_qk"] = total_constraints_qk
                
                if total_constraints_qk > 0 and h in layer_task_raw["qk"]:
                    task_qk = layer_task_raw["qk"][h]
                    task_dQ = task_qk.get("dQ") if merge_q and "dQ" in task_qk else None
                    task_dK = task_qk.get("dK") if merge_k and "dK" in task_qk else None
                    
                    if task_dQ is not None and task_dK is not None:
                        # CG projection solve
                        dQ_proj, dK_proj, cg_info = cg_single_head_batched(
                            cons_h_qk, task_dQ, task_dK, lambda_ridge, cg_maxit, cg_tol, 
                            device=qk_device, compute_dtype=compute_dtype
                        )
                        
                        # Save projected results
                        projected_task_vectors["qk"][li][h] = {}
                        if merge_q:
                            projected_task_vectors["qk"][li][h]["dQ_proj"] = dQ_proj.cpu()
                        if merge_k:
                            projected_task_vectors["qk"][li][h]["dK_proj"] = dK_proj.cpu()
                        
                        head_stat["residual_norm_qk"] = cg_info["residual_norm"]
                        head_stat["cg_iterations"] += cg_info["iterations"]
                        print(f"    ✅ QK projection done: {cg_info.get('solver', 'cg')}, "
                              f"residual={cg_info['residual_norm']:.2e}, time={cg_info.get('time', 0.0):.2f}s")
            
            # V projection
            if merge_v and "vo" in layer_cons and h in layer_cons["vo"]:
                cons_h_v = layer_cons["vo"][h]
                if "Xi_v" in cons_h_v and cons_h_v["Xi_v"].numel() > 0:
                    head_stat["constraints_v"] = cons_h_v["Xi_v"].shape[0]
                
                    if h in layer_task_raw["vo"] and "dV" in layer_task_raw["vo"][h]:
                        dV_task = layer_task_raw["vo"][h]["dV"]
                        dV_proj, info_v = cg_v(cons_h_v, dV_task, lambda_ridge, cg_maxit, cg_tol, 
                                             device=vo_device, compute_dtype=compute_dtype)
                        
                        # Save projected results
                        if li not in projected_task_vectors["vo"]:
                            projected_task_vectors["vo"][li] = {}
                        if h not in projected_task_vectors["vo"][li]:
                            projected_task_vectors["vo"][li][h] = {}
                        projected_task_vectors["vo"][li][h]["dV_proj"] = dV_proj.cpu()
                        
                        head_stat["residual_norm_v"] = info_v["residual_norm"]
                        head_stat["cg_iterations"] += info_v["iterations"]
                        print(f"    ✅ V projection done: {info_v.get('solver', 'cg')}, "
                              f"residual={info_v['residual_norm']:.2e}, time={info_v.get('time', 0.0):.2f}s")
            # O projection
            if merge_o and "vo" in layer_cons and h in layer_cons["vo"]:
                cons_h_o = layer_cons["vo"][h]
                if "c_vec" in cons_h_o and cons_h_o["c_vec"].numel() > 0:
                    head_stat["constraints_o"] = cons_h_o["c_vec"].shape[0]
                
                    if h in layer_task_raw["vo"] and "dO" in layer_task_raw["vo"][h]:
                        dO_task = layer_task_raw["vo"][h]["dO"]
                        dO_proj, info_o = cg_o(cons_h_o, dO_task, lambda_ridge, cg_maxit, cg_tol, 
                                             device=vo_device, compute_dtype=compute_dtype)
                        
                        # Save projected results
                        if li not in projected_task_vectors["vo"]:
                            projected_task_vectors["vo"][li] = {}
                        if h not in projected_task_vectors["vo"][li]:
                            projected_task_vectors["vo"][li][h] = {}
                        projected_task_vectors["vo"][li][h]["dO_proj"] = dO_proj.cpu()
                        
                        head_stat["residual_norm_o"] = info_o["residual_norm"]
                        head_stat["cg_iterations"] += info_o["iterations"]
                        print(f"    ✅ O projection done: {info_o.get('solver', 'cg')}, "
                              f"residual={info_o['residual_norm']:.2e}, time={info_o.get('time', 0.0):.2f}s")
            
            layer_stats["heads"][h] = head_stat
            projection_stats["total_cg_iterations"] += head_stat["cg_iterations"]
            projection_stats["total_constraint_residual"] += (head_stat["residual_norm_qk"] + 
                                                            head_stat["residual_norm_v"] + 
                                                            head_stat["residual_norm_o"])
        
        # FFN projection (once per layer, includes gate/up/down)
        if merge_f and "ffn" in layer_cons and "ffn" in layer_task_raw:
            print(f"    🔧 Processing FFN projection for layer {li}...")
            ffn_cons = layer_cons["ffn"]
            layer_task_ffn = layer_task_raw["ffn"]
            
            # Initialize FFN result storage
            projected_task_vectors["ffn"][li] = {}
            
            # FFN stats
            layer_stats["ffn"] = {
                "constraints_gate": 0, "constraints_up": 0, "constraints_down": 0,
                "residual_norm_gate": 0.0, "residual_norm_up": 0.0, "residual_norm_down": 0.0,
                "cg_iterations": 0
            }
            
            # Gate projection (adaptive solver choice)
            if "dGate" in layer_task_ffn and ffn_cons.get("X_gate", torch.empty(0)).numel() > 0:
                m_gate = ffn_cons["X_gate"].shape[0]
                print(f"      📐 Gate projection (m={m_gate})...")
                
                dGate_task = layer_task_ffn["dGate"]
                start_time = time.time()
                
                # Adaptive choice: explicit for small m, CG for large m
                if m_gate <= 4000:  # heuristic threshold, tune per memory
                    print(f"        🚀 Using Cholesky explicit solver...")
                    dGate_proj, info_gate = ffn_gate_dense_project(ffn_cons, dGate_task,
                                                                  lam=lambda_ridge,
                                                                  device=ffn_device,
                                                                  compute_dtype=compute_dtype)
                else:
                    print(f"        🔄 Using CG iterative solver...")
                    dGate_proj, info_gate = cg_ffn_gate(ffn_cons, dGate_task, lambda_ridge, 
                                                       cg_maxit, cg_tol,
                                                       device=ffn_device, 
                                                       compute_dtype=compute_dtype)
                
                gate_time = time.time() - start_time
                projected_task_vectors["ffn"][li]["dGate_proj"] = dGate_proj.cpu()
                layer_stats["ffn"]["constraints_gate"] = m_gate
                layer_stats["ffn"]["residual_norm_gate"] = info_gate["residual_norm"]
                layer_stats["ffn"]["cg_iterations"] += info_gate["iterations"]
                layer_stats["ffn"]["gate_solver"] = info_gate.get("solver", "cg")
                layer_stats["ffn"]["gate_time"] = gate_time
                print(f"        ✅ Gate solve done: {info_gate.get('solver', 'cg')}, "
                      f"residual={info_gate['residual_norm']:.2e}, time={gate_time:.2f}s")
            elif "dGate" in layer_task_ffn:
                # No constraints: apply directly
                print(f"      📐 Gate applied directly (no constraints)...")
                projected_task_vectors["ffn"][li]["dGate_proj"] = layer_task_ffn["dGate"].cpu()
                layer_stats["ffn"]["constraints_gate"] = 0
                layer_stats["ffn"]["residual_norm_gate"] = 0.0
                layer_stats["ffn"]["gate_solver"] = "direct"
                layer_stats["ffn"]["gate_time"] = 0.0
            
            # Up projection (adaptive solver choice)
            if "dUp" in layer_task_ffn and ffn_cons.get("X_up", torch.empty(0)).numel() > 0:
                m_up = ffn_cons["X_up"].shape[0]
                print(f"      📐 Up projection (m={m_up})...")
                
                dUp_task = layer_task_ffn["dUp"]
                start_time = time.time()
                
                # Adaptive choice: explicit for small m, CG for large m
                if m_up <= 4000:  # heuristic threshold, tune per memory
                    print(f"        🚀 Using Cholesky explicit solver...")
                    dUp_proj, info_up = ffn_up_dense_project(ffn_cons, dUp_task,
                                                            lam=lambda_ridge,
                                                            device=ffn_device,
                                                            compute_dtype=compute_dtype)
                else:
                    print(f"        🔄 Using CG iterative solver...")
                    dUp_proj, info_up = cg_ffn_up(ffn_cons, dUp_task, lambda_ridge, 
                                                 cg_maxit, cg_tol,
                                                 device=ffn_device, 
                                                 compute_dtype=compute_dtype)
                
                up_time = time.time() - start_time
                projected_task_vectors["ffn"][li]["dUp_proj"] = dUp_proj.cpu()
                layer_stats["ffn"]["constraints_up"] = m_up
                layer_stats["ffn"]["residual_norm_up"] = info_up["residual_norm"]
                layer_stats["ffn"]["cg_iterations"] += info_up["iterations"]
                layer_stats["ffn"]["up_solver"] = info_up.get("solver", "cg")
                layer_stats["ffn"]["up_time"] = up_time
                print(f"        ✅ Up solve done: {info_up.get('solver', 'cg')}, "
                      f"residual={info_up['residual_norm']:.2e}, time={up_time:.2f}s")
            elif "dUp" in layer_task_ffn:
                # No constraints: apply directly
                print(f"      📐 Up applied directly (no constraints)...")
                projected_task_vectors["ffn"][li]["dUp_proj"] = layer_task_ffn["dUp"].cpu()
                layer_stats["ffn"]["constraints_up"] = 0
                layer_stats["ffn"]["residual_norm_up"] = 0.0
                layer_stats["ffn"]["up_solver"] = "direct"
                layer_stats["ffn"]["up_time"] = 0.0
            
            # Down projection (adaptive solver choice)
            if "dDown_T" in layer_task_ffn and ffn_cons.get("H", torch.empty(0)).numel() > 0:
                m_down = ffn_cons["H"].shape[0]
                print(f"      📐 Down projection (m={m_down})...")
                
                dDown_T_task = layer_task_ffn["dDown_T"]
                start_time = time.time()
                
                # Adaptive choice: explicit for small m, CG for large m
                if m_down <= 4000:  # heuristic threshold, tune per memory
                    print(f"        🚀 Using Cholesky explicit solver...")
                    dDown_T_proj, info_down = ffn_down_dense_project(ffn_cons, dDown_T_task,
                                                                   lam=lambda_ridge,
                                                                   device=ffn_device,
                                                                   compute_dtype=compute_dtype)
                else:
                    print(f"        🔄 Using CG iterative solver...")
                    dDown_T_proj, info_down = cg_ffn_down(ffn_cons, dDown_T_task, lambda_ridge, 
                                                        cg_maxit, cg_tol,
                                                        device=ffn_device, 
                                                        compute_dtype=compute_dtype)
                
                down_time = time.time() - start_time
                projected_task_vectors["ffn"][li]["dDown_T_proj"] = dDown_T_proj.cpu()
                layer_stats["ffn"]["constraints_down"] = m_down
                layer_stats["ffn"]["residual_norm_down"] = info_down["residual_norm"]
                layer_stats["ffn"]["cg_iterations"] += info_down["iterations"]
                layer_stats["ffn"]["down_solver"] = info_down.get("solver", "cg")
                layer_stats["ffn"]["down_time"] = down_time
                print(f"        ✅ Down solve done: {info_down.get('solver', 'cg')}, "
                      f"residual={info_down['residual_norm']:.2e}, time={down_time:.2f}s")
            elif "dDown_T" in layer_task_ffn:
                # No constraints: apply directly
                print(f"      📐 Down applied directly (no constraints)...")
                projected_task_vectors["ffn"][li]["dDown_T_proj"] = layer_task_ffn["dDown_T"].cpu()
                layer_stats["ffn"]["constraints_down"] = 0
                layer_stats["ffn"]["residual_norm_down"] = 0.0
                layer_stats["ffn"]["down_solver"] = "direct"
                layer_stats["ffn"]["down_time"] = 0.0
            
            # Update totals
            total_residual = (layer_stats["ffn"]["residual_norm_gate"] + 
                            layer_stats["ffn"]["residual_norm_up"] + 
                            layer_stats["ffn"]["residual_norm_down"])
            projection_stats["total_constraint_residual"] += total_residual
            projection_stats["total_cg_iterations"] += layer_stats["ffn"]["cg_iterations"]
        
        projection_stats["layer_stats"][li] = layer_stats
        
        # Free current layer's constraints to save VRAM
        del layer_cons
        cleanup_memory()
        print(f"  🧹 Cleared constraints for layer {li}, freed VRAM")
    
    # Free the shared model after processing all layers
    del model_R_shared
    cleanup_memory()
    print("🧹 Released shared constraint model")

    print(f"\n✅ Null-space projection finished!")
    print(f"  📊 Totals:")
    print(f"     - Total CG iterations: {projection_stats['total_cg_iterations']}")
    print(f"     - Sum of constraint residuals: {projection_stats['total_constraint_residual']:.6f}")
    
    # Solver usage stats
    solver_stats = {"dense_cholesky": 0, "cg": 0, "direct": 0}
    total_solver_time = 0.0
    
    for layer_stat in projection_stats["layer_stats"].values():
        if "ffn" in layer_stat:
            ffn_stat = layer_stat["ffn"]
            for solver_key in ["gate_solver", "up_solver", "down_solver"]:
                if solver_key in ffn_stat:
                    solver_type = ffn_stat[solver_key]
                    solver_stats[solver_type] = solver_stats.get(solver_type, 0) + 1
            
            for time_key in ["gate_time", "up_time", "down_time"]:
                if time_key in ffn_stat:
                    total_solver_time += ffn_stat[time_key]
    
    print(f"  🚀 FFN solver performance:")
    print(f"     - Cholesky explicit: {solver_stats.get('dense_cholesky', 0)} time(s)")
    print(f"     - CG iterative: {solver_stats.get('cg', 0)} time(s)") 
    print(f"     - Direct application: {solver_stats.get('direct', 0)} time(s)")
    print(f"     - Total FFN solve time: {total_solver_time:.2f}s")
    
    return {
        "projected_task_vectors": projected_task_vectors,
        "projection_stats": projection_stats,
        "config": {
            "merge_types": merge_types,
            "selected_layers": selected_layers,
            "selected_heads": selected_heads,
            "d_model": d_model,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "kv_heads": kv_heads,
            "compute_dtype": str(compute_dtype),
            "lambda_ridge": lambda_ridge,
            "cg_maxit": cg_maxit,
            "cg_tol": cg_tol
        }
    }


def save_projected_task_vectors(projected_data: Dict[str, Any], output_path: str):
    """Save the projected task vectors to file"""
    print(f"💾 Saving projections to: {output_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use pickle (supports torch tensors)
    with open(output_path, 'wb') as f:
        pickle.dump(projected_data, f)
    
    # Also save a JSON config for quick inspection
    config_path = output_path.replace('.pkl', '_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        config_data = projected_data["config"].copy()
        config_data["stats"] = {
            "total_cg_iterations": projected_data["projection_stats"]["total_cg_iterations"],
            "total_constraint_residual": projected_data["projection_stats"]["total_constraint_residual"]
        }
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    # Print file size
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ Saved: {output_path} ({file_size:.1f} MB)")
    print(f"📋 Config info: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Null-space projection - compute and save projected task vectors")
    
    # Base paths
    parser.add_argument("--base", type=str, 
                       default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B")
    parser.add_argument("--instruct", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target", type=str,
                       default="/opt/data/private/hzhcode/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # Data & constraint params
    parser.add_argument("--texts_r", type=str, required=True, help="Path to JSON sample file")
    parser.add_argument("--max_samples_r", type=int, default=10, help="Max number of samples")
    parser.add_argument("--neigh_radius", type=int, default=5, help="Boundary neighborhood radius")
    
    # Layer & head config
    parser.add_argument("--layers_tail", type=int, default=2, help="Process the last N layers")
    parser.add_argument("--heads", type=str, default="all", help="Heads to process ('all' or comma-separated indices)")
    
    # Weights & solver params
    parser.add_argument("--lambda_ridge", type=float, default=1e-4, help="Ridge parameter (λ)")
    parser.add_argument("--cg_maxit", type=int, default=100, help="Max CG iterations")
    parser.add_argument("--cg_tol", type=float, default=1e-5, help="CG convergence tolerance")
    
    # Compute config
    parser.add_argument("--compute_precision", type=str, choices=["fp32", "fp64"], default="fp32",
                       help="Computation precision")
    
    # Merge types
    parser.add_argument("--merge_types", type=str, default="qk", 
                       help="Merge types: a combination of q/k/v/o/f")
    
    # QK params
    parser.add_argument("--q_rows_per_text", type=int, default=8, help="Q constraint rows per text")
    parser.add_argument("--k_rows_per_text", type=int, default=8, help="K constraint rows per text")
    parser.add_argument("--w_q", type=float, default=1.0, help="Weight for Q constraints")
    parser.add_argument("--w_k", type=float, default=1.0, help="Weight for K constraints")
    
    # VO params
    parser.add_argument("--v_rows_per_text", type=int, default=4, help="V constraint target positions per text")
    parser.add_argument("--o_rows_per_text", type=int, default=4, help="O constraint target positions per text")
    parser.add_argument("--w_v", type=float, default=1.0, help="Weight for V constraints")
    parser.add_argument("--w_o", type=float, default=1.0, help="Weight for O constraints")
    
    # FFN params
    parser.add_argument("--ffn_rows_per_text", type=int, default=4, help="FFN-Down constraint target positions per text")
    parser.add_argument("--readout_dirs", type=int, default=2, help="Number of output readout directions c per head/layer")
    parser.add_argument("--w_ffn", type=float, default=1.0, help="Weight for FFN-Down constraints")
    
    # Multi-device config
    parser.add_argument("--qk_device", type=str, default="auto",
                       help="Computation device for QK constraints")
    parser.add_argument("--vo_device", type=str, default="auto",
                       help="Computation device for VO constraints")
    parser.add_argument("--ffn_device", type=str, default="auto",
                       help="Computation device for FFN constraints")
    
    # Hook config
    parser.add_argument("--use_hooks", action="store_true", default=True,
                       help="Use hooks to capture exact layer internals (recommended)")
    parser.add_argument("--no_hooks", action="store_true",
                       help="Disable hooks and use the legacy extraction method")
    
    # Sequence length limit
    parser.add_argument("--max_seq_len", type=int, default=7168,
                       help="Max sequence length (BF16-optimized attention, default 7168; BF16 halves memory usage)")
    
    # Output config
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Output file path (*.pkl)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Precision
    compute_dtype = torch.float64 if args.compute_precision == "fp64" else torch.float32
    
    print("🚀 Null-space projection - Stage 1")
    print("=" * 70)
    print(f"Base: {args.base}")
    print(f"Instruct: {args.instruct}")
    print(f"Target: {args.target}")
    print(f"Output file: {args.output_file}")
    print(f"Precision: {args.compute_precision.upper()}")
    print(f"Merge types: {args.merge_types.upper()}")
    
    # Hook method selection
    use_hooks = args.use_hooks and not args.no_hooks
    print(f"Feature extraction method: {'Hook-based (recommended)' if use_hooks else 'Legacy method'}")

    start_time = time.time()

    # Load models (on CPU)
    print("\n📥 Loading models on CPU...")
    model_base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()
    
    model_instruct = AutoModelForCausalLM.from_pretrained(
        args.instruct, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()
    
    model_target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True, trust_remote_code=True)

    # Config
    num_layers = model_target.config.num_hidden_layers
    n_heads = model_target.config.num_attention_heads

    selected_layers = list(range(num_layers - args.layers_tail, num_layers))
    if args.heads == "all":
        selected_heads = list(range(n_heads))
    else:
        selected_heads = [int(x) for x in args.heads.split(",")]

    print(f"📋 Run config:")
    print(f"  Layers: {selected_layers}")
    print(f"  Heads: {len(selected_heads)}/{n_heads}")

    # Read data
    texts_R = read_json_samples_robust(args.texts_r, tokenizer, args.max_samples_r)
    print(f"📊 # JSON samples: {len(texts_R)}")

    # Compute null-space projections
    print("\n🔬 Computing null-space projections...")
    projected_data = compute_nullspace_projections(
        model_base, model_instruct, model_target,
        texts_R, tokenizer,
        selected_layers, selected_heads,
        args.neigh_radius, args.lambda_ridge, args.cg_maxit, args.cg_tol,
        compute_dtype, args.merge_types,
        # QK
        args.q_rows_per_text, args.k_rows_per_text, args.w_q, args.w_k,
        # VO
        args.v_rows_per_text, args.o_rows_per_text, args.w_v, args.w_o,
        # FFN
        args.ffn_rows_per_text, args.w_ffn, args.readout_dirs, args.seed,
        # Devices
        args.qk_device, args.vo_device, args.ffn_device,
        # Hooks
        use_hooks,
        # Max seq len
        args.max_seq_len
    )

    # Save results
    end_time = time.time()
    
    # Attach runtime info
    projected_data["runtime_info"] = {
        "runtime_seconds": end_time - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args)
    }
    
    save_projected_task_vectors(projected_data, args.output_file)

    print(f"\n✅ Null-space projection finished! Elapsed: {end_time - start_time:.1f}s")
    print(f"📁 Output file: {args.output_file}")
    print(f"🚀 Next: use scaling_model_merge.py to apply different scaling factors for model merging")


if __name__ == "__main__":
    main()