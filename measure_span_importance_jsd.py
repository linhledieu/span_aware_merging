#!/usr/bin/env python3
"""
Measure span importance and compressibility for structured recommendation traces.

This is an analysis utility only. It does not train, probe Stage 2, or modify
model weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("measure_span_importance_jsd")
SPAN_ORDER = ("user", "item", "match")
TAG_STRINGS = {
    "user_open": "<analyze user>",
    "user_close": "</analyze user>",
    "item_open": "<analyze item>",
    "item_close": "</analyze item>",
    "match_open": "<match>",
    "match_close": "</match>",
    "rate_open": "<rate>",
    "rate_close": "</rate>",
}


@dataclass
class ParsedExample:
    sample_id: str
    full_text: str
    input_ids: List[int]
    user_range: Tuple[int, int]
    item_range: Tuple[int, int]
    match_range: Tuple[int, int]
    rate_range: Tuple[int, int]
    gt_rating: Optional[float]


@dataclass
class RolloutAuditResult:
    num_examples_used: int
    valid_rollouts: int
    invalid_rollouts: int
    mean_baseline_abs_error: Optional[float]
    mean_neutralized_abs_error: Optional[float]
    mean_delta_abs_error: Optional[float]


@dataclass
class SpanUnit:
    unit_index: int
    start: int
    end: int
    token_length: int
    text: str
    source: str


@dataclass
class ExampleSpanCompressibility:
    sample_id: str
    span: str
    measured: bool
    skip_reason: Optional[str]
    segmentation_mode: Optional[str]
    span_token_length: int
    num_units: int
    unit_importances: List[Dict[str, Any]]
    cumulative_curve: List[Dict[str, Any]]
    compressibility: Optional[float]


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def build_full_text(record: Dict[str, Any]) -> str:
    for key in ("full_text", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    prompt_response_pairs = (
        ("prompt", "response"),
        ("prompt", "output"),
        ("prompt_long_native", "raw_output_long_native"),
        ("prompt_short_native", "raw_output_short_native"),
        ("prompt_long_native", "response"),
        ("prompt_long_native", "output"),
    )
    for prompt_key, response_key in prompt_response_pairs:
        prompt = record.get(prompt_key)
        response = record.get(response_key)
        if isinstance(prompt, str) and prompt and isinstance(response, str) and response:
            return prompt + response
    raise ValueError("Could not build full_text from record")


def extract_gt_rating(record: Dict[str, Any]) -> Optional[float]:
    for key in ("gt", "ground_truth", "rating", "label"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    meta = record.get("meta")
    if isinstance(meta, dict):
        for key in ("gt", "ground_truth", "rating", "label"):
            value = meta.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def find_subsequence_positions(sequence_ids: Sequence[int], pattern_ids: Sequence[int]) -> List[int]:
    if not pattern_ids:
        return []
    positions: List[int] = []
    last = len(sequence_ids) - len(pattern_ids) + 1
    for idx in range(max(last, 0)):
        if list(sequence_ids[idx:idx + len(pattern_ids)]) == list(pattern_ids):
            positions.append(idx)
    return positions


def find_first_subsequence(sequence_ids: Sequence[int], pattern_ids: Sequence[int]) -> int:
    positions = find_subsequence_positions(sequence_ids, pattern_ids)
    if not positions:
        raise ValueError("subsequence_not_found")
    return positions[0]


def find_last_subsequence(sequence_ids: Sequence[int], pattern_ids: Sequence[int]) -> int:
    positions = find_subsequence_positions(sequence_ids, pattern_ids)
    if not positions:
        raise ValueError("subsequence_not_found")
    return positions[-1]


def find_last_tag_token_bounds_from_text(
    text: str,
    offset_mapping: Sequence[Tuple[int, int]],
    tag_text: str,
) -> Tuple[int, int]:
    char_start = text.rfind(tag_text)
    if char_start < 0:
        raise ValueError("subsequence_not_found")
    char_end = char_start + len(tag_text)
    token_positions = [
        idx
        for idx, (start, end) in enumerate(offset_mapping)
        if end > start and end > char_start and start < char_end
    ]
    if not token_positions:
        raise ValueError("subsequence_not_found")
    return token_positions[0], token_positions[-1] + 1


def get_tag_token_ids(tokenizer, tag_string: str) -> List[int]:
    token_ids = tokenizer(tag_string, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError(f"Tag tokenized to empty sequence: {tag_string}")
    return token_ids


def validate_neutral_token_id(
    tokenizer,
    token_id: int,
    disallowed_ids: Sequence[int],
) -> None:
    if token_id in disallowed_ids:
        raise ValueError(f"Neutral token id {token_id} is disallowed")
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        special_id = getattr(tokenizer, attr, None)
        if special_id is not None and token_id == special_id:
            raise ValueError(f"Neutral token id {token_id} matches tokenizer {attr}")


def parse_structured_example(
    record: Dict[str, Any],
    tokenizer,
    tag_token_ids: Dict[str, List[int]],
    sample_id: str,
) -> ParsedExample:
    full_text = build_full_text(record)
    tokenized = tokenize_with_offsets(tokenizer, full_text)
    input_ids = tokenized["input_ids"]
    offset_mapping = tokenized["offset_mapping"]

    def _locate(open_key: str, close_key: str) -> Tuple[Tuple[int, int], int, int]:
        # Prompts in this dataset often contain format examples with the same
        # XML-like tags. The assistant's real structured answer appears last,
        # so we anchor on the final textual tag occurrences. This also avoids
        # tokenizer-context mismatches for standalone tag tokenization.
        try:
            open_start, open_end = find_last_tag_token_bounds_from_text(
                full_text,
                offset_mapping,
                TAG_STRINGS[open_key],
            )
            close_start, _ = find_last_tag_token_bounds_from_text(
                full_text,
                offset_mapping,
                TAG_STRINGS[close_key],
            )
        except ValueError:
            open_start = find_last_subsequence(input_ids, tag_token_ids[open_key])
            close_start = find_last_subsequence(input_ids, tag_token_ids[close_key])
            open_end = open_start + len(tag_token_ids[open_key])
        if close_start < open_end:
            raise ValueError("bad_tag_order")
        interior = (open_end, close_start)
        return interior, open_start, close_start

    user_range, user_open_start, user_close_start = _locate("user_open", "user_close")
    item_range, item_open_start, item_close_start = _locate("item_open", "item_close")
    match_range, match_open_start, match_close_start = _locate("match_open", "match_close")
    rate_range, rate_open_start, rate_close_start = _locate("rate_open", "rate_close")

    order = [
        user_open_start, user_close_start,
        item_open_start, item_close_start,
        match_open_start, match_close_start,
        rate_open_start, rate_close_start,
    ]
    if order != sorted(order):
        raise ValueError("bad_tag_order")
    if user_range[0] >= user_range[1] or item_range[0] >= item_range[1] or match_range[0] >= match_range[1]:
        raise ValueError("empty_span")
    if rate_range[0] >= rate_range[1]:
        raise ValueError("empty_rate")

    return ParsedExample(
        sample_id=sample_id,
        full_text=full_text,
        input_ids=input_ids,
        user_range=user_range,
        item_range=item_range,
        match_range=match_range,
        rate_range=rate_range,
        gt_rating=extract_gt_rating(record),
    )


def tokenize_with_offsets(tokenizer, text: str) -> Dict[str, Any]:
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "offset_mapping": tokenized["offset_mapping"],
    }


def make_neutral_replacement_ids(
    tokenizer,
    target_len: int,
    neutral_token_id: Optional[int],
    neutral_token_text: str,
    neutral_mode: str,
    disallowed_ids: Sequence[int],
) -> List[int]:
    if target_len < 0:
        raise ValueError("target_len must be nonnegative")
    if target_len == 0:
        return []

    base_ids: List[int]
    if neutral_token_id is not None:
        validate_neutral_token_id(tokenizer, neutral_token_id, disallowed_ids)
        base_ids = [neutral_token_id]
    else:
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None:
            validate_neutral_token_id(tokenizer, int(unk_id), disallowed_ids)
            base_ids = [int(unk_id)]
        else:
            base_ids = tokenizer(neutral_token_text, add_special_tokens=False)["input_ids"]
            if not base_ids:
                raise ValueError("--neutral_token_text tokenizes to empty sequence")
            for token_id in base_ids:
                validate_neutral_token_id(tokenizer, int(token_id), disallowed_ids)

    if neutral_mode == "repeat_single_token":
        filler_id = base_ids[0]
        return [filler_id] * target_len
    if neutral_mode == "repeat_token_sequence":
        out: List[int] = []
        while len(out) < target_len:
            out.extend(base_ids)
        return out[:target_len]
    raise ValueError(f"Unsupported neutral_mode: {neutral_mode}")


def neutralize_span_ids(input_ids: Sequence[int], span_range: Tuple[int, int], filler_ids: Sequence[int]) -> List[int]:
    start, end = span_range
    if end - start != len(filler_ids):
        raise ValueError("filler length must match span length")
    return list(input_ids[:start]) + list(filler_ids) + list(input_ids[end:])


def decode_token_span(tokenizer, input_ids: Sequence[int], start: int, end: int) -> str:
    return tokenizer.decode(
        list(input_ids[start:end]),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def split_text_into_candidate_units(text: str) -> List[str]:
    parts: List[str] = []
    for chunk in re.split(r"\n+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                parts.append(sentence)
    return parts


def split_text_into_candidate_unit_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    n = len(text)
    start = 0
    idx = 0
    while idx < n:
        char = text[idx]
        end_here = False
        if char == "\n":
            end_here = True
        elif char in ".!?":
            next_char = text[idx + 1] if idx + 1 < n else ""
            if idx + 1 == n or next_char.isspace() or next_char == "\n":
                end_here = True
        if end_here:
            end = idx + 1
            segment_text = text[start:end]
            if segment_text.strip():
                spans.append((start, end, segment_text.strip()))
            start = end
        idx += 1
    if start < n:
        tail = text[start:n]
        if tail.strip():
            spans.append((start, n, tail.strip()))
    return spans


def align_char_spans_to_token_ranges(
    offset_mapping: Sequence[Tuple[int, int]],
    char_spans: Sequence[Tuple[int, int, str]],
) -> List[Tuple[int, int, str]]:
    aligned: List[Tuple[int, int, str]] = []
    token_count = len(offset_mapping)

    def _find_token_start(char_start: int) -> Optional[int]:
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_end <= tok_start:
                continue
            if tok_end > char_start and tok_start < char_start + 1:
                return tok_idx
            if tok_start <= char_start < tok_end:
                return tok_idx
        return None

    def _find_token_end(char_end: int) -> Optional[int]:
        for tok_idx in range(token_count - 1, -1, -1):
            tok_start, tok_end = offset_mapping[tok_idx]
            if tok_end <= tok_start:
                continue
            if tok_start < char_end and tok_end >= char_end:
                return tok_idx + 1
            if tok_start < char_end <= tok_end:
                return tok_idx + 1
        return None

    for char_start, char_end, segment_text in char_spans:
        tok_start = _find_token_start(char_start)
        tok_end = _find_token_end(char_end)
        if tok_start is None or tok_end is None or tok_end <= tok_start:
            return []
        aligned.append((tok_start, tok_end, segment_text))

    if not aligned:
        return []
    for idx in range(len(aligned) - 1):
        if aligned[idx][1] > aligned[idx + 1][0]:
            return []
    if aligned[0][0] != 0 or aligned[-1][1] != token_count:
        return []
    return aligned


def build_fixed_token_chunks(span_len: int, chunk_size: int) -> List[Tuple[int, int]]:
    chunks: List[Tuple[int, int]] = []
    for start in range(0, span_len, chunk_size):
        end = min(span_len, start + chunk_size)
        chunks.append((start, end))
    return chunks


def segment_span_into_units(
    input_ids: Sequence[int],
    span_range: Tuple[int, int],
    tokenizer,
    chunk_size: int,
    min_span_tokens: int,
) -> Tuple[List[SpanUnit], Optional[str], Optional[str], Optional[str]]:
    def _build_chunk_units() -> List[SpanUnit]:
        chunk_units: List[SpanUnit] = []
        chunks = build_fixed_token_chunks(span_len, chunk_size)
        if len(chunks) < 2:
            return []
        for idx, (local_start, local_end) in enumerate(chunks):
            abs_start = start + local_start
            abs_end = start + local_end
            chunk_units.append(SpanUnit(
                unit_index=idx,
                start=abs_start,
                end=abs_end,
                token_length=abs_end - abs_start,
                text=decode_token_span(tokenizer, input_ids, abs_start, abs_end),
                source="chunk",
            ))
        return chunk_units

    start, end = span_range
    span_len = end - start
    if span_len < min_span_tokens:
        return [], "too_short", None, None
    span_text = decode_token_span(tokenizer, input_ids, start, end)
    tokenized = tokenize_with_offsets(tokenizer, span_text)
    original_span_ids = list(input_ids[start:end])
    span_token_ids = tokenized["input_ids"]
    offset_mapping = tokenized["offset_mapping"]
    if span_token_ids != original_span_ids:
        units = _build_chunk_units()
        if len(units) < 2:
            return [], "too_short", None, None
        return units, None, "chunk", "retokenization_mismatch_fallback"
    char_spans = split_text_into_candidate_unit_spans(span_text)
    aligned = align_char_spans_to_token_ranges(offset_mapping, char_spans)
    units: List[SpanUnit] = []
    if aligned and len(aligned) >= 2:
        for idx, (local_start, local_end, unit_text) in enumerate(aligned):
            abs_start = start + local_start
            abs_end = start + local_end
            units.append(SpanUnit(idx, abs_start, abs_end, abs_end - abs_start, unit_text, "sentence"))
        segmentation_mode = "sentence"
        fallback_reason = None
    else:
        units = _build_chunk_units()
        if len(units) < 2:
            return [], "too_short", None, None
        segmentation_mode = "chunk"
        fallback_reason = None
    covered = [(unit.start, unit.end) for unit in units]
    if covered[0][0] != start or covered[-1][1] != end:
        return [], "alignment_failed", None, None
    for idx in range(len(covered) - 1):
        if covered[idx][1] != covered[idx + 1][0]:
            return [], "alignment_failed", None, None
    return units, None, segmentation_mode, fallback_reason


def neutralize_multiple_ranges_ids(
    input_ids: Sequence[int],
    ranges: Sequence[Tuple[int, int]],
    tokenizer,
    neutral_token_id: Optional[int],
    neutral_token_text: str,
    neutral_mode: str,
    disallowed_ids: Sequence[int],
) -> List[int]:
    output_ids = list(input_ids)
    for start, end in sorted(ranges, key=lambda r: r[0], reverse=True):
        filler_ids = make_neutral_replacement_ids(
            tokenizer,
            end - start,
            neutral_token_id,
            neutral_token_text,
            neutral_mode,
            disallowed_ids,
        )
        output_ids[start:end] = filler_ids
    return output_ids


def enforce_monotone_running_max(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    sorted_points = sorted(points, key=lambda item: item[0])
    monotone: List[Tuple[float, float]] = []
    running_max = -float("inf")
    for removable_fraction, cumulative_jsd in sorted_points:
        running_max = max(running_max, cumulative_jsd)
        monotone.append((removable_fraction, running_max))
    return monotone


def compute_per_example_compressibility(
    cumulative_points: Sequence[Tuple[float, float]],
    epsilon_kl: float,
) -> float:
    if not cumulative_points:
        return 0.0
    points = sorted(cumulative_points, key=lambda item: item[0])
    nonzero_points = [(r, j) for r, j in points if r > 0.0]
    if not nonzero_points:
        return 0.0
    if nonzero_points[0][1] > epsilon_kl:
        return 0.0
    if nonzero_points[-1][1] <= epsilon_kl:
        return 1.0
    prev_r, prev_j = 0.0, 0.0
    for r, j in nonzero_points:
        if prev_j <= epsilon_kl < j:
            if abs(j - prev_j) <= 1e-12:
                return float(prev_r)
            value = prev_r + (r - prev_r) * ((epsilon_kl - prev_j) / (j - prev_j))
            return float(max(0.0, min(1.0, value)))
        prev_r, prev_j = r, j
    return 0.0


def compute_rate_jsd(
    model,
    input_ids_a: Sequence[int],
    input_ids_b: Sequence[int],
    rate_range: Tuple[int, int],
    device: str,
) -> Dict[str, float]:
    rate_start, rate_end = rate_range
    assert rate_start > 0
    positions = list(range(rate_start - 1, rate_end - 1))
    tensor_a = torch.tensor([list(input_ids_a)], dtype=torch.long, device=device)
    tensor_b = torch.tensor([list(input_ids_b)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits_a = model(input_ids=tensor_a).logits[0, positions, :].float()
        logits_b = model(input_ids=tensor_b).logits[0, positions, :].float()
    p = torch.softmax(logits_a, dim=-1)
    q = torch.softmax(logits_b, dim=-1)
    m = 0.5 * (p + q)
    kl_ab = torch.sum(p * (torch.log(p.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12))), dim=-1)
    kl_ba = torch.sum(q * (torch.log(q.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12))), dim=-1)
    jsd = 0.5 * kl_ab + 0.5 * kl_ba
    return {
        "jsd_mean": float(jsd.mean().item()),
        "kl_ab_mean": float(kl_ab.mean().item()),
        "kl_ba_mean": float(kl_ba.mean().item()),
    }


def normalize_to_unit_interval(values_dict: Dict[str, float]) -> Dict[str, float]:
    values = [values_dict[span] for span in SPAN_ORDER if span in values_dict]
    if not values:
        return {}
    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) <= 1e-12:
        return {span: 0.5 for span in values_dict}
    return {span: (value - v_min) / (v_max - v_min) for span, value in values_dict.items()}


def bootstrap_mean_ci(values: Sequence[float], num_resamples: int, seed: int) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    samples: List[float] = []
    values_list = list(values)
    for _ in range(num_resamples):
        resample = [values_list[rng.randrange(len(values_list))] for _ in range(len(values_list))]
        samples.append(sum(resample) / len(resample))
    samples.sort()
    low_idx = max(0, int(0.025 * (len(samples) - 1)))
    high_idx = min(len(samples) - 1, int(0.975 * (len(samples) - 1)))
    return {
        "mean": sum(values_list) / len(values_list),
        "ci_low": samples[low_idx],
        "ci_high": samples[high_idx],
    }


def parse_rating_from_generated_text(text: str) -> Optional[float]:
    match = re.search(r"<rate>\s*([+-]?\d+(?:\.\d+)?)\s*</rate>", text, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    match = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def load_and_parse_examples(
    data_path: str,
    tokenizer,
    max_examples: int,
    verbose: bool,
) -> Tuple[List[ParsedExample], Dict[str, int], int, List[str]]:
    tag_token_ids = {key: get_tag_token_ids(tokenizer, value) for key, value in TAG_STRINGS.items()}
    valid_examples: List[ParsedExample] = []
    skip_reasons = Counter()
    warnings: List[str] = []
    num_records_read = 0

    with open(data_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            num_records_read += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skip_reasons["parse_error"] += 1
                continue
            try:
                sample_id = str(record.get("id", record.get("example_id", f"sample_{line_no}")))
                parsed = parse_structured_example(record, tokenizer, tag_token_ids, sample_id)
                valid_examples.append(parsed)
            except ValueError as exc:
                message = str(exc)
                if "subsequence_not_found" in message:
                    skip_reasons["missing_tag"] += 1
                elif message in {"bad_tag_order", "empty_span", "empty_rate"}:
                    skip_reasons[message] += 1
                else:
                    skip_reasons["parse_error"] += 1
                if verbose:
                    LOGGER.info("Skipping line %d: %s", line_no, message)
            if len(valid_examples) >= max_examples:
                break

    if len(valid_examples) < 10:
        warnings.append(f"Only {len(valid_examples)} valid examples remain after parsing")
    return valid_examples, dict(skip_reasons), num_records_read, warnings


def _span_range(example: ParsedExample, span: str) -> Tuple[int, int]:
    return getattr(example, f"{span}_range")


def measure_span_importance_jsd(
    model,
    tokenizer,
    examples: Sequence[ParsedExample],
    spans: Sequence[str],
    neutral_token_id: Optional[int],
    neutral_token_text: str,
    neutral_mode: str,
    device: str,
    verbose: bool,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Optional[List[Dict[str, Any]]]]:
    tag_ids_flat = [token_id for tag in TAG_STRINGS.values() for token_id in get_tag_token_ids(tokenizer, tag)]
    per_span_metrics: Dict[str, Dict[str, List[float]]] = {
        span: {"jsd": [], "kl_ab": [], "kl_ba": []} for span in spans
    }
    per_example_rows: List[Dict[str, Any]] = []
    num_bootstrap = 1000 if verbose else 200

    for example in examples:
        row = {
            "id": example.sample_id,
            "gt_rating": example.gt_rating,
            "rate_token_count": example.rate_range[1] - example.rate_range[0],
        }
        for span in spans:
            span_range = _span_range(example, span)
            filler_ids = make_neutral_replacement_ids(
                tokenizer,
                span_range[1] - span_range[0],
                neutral_token_id,
                neutral_token_text,
                neutral_mode,
                tag_ids_flat,
            )
            neutral_ids = neutralize_span_ids(example.input_ids, span_range, filler_ids)
            metrics = compute_rate_jsd(model, example.input_ids, neutral_ids, example.rate_range, device)
            per_span_metrics[span]["jsd"].append(metrics["jsd_mean"])
            per_span_metrics[span]["kl_ab"].append(metrics["kl_ab_mean"])
            per_span_metrics[span]["kl_ba"].append(metrics["kl_ba_mean"])
            row[f"importance_jsd_{span}"] = metrics["jsd_mean"]
            row[f"importance_kl_ab_{span}"] = metrics["kl_ab_mean"]
            row[f"importance_kl_ba_{span}"] = metrics["kl_ba_mean"]
        per_example_rows.append(row)

    aggregated: Dict[str, Dict[str, float]] = {}
    raw_means: Dict[str, Dict[str, float]] = {"jsd": {}, "kl_ab": {}, "kl_ba": {}}
    for span in spans:
        jsd_values = per_span_metrics[span]["jsd"]
        kl_ab_values = per_span_metrics[span]["kl_ab"]
        kl_ba_values = per_span_metrics[span]["kl_ba"]
        jsd_ci = bootstrap_mean_ci(jsd_values, num_bootstrap, seed)
        aggregated[span] = {
            "mean": jsd_ci["mean"],
            "std": statistics.pstdev(jsd_values) if len(jsd_values) > 1 else 0.0,
            "median": statistics.median(jsd_values) if jsd_values else 0.0,
            "ci_low": jsd_ci["ci_low"],
            "ci_high": jsd_ci["ci_high"],
            "kl_ab_mean": sum(kl_ab_values) / len(kl_ab_values) if kl_ab_values else 0.0,
            "kl_ba_mean": sum(kl_ba_values) / len(kl_ba_values) if kl_ba_values else 0.0,
        }
        raw_means["jsd"][span] = aggregated[span]["mean"]
        raw_means["kl_ab"][span] = aggregated[span]["kl_ab_mean"]
        raw_means["kl_ba"][span] = aggregated[span]["kl_ba_mean"]
    return aggregated, raw_means, per_example_rows


def measure_span_compressibility(
    model,
    tokenizer,
    examples: Sequence[ParsedExample],
    spans: Sequence[str],
    epsilon_kl: float,
    neutral_token_id: Optional[int],
    neutral_token_text: str,
    neutral_mode: str,
    chunk_size: int,
    min_span_tokens: int,
    device: str,
    per_example_rows: Optional[List[Dict[str, Any]]],
    verbose: bool,
    seed: int,
) -> Tuple[
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    List[str],
]:
    tag_ids_flat = [token_id for tag in TAG_STRINGS.values() for token_id in get_tag_token_ids(tokenizer, tag)]
    warnings: List[str] = []
    compressibility: Dict[str, float] = {}
    compressibility_stats: Dict[str, Dict[str, float]] = {}
    compressibility_counts: Dict[str, Dict[str, int]] = {}
    unit_importance_by_span_examplewise: Dict[str, Dict[str, Any]] = {}
    cumulative_removal_curves_examplewise: Dict[str, Dict[str, Any]] = {}
    num_bootstrap = 1000 if verbose else 200

    for span in spans:
        per_example_c: List[float] = []
        example_records: List[ExampleSpanCompressibility] = []
        unit_importances_flat: List[float] = []
        measured_count = 0
        skipped_too_short = 0
        skipped_other = 0
        measured_with_sentence = 0
        measured_with_chunk = 0
        retokenization_mismatch_fallback = 0
        aggregated_curve_buckets: Dict[float, List[float]] = defaultdict(list)
        avg_num_units_values: List[float] = []

        for ex_idx, example in enumerate(examples):
            span_range = _span_range(example, span)
            span_token_length = span_range[1] - span_range[0]
            units, skip_reason, segmentation_mode, fallback_reason = segment_span_into_units(
                example.input_ids,
                span_range,
                tokenizer,
                chunk_size,
                min_span_tokens,
            )
            if not units:
                if skip_reason == "too_short":
                    skipped_too_short += 1
                else:
                    skipped_other += 1
                record = ExampleSpanCompressibility(
                    sample_id=example.sample_id,
                    span=span,
                    measured=False,
                    skip_reason=skip_reason,
                    segmentation_mode=None,
                    span_token_length=span_token_length,
                    num_units=0,
                    unit_importances=[],
                    cumulative_curve=[],
                    compressibility=None,
                )
                example_records.append(record)
                if per_example_rows is not None:
                    row = per_example_rows[ex_idx]
                    row[f"compressibility_{span}"] = None
                    row[f"compressibility_measured_{span}"] = False
                    row[f"compressibility_skip_reason_{span}"] = skip_reason
                    row[f"num_units_{span}"] = 0
                    row[f"span_token_length_{span}"] = span_token_length
                    row[f"unit_importances_{span}"] = []
                    row[f"cumulative_curve_{span}"] = []
                continue

            unit_scores: List[Tuple[SpanUnit, float]] = []
            for unit in units:
                neutral_ids = neutralize_multiple_ranges_ids(
                    example.input_ids,
                    [(unit.start, unit.end)],
                    tokenizer,
                    neutral_token_id,
                    neutral_token_text,
                    neutral_mode,
                    tag_ids_flat,
                )
                metrics = compute_rate_jsd(model, example.input_ids, neutral_ids, example.rate_range, device)
                unit_scores.append((unit, metrics["jsd_mean"]))
                unit_importances_flat.append(metrics["jsd_mean"])

            sorted_unit_scores = sorted(unit_scores, key=lambda item: item[1])
            total_span_token_mass = float(sum(unit.token_length for unit in units))
            cumulative_points_raw: List[Tuple[float, float]] = [(0.0, 0.0)]
            cumulative_curve_meta: List[Tuple[int, List[int]]] = [(0, [])]
            removed_ranges: List[Tuple[int, int]] = []
            removed_unit_indices: List[int] = []
            removed_token_mass = 0
            unit_importance_records: List[Dict[str, Any]] = []

            for sorted_rank, (unit, jsd_value) in enumerate(sorted_unit_scores):
                unit_importance_records.append({
                    "unit_index_original": unit.unit_index,
                    "sorted_rank": sorted_rank,
                    "start": unit.start,
                    "end": unit.end,
                    "token_length": unit.token_length,
                    "text": unit.text,
                    "source": unit.source,
                    "leave_one_out_jsd": jsd_value,
                })
                removed_ranges.append((unit.start, unit.end))
                removed_unit_indices.append(unit.unit_index)
                removed_token_mass += unit.token_length
                cumulative_ids = neutralize_multiple_ranges_ids(
                    example.input_ids,
                    sorted(removed_ranges),
                    tokenizer,
                    neutral_token_id,
                    neutral_token_text,
                    neutral_mode,
                    tag_ids_flat,
                )
                cumulative_metrics = compute_rate_jsd(model, example.input_ids, cumulative_ids, example.rate_range, device)
                removable_fraction = removed_token_mass / total_span_token_mass
                cumulative_points_raw.append((removable_fraction, cumulative_metrics["jsd_mean"]))
                cumulative_curve_meta.append((sorted_rank + 1, list(removed_unit_indices)))

            monotone_points = enforce_monotone_running_max(cumulative_points_raw)
            full_curve_records: List[Dict[str, Any]] = []
            for (raw_r, raw_j), (mono_r, mono_j), (k_value, removed_indices) in zip(
                cumulative_points_raw,
                monotone_points,
                cumulative_curve_meta,
            ):
                if abs(raw_r - mono_r) > 1e-12:
                    raise ValueError("Monotone curve alignment mismatch")
                full_curve_records.append({
                    "k": k_value,
                    "removed_unit_indices_original": removed_indices,
                    "removable_fraction": raw_r,
                    "cumulative_jsd_raw": raw_j,
                    "cumulative_jsd_monotone": mono_j,
                })
            example_compressibility = compute_per_example_compressibility(monotone_points, epsilon_kl)
            per_example_c.append(example_compressibility)
            measured_count += 1
            if segmentation_mode == "sentence":
                measured_with_sentence += 1
            elif segmentation_mode == "chunk":
                measured_with_chunk += 1
            if fallback_reason == "retokenization_mismatch_fallback":
                retokenization_mismatch_fallback += 1
            avg_num_units_values.append(len(units))

            for removable_fraction, cumulative_jsd in monotone_points:
                aggregated_curve_buckets[round(removable_fraction, 2)].append(cumulative_jsd)

            record = ExampleSpanCompressibility(
                sample_id=example.sample_id,
                span=span,
                measured=True,
                skip_reason=None,
                segmentation_mode=segmentation_mode,
                span_token_length=span_token_length,
                num_units=len(units),
                unit_importances=unit_importance_records,
                cumulative_curve=full_curve_records,
                compressibility=example_compressibility,
            )
            example_records.append(record)
            if per_example_rows is not None:
                row = per_example_rows[ex_idx]
                row[f"compressibility_{span}"] = example_compressibility
                row[f"compressibility_measured_{span}"] = True
                row[f"compressibility_skip_reason_{span}"] = None
                row[f"num_units_{span}"] = len(units)
                row[f"span_token_length_{span}"] = span_token_length
                row[f"unit_importances_{span}"] = unit_importance_records
                row[f"cumulative_curve_{span}"] = full_curve_records

        if measured_count == 0:
            warnings.append(f"{span}: no examples measured for compressibility")
            compressibility[span] = 0.0
            compressibility_stats[span] = {"mean": 0.0, "std": 0.0, "median": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        else:
            ci = bootstrap_mean_ci(per_example_c, num_bootstrap, seed)
            compressibility[span] = sum(per_example_c) / len(per_example_c)
            compressibility_stats[span] = {
                "mean": compressibility[span],
                "std": statistics.pstdev(per_example_c) if len(per_example_c) > 1 else 0.0,
                "median": statistics.median(per_example_c),
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        compressibility_counts[span] = {
            "measured": measured_count,
            "skipped_too_short": skipped_too_short,
            "skipped_other": skipped_other,
            "measured_with_sentence": measured_with_sentence,
            "measured_with_chunk": measured_with_chunk,
            "retokenization_mismatch_fallback": retokenization_mismatch_fallback,
        }
        if skipped_too_short > max(1, len(examples) // 2):
            warnings.append(f"{span}: many examples too short for compressibility ({skipped_too_short}/{len(examples)})")
        if compressibility[span] <= 1e-6:
            warnings.append(f"{span}: compressibility is very close to 0")
        if measured_with_chunk > measured_with_sentence:
            warnings.append(
                f"{span}: chunk fallback used for {measured_with_chunk}/{measured_count} measured examples; "
                "sentence alignment may be weak for this tokenizer/output format"
            )
        if retokenization_mismatch_fallback > 0:
            warnings.append(
                f"{span}: retokenization mismatch forced chunk fallback for "
                f"{retokenization_mismatch_fallback}/{measured_count} measured examples"
            )

        compact_example_records = []
        compact_example_curves = []
        for record in example_records:
            if record.measured:
                compact_example_records.append({
                    "sample_id": record.sample_id,
                    "segmentation_mode": record.segmentation_mode,
                    "span_token_length": record.span_token_length,
                    "num_units": record.num_units,
                    "units": [
                        {
                            "original_index": unit["unit_index_original"],
                            "sorted_rank": unit["sorted_rank"],
                            "token_length": unit["token_length"],
                            "leave_one_out_jsd": unit["leave_one_out_jsd"],
                            "source": unit["source"],
                            "text": unit["text"],
                        }
                        for unit in record.unit_importances
                    ],
                })
                compact_example_curves.append({
                    "sample_id": record.sample_id,
                    "curve": record.cumulative_curve,
                })
        aggregated_curve = []
        for bucket_key in sorted(aggregated_curve_buckets.keys()):
            values = aggregated_curve_buckets[bucket_key]
            aggregated_curve.append({
                "bucket_index": bucket_key,
                "removable_fraction_mean": bucket_key,
                "cumulative_jsd_mean": sum(values) / len(values),
                "count": len(values),
            })
        unit_importance_by_span_examplewise[span] = {
            "num_examples_measured": measured_count,
            "num_examples_sentence_mode": measured_with_sentence,
            "num_examples_chunk_mode": measured_with_chunk,
            "avg_num_units": (sum(avg_num_units_values) / len(avg_num_units_values)) if avg_num_units_values else 0.0,
            "leave_one_out_jsd_mean": (sum(unit_importances_flat) / len(unit_importances_flat)) if unit_importances_flat else 0.0,
            "leave_one_out_jsd_median": statistics.median(unit_importances_flat) if unit_importances_flat else 0.0,
            "example_records": compact_example_records,
        }
        cumulative_removal_curves_examplewise[span] = {
            "num_examples_measured": measured_count,
            "aggregated_curve": aggregated_curve,
            "example_curves": compact_example_curves,
        }

    return (
        compressibility,
        compressibility_stats,
        compressibility_counts,
        unit_importance_by_span_examplewise,
        cumulative_removal_curves_examplewise,
        warnings,
    )


def compute_beta_from_metrics(
    span_importance_jsd_raw: Dict[str, float],
    span_compressibility_raw: Dict[str, float],
    lambda_importance: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    importance_norm = normalize_to_unit_interval(span_importance_jsd_raw)
    compressibility_norm = normalize_to_unit_interval(span_compressibility_raw)
    beta = {
        span: compressibility_norm[span] - lambda_importance * importance_norm[span]
        for span in span_importance_jsd_raw
    }
    return importance_norm, compressibility_norm, beta


def run_rollout_audit(
    model,
    tokenizer,
    examples: Sequence[ParsedExample],
    spans: Sequence[str],
    args,
    skip_reasons: Counter,
) -> Dict[str, Dict[str, Any]]:
    if not args.run_rollout_audit:
        return {}
    tag_ids_flat = [token_id for tag in TAG_STRINGS.values() for token_id in get_tag_token_ids(tokenizer, tag)]
    audit_examples = list(examples[:args.rollout_audit_examples])
    results: Dict[str, Dict[str, Any]] = {}
    for span in spans:
        baseline_errors: List[float] = []
        neutralized_errors: List[float] = []
        valid_rollouts = 0
        invalid_rollouts = 0
        used_examples = 0
        for example in audit_examples:
            if example.gt_rating is None:
                skip_reasons["missing_gt_for_rollout"] += 1
                continue
            span_range = _span_range(example, span)
            filler_ids = make_neutral_replacement_ids(
                tokenizer,
                span_range[1] - span_range[0],
                args.neutral_token_id,
                args.neutral_token_text,
                args.neutral_mode,
                tag_ids_flat,
            )
            neutral_ids = neutralize_span_ids(example.input_ids, span_range, filler_ids)
            prompt_ids = torch.tensor([example.input_ids], dtype=torch.long, device=args.device)
            neutral_prompt_ids = torch.tensor([neutral_ids], dtype=torch.long, device=args.device)
            example_valid = False
            for prompt_tensor, sink in ((prompt_ids, baseline_errors), (neutral_prompt_ids, neutralized_errors)):
                for _ in range(args.rollout_num_samples):
                    with torch.no_grad():
                        out = model.generate(
                            input_ids=prompt_tensor,
                            do_sample=True,
                            temperature=args.rollout_temperature,
                            top_p=args.rollout_top_p,
                            max_new_tokens=args.rollout_max_new_tokens,
                        )
                    generated = tokenizer.decode(
                        out[0][prompt_tensor.shape[1]:],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    rating = parse_rating_from_generated_text(generated)
                    if rating is None:
                        invalid_rollouts += 1
                        continue
                    sink.append(abs(rating - example.gt_rating))
                    valid_rollouts += 1
                    example_valid = True
            if example_valid:
                used_examples += 1
        results[span] = asdict(RolloutAuditResult(
            num_examples_used=used_examples,
            valid_rollouts=valid_rollouts,
            invalid_rollouts=invalid_rollouts,
            mean_baseline_abs_error=(sum(baseline_errors) / len(baseline_errors)) if baseline_errors else None,
            mean_neutralized_abs_error=(sum(neutralized_errors) / len(neutralized_errors)) if neutralized_errors else None,
            mean_delta_abs_error=(
                (sum(neutralized_errors) / len(neutralized_errors)) -
                (sum(baseline_errors) / len(baseline_errors))
            ) if baseline_errors and neutralized_errors else None,
        ))
    return results


def save_outputs(
    output_dir: str,
    config: Dict[str, Any],
    num_records_read: int,
    examples: Sequence[ParsedExample],
    skip_reasons_count: Dict[str, int],
    warnings: Sequence[str],
    importance_stats: Dict[str, Dict[str, float]],
    importance_raw_means: Dict[str, Dict[str, float]],
    importance_norm: Dict[str, float],
    compressibility_raw: Dict[str, float],
    compressibility_norm: Dict[str, float],
    compressibility_stats: Dict[str, Dict[str, float]],
    compressibility_counts: Dict[str, Dict[str, int]],
    beta: Dict[str, float],
    unit_importance_by_span_examplewise: Dict[str, Dict[str, Any]],
    cumulative_removal_curves_examplewise: Dict[str, Dict[str, Any]],
    rollout_audit_summary: Optional[Dict[str, Dict[str, Any]]],
    per_example_rows: Optional[List[Dict[str, Any]]],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    skip_compressibility = bool(config.get("skip_compressibility", False))
    summary = {
        "config": config,
        "num_records_read": num_records_read,
        "num_valid_examples": len(examples),
        "num_skipped_examples": num_records_read - len(examples),
        "skip_reasons_count": skip_reasons_count,
        "warnings": list(warnings),
        "span_importance_jsd_raw": {span: importance_stats[span]["mean"] for span in importance_stats},
        "span_importance_jsd_norm": importance_norm,
        "span_importance_kl_ab_raw": importance_raw_means["kl_ab"],
        "span_importance_kl_ba_raw": importance_raw_means["kl_ba"],
        "span_importance_stats": importance_stats,
        "span_compressibility_raw": compressibility_raw,
        "span_compressibility_norm": compressibility_norm,
        "span_compressibility_stats": compressibility_stats,
        "span_compressibility_counts": compressibility_counts,
        "beta": beta,
        "unit_importance_by_span_examplewise": unit_importance_by_span_examplewise,
        "cumulative_removal_curves_examplewise": cumulative_removal_curves_examplewise,
    }
    if rollout_audit_summary:
        summary["rollout_audit_summary"] = rollout_audit_summary
    save_json(os.path.join(output_dir, "summary.json"), summary)

    if per_example_rows is not None:
        with open(os.path.join(output_dir, "per_example_importance.jsonl"), "w", encoding="utf-8") as handle:
            for row in per_example_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = []
    lines.append("Span Ranking By Importance")
    for span, value in sorted(((span, importance_stats[span]["mean"]) for span in importance_stats), key=lambda x: x[1], reverse=True):
        lines.append(f"{span}: {value:.6f}")
    lines.append("")
    if not skip_compressibility:
        lines.append("Span Ranking By Compressibility")
        for span, value in sorted(compressibility_raw.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{span}: {value:.6f}")
        lines.append("")
        lines.append("Beta Values")
        for span, value in sorted(beta.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{span}: {value:.6f}")
        lines.append("")
        lines.append("Compressibility Stats")
        for span in compressibility_stats:
            stats = compressibility_stats[span]
            counts = compressibility_counts[span]
            avg_units = unit_importance_by_span_examplewise[span]["avg_num_units"]
            lines.append(
                f"{span}: mean={stats['mean']:.6f} std={stats['std']:.6f} "
                f"CI=[{stats['ci_low']:.6f}, {stats['ci_high']:.6f}] "
                f"measured={counts['measured']} skipped_too_short={counts['skipped_too_short']} "
                f"skipped_other={counts['skipped_other']} sentence_mode={counts['measured_with_sentence']} "
                f"chunk_mode={counts['measured_with_chunk']} "
                f"retok_mismatch_fallback={counts['retokenization_mismatch_fallback']} "
                f"avg_num_units={avg_units:.2f}"
            )
        lines.append("")
        lines.append("Most Often Removed First")
        for span, payload in unit_importance_by_span_examplewise.items():
            first_rank_counter: Counter[str] = Counter()
            first_rank_mode_counter: Counter[str] = Counter()
            for record in payload["example_records"]:
                units = record["units"]
                if not units:
                    continue
                first_unit = min(units, key=lambda item: item["sorted_rank"])
                first_rank_counter[first_unit["text"]] += 1
                first_rank_mode_counter[record.get("segmentation_mode") or first_unit.get("source") or "unknown"] += 1
            lines.append(span)
            if first_rank_mode_counter:
                mode_summary = ", ".join(
                    f"{mode}={count}" for mode, count in first_rank_mode_counter.most_common()
                )
                lines.append(f"  modes: {mode_summary}")
            for text, count in first_rank_counter.most_common(5):
                lines.append(f"  count={count} text={text[:120]}")
        lines.append("")
    if warnings:
        lines.append("Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
    with open(os.path.join(output_dir, "human_readable_report.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure span importance and compressibility with JSD")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--spans", type=str, default="user,item,match")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--min_span_tokens", type=int, default=16)
    parser.add_argument("--epsilon_kl", type=float, default=0.02)
    parser.add_argument("--lambda_importance", type=float, default=1.0)
    parser.add_argument("--skip_compressibility", action="store_true")
    parser.add_argument("--neutral_token_id", type=int, default=None)
    parser.add_argument("--neutral_token_text", type=str, default=" neutral")
    parser.add_argument("--neutral_mode", choices=["repeat_single_token", "repeat_token_sequence"], default="repeat_single_token")
    parser.add_argument("--run_rollout_audit", action="store_true")
    parser.add_argument("--rollout_audit_examples", type=int, default=20)
    parser.add_argument("--rollout_num_samples", type=int, default=4)
    parser.add_argument("--rollout_max_new_tokens", type=int, default=128)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_top_p", type=float, default=0.9)
    parser.add_argument("--save_per_example", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    spans = [span.strip() for span in args.spans.split(",") if span.strip()]
    for span in spans:
        if span not in SPAN_ORDER:
            raise ValueError(f"Unsupported span: {span}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.min_span_tokens <= 0:
        raise ValueError("--min_span_tokens must be positive")

    dtype = parse_dtype(args.dtype)
    LOGGER.info("Loading tokenizer from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    LOGGER.info("Loading model from %s", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    model.requires_grad_(False)
    model.to(args.device)

    examples, skip_reasons_count, num_records_read, warnings = load_and_parse_examples(
        args.data_path,
        tokenizer,
        args.max_examples,
        args.verbose,
    )
    skip_reasons_counter = Counter(skip_reasons_count)
    LOGGER.warning(
        "Examples read=%d valid=%d skipped=%d",
        num_records_read,
        len(examples),
        num_records_read - len(examples),
    )
    LOGGER.warning("Skip reasons: %s", dict(skip_reasons_counter))
    if not examples:
        raise ValueError("No valid examples after parsing")

    importance_stats, importance_raw_means, per_example_rows = measure_span_importance_jsd(
        model,
        tokenizer,
        examples,
        spans,
        args.neutral_token_id,
        args.neutral_token_text,
        args.neutral_mode,
        args.device,
        args.verbose,
        args.seed,
    )
    if args.skip_compressibility:
        compressibility_raw = {}
        compressibility_stats = {}
        compressibility_counts = {}
        unit_importance_by_span_examplewise = {}
        cumulative_removal_curves_examplewise = {}
        compressibility_norm = {}
        beta = {}
    else:
        (
            compressibility_raw,
            compressibility_stats,
            compressibility_counts,
            unit_importance_by_span_examplewise,
            cumulative_removal_curves_examplewise,
            compression_warnings,
        ) = (
            measure_span_compressibility(
                model,
                tokenizer,
                examples,
                spans,
                args.epsilon_kl,
                args.neutral_token_id,
                args.neutral_token_text,
                args.neutral_mode,
                args.chunk_size,
                args.min_span_tokens,
                args.device,
                per_example_rows if args.save_per_example else None,
                args.verbose,
                args.seed,
            )
        )
        warnings.extend(compression_warnings)
        _, compressibility_norm, beta = compute_beta_from_metrics(
            {span: importance_stats[span]["mean"] for span in spans},
            compressibility_raw,
            args.lambda_importance,
        )
    importance_norm = normalize_to_unit_interval(
        {span: importance_stats[span]["mean"] for span in spans}
    )
    rollout_audit_summary = run_rollout_audit(
        model,
        tokenizer,
        examples,
        spans,
        args,
        skip_reasons_counter,
    ) if args.run_rollout_audit else None

    for span in spans:
        stats = importance_stats[span]
        LOGGER.warning(
            "%s importance JSD: %.6f +/- %.6f CI[%.6f, %.6f]",
            span, stats["mean"], stats["std"], stats["ci_low"], stats["ci_high"],
        )
    if not args.skip_compressibility:
        for span in spans:
            stats = compressibility_stats[span]
            counts = compressibility_counts[span]
            LOGGER.warning(
                "%s compressibility: mean=%.6f std=%.6f CI[%.6f, %.6f] measured=%d skipped_too_short=%d skipped_other=%d sentence_mode=%d chunk_mode=%d retok_mismatch_fallback=%d beta=%.6f",
                span,
                stats["mean"],
                stats["std"],
                stats["ci_low"],
                stats["ci_high"],
                counts["measured"],
                counts["skipped_too_short"],
                counts["skipped_other"],
                counts["measured_with_sentence"],
                counts["measured_with_chunk"],
                counts["retokenization_mismatch_fallback"],
                beta[span],
            )
    if rollout_audit_summary:
        LOGGER.warning("Rollout audit summary: %s", rollout_audit_summary)

    save_outputs(
        output_dir=args.output_dir,
        config=vars(args),
        num_records_read=num_records_read,
        examples=examples,
        skip_reasons_count=dict(skip_reasons_counter),
        warnings=warnings,
        importance_stats=importance_stats,
        importance_raw_means=importance_raw_means,
        importance_norm=importance_norm,
        compressibility_raw=compressibility_raw,
        compressibility_norm=compressibility_norm,
        compressibility_stats=compressibility_stats,
        compressibility_counts=compressibility_counts,
        beta=beta,
        unit_importance_by_span_examplewise=unit_importance_by_span_examplewise,
        cumulative_removal_curves_examplewise=cumulative_removal_curves_examplewise,
        rollout_audit_summary=rollout_audit_summary,
        per_example_rows=per_example_rows if args.save_per_example else None,
    )


if __name__ == "__main__":
    main()
