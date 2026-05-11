import logging

logger = logging.getLogger(__name__)


def _char_to_token_map(offset_mapping):
    """Build char_pos -> token_idx mapping from offset_mapping."""
    char_to_tok = {}
    for tok_idx, (start, end) in enumerate(offset_mapping):
        for c in range(start, end):
            char_to_tok[c] = tok_idx
    return char_to_tok


def _find_last_tag_char(text, tag_str):
    """Return last character position of tag_str in text, or -1."""
    pos = -1
    search_from = 0
    while True:
        idx = text.find(tag_str, search_from)
        if idx == -1:
            break
        pos = idx
        search_from = idx + 1
    return pos


def parse_span_masks(input_ids, tokenizer, cfg, full_text=None, offset_mapping=None):
    """
    Returns:
    {
        "valid": bool,
        "skip_reason": str or None,
        "response_start_idx": int or None,
        "instruction_mask": list[int],
        "span_masks": {
            "user": list[int],
            "item": list[int],
            "match": list[int],
            "tag": list[int],
            "rate": list[int],
        }
    }
    """
    empty_spans = {"user": [], "item": [], "match": [], "tag": [], "rate": []}

    def _fail(reason):
        return {
            "valid": False,
            "skip_reason": reason,
            "response_start_idx": None,
            "instruction_mask": [],
            "user_history_mask": [],
            "span_masks": dict(empty_spans),
        }

    tags = cfg["tags"]
    # tags[0..7]: user_open, user_close, item_open, item_close,
    #             match_open, match_close, rate_open, rate_close

    # Build char->token map using offset_mapping (avoids context-dependent tokenization issues)
    if full_text is None or offset_mapping is None:
        return _fail("missing_full_text_or_offset_mapping")

    char_to_tok = _char_to_token_map(offset_mapping)

    def char_range_to_tok_range(char_start, char_end):
        """Convert [char_start, char_end) to sorted unique token indices."""
        toks = set()
        for c in range(char_start, char_end):
            if c in char_to_tok:
                toks.add(char_to_tok[c])
        return sorted(toks)

    # Step 1: find last occurrence of each tag in raw text, map to token index
    tag_char_starts = []
    for tag_str in tags:
        char_pos = _find_last_tag_char(full_text, tag_str)
        if char_pos == -1:
            return _fail(f"missing_tag_{tag_str}")
        tag_char_starts.append(char_pos)

    tag_char_ends = [tag_char_starts[i] + len(tags[i]) for i in range(8)]

    # Convert char positions to token positions
    starts = []
    ends_tok = []
    for i in range(8):
        tok_indices = char_range_to_tok_range(tag_char_starts[i], tag_char_ends[i])
        if not tok_indices:
            return _fail(f"tag_not_in_tokens_{tags[i]}")
        starts.append(tok_indices[0])
        ends_tok.append(tok_indices[-1] + 1)

    ends = ends_tok
    # starts: [user_open_s, user_close_s, item_open_s, item_close_s,
    #          match_open_s, match_close_s, rate_open_s, rate_close_s]

    # Step 3: order validation
    order = [starts[0], starts[1], starts[2], starts[3],
             starts[4], starts[5], starts[6], starts[7]]
    for i in range(len(order) - 1):
        if order[i] >= order[i + 1]:
            return _fail("tag_order_violation")

    # Step 4: content span interiors (end_of_open to start_of_close, exclusive)
    span_defs = [
        ("user",  1, 0, 1),   # interior: ends[0]..starts[1]
        ("item",  3, 2, 3),   # interior: ends[2]..starts[3]
        ("match", 5, 4, 5),   # interior: ends[4]..starts[5]
        ("rate",  7, 6, 7),   # interior: ends[6]..starts[7]
    ]
    span_masks = {}
    for span_name, _, open_idx, close_idx in span_defs:
        interior = list(range(ends[open_idx], starts[close_idx]))
        if len(interior) == 0:
            return _fail(f"empty_span_{span_name}")
        span_masks[span_name] = interior

    # Step 5 & 6: response_start_idx and instruction_mask
    response_start_idx = starts[0]
    instruction_mask = list(range(0, response_start_idx))

    # Step 6b: user_history_mask — tokens from the last "user_history:" marker to response_start_idx
    user_history_mask = []
    for search_str in ("user_history:", "user_history"):
        char_pos = _find_last_tag_char(full_text[:], search_str)
        if char_pos != -1:
            # find the token index at the end of this substring
            end_char = char_pos + len(search_str)
            # map char positions to tokens
            tok_indices = char_range_to_tok_range(char_pos, end_char)
            if tok_indices:
                hist_start_tok = tok_indices[-1] + 1  # first token after the marker
                user_history_mask = list(range(hist_start_tok, response_start_idx))
            break
    if not user_history_mask:
        logger.warning(
            "user_history_mask: could not find 'user_history:' marker; "
            "falling back to instruction_mask (response_start_idx=%d)", response_start_idx
        )
        user_history_mask = list(instruction_mask)

    # Step 7: tag token indices (union of all 8 tag occurrences)
    tag_indices = []
    for i in range(8):
        tag_indices.extend(range(starts[i], ends[i]))
    span_masks["tag"] = sorted(set(tag_indices))

    return {
        "valid": True,
        "skip_reason": None,
        "response_start_idx": response_start_idx,
        "instruction_mask": instruction_mask,
        "user_history_mask": user_history_mask,
        "span_masks": span_masks,
    }


def batch_parse_masks(examples, tokenizer, cfg):
    results = []
    skip_reasons = {}

    for ex in examples:
        enc = tokenizer(ex["full_text"], add_special_tokens=False, return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        offset_mapping = enc["offset_mapping"]
        parsed = parse_span_masks(input_ids, tokenizer, cfg,
                                  full_text=ex["full_text"],
                                  offset_mapping=offset_mapping)
        record = {
            "id": ex["id"],
            "full_text": ex["full_text"],
            "input_ids": input_ids,
            "gt": float(ex["gt"]),
            "valid": parsed["valid"],
            "skip_reason": parsed["skip_reason"],
            "response_start_idx": parsed["response_start_idx"],
            "instruction_mask": parsed["instruction_mask"],
            "user_history_mask": parsed["user_history_mask"],
            "span_masks": parsed["span_masks"],
        }
        results.append(record)
        if not parsed["valid"]:
            r = parsed["skip_reason"]
            skip_reasons[r] = skip_reasons.get(r, 0) + 1

    valid_count = sum(1 for r in results if r["valid"])
    print(f"\n[batch_parse_masks] Total: {len(results)}, Valid: {valid_count}")
    if skip_reasons:
        print("  Skip reasons:")
        for reason, cnt in sorted(skip_reasons.items()):
            print(f"    {reason}: {cnt}")

    return results


def extract_prompt_from_full_text(full_text, tokenizer, cfg):
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offset_mapping = enc["offset_mapping"]
    parsed = parse_span_masks(input_ids, tokenizer, cfg,
                              full_text=full_text, offset_mapping=offset_mapping)
    if not parsed["valid"]:
        logger.warning(
            "extract_prompt_from_full_text: parsing failed (%s), returning full text",
            parsed["skip_reason"],
        )
        return full_text
    prompt_ids = input_ids[: parsed["response_start_idx"]]
    return tokenizer.decode(prompt_ids, skip_special_tokens=False)
