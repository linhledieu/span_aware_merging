import logging
import math
import time

import torch

logger = logging.getLogger(__name__)


def compute_attention_statistics(model, records, head_config, cfg):
    """
    Computes U (rate→span) and a (alignment) averaged over valid records.

    Uses torch.inference_mode() since no gradients are needed here.

    Returns dict with keys:
      "U": [num_layers, num_attention_heads, 3]  (user, item, match)
      "a": [num_layers, num_attention_heads, 3]
      "u": [num_layers, num_attention_heads, 3]  (always zeros — leakage dropped)
    All float32 on CPU.
    """
    device = cfg["device"]
    num_layers = head_config["num_layers"]
    H_q = head_config["num_attention_heads"]
    chunk_size = int(cfg.get("attention_chunk_size", 8))
    log_every = max(1, int(cfg.get("attention_log_every", 10)))
    start_time = time.time()

    accum_device = torch.device(device)
    U_sum = torch.zeros(num_layers, H_q, 3, dtype=torch.float32, device=accum_device)
    U_cnt = torch.zeros(num_layers, H_q, dtype=torch.float32, device=accum_device)
    a_sum = torch.zeros(num_layers, H_q, 3, dtype=torch.float32, device=accum_device)
    a_cnt = torch.zeros(num_layers, H_q, dtype=torch.float32, device=accum_device)

    span_order = ["user", "item", "match"]

    for rec_idx, record in enumerate(records):
        if not record["valid"]:
            continue

        input_ids = record["input_ids"]
        seq_len = len(input_ids)
        num_chunks = math.ceil(seq_len / chunk_size)

        T_instr = record["instruction_mask"]
        T_rate = record["span_masks"]["rate"]
        T_user = record["span_masks"]["user"]
        T_item = record["span_masks"]["item"]
        T_match = record["span_masks"]["match"]
        T_tag = record["span_masks"]["tag"]
        has_instr = len(T_instr) > 0

        # Separate key sets for alignment: user span uses user_history_mask,
        # item and match spans use instruction_mask.
        user_history = record.get("user_history_mask", [])
        if not user_history:
            logger.warning(
                "record id=%s has empty user_history_mask; falling back to instruction_mask",
                record.get("id"),
            )
            user_history = T_instr
        T_instr_user = user_history
        T_instr_other = T_instr

        query_union = set(T_rate) | set(T_user) | set(T_item) | set(T_match)
        rate_set = set(T_rate)
        span_sets = [set(T_user), set(T_item), set(T_match)]
        span_token_counts = [len(T_user), len(T_item), len(T_match)]
        has_span_queries = any(count > 0 for count in span_token_counts)
        rate_att_sum = torch.zeros(num_layers, H_q, 3, dtype=torch.float32, device=accum_device)
        align_att_sum = torch.zeros(num_layers, H_q, 3, dtype=torch.float32, device=accum_device)
        past_key_values = None

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(seq_len, chunk_start + chunk_size)
            chunk_input = torch.tensor(
                [input_ids[chunk_start:chunk_end]], dtype=torch.long, device=device
            )

            with torch.inference_mode():
                outputs = model(
                    input_ids=chunk_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )

            past_key_values = outputs.past_key_values
            attentions = outputs.attentions

            query_positions = [
                pos for pos in range(chunk_start, chunk_end) if pos in query_union
            ]
            if not query_positions:
                continue

            local_rows = [pos - chunk_start for pos in query_positions]
            rate_positions = [i for i, pos in enumerate(query_positions) if pos in rate_set]
            span_query_positions = [
                [i for i, pos in enumerate(query_positions) if pos in span_set]
                for span_set in span_sets
            ]
            prefix_limit = chunk_end
            span_key_positions = [
                [pos for pos in T_s if pos < prefix_limit]
                for T_s in (T_user, T_item, T_match)
            ]
            align_key_positions = [
                [pos for pos in T_instr_user if pos < prefix_limit],
                [pos for pos in T_instr_other if pos < prefix_limit],
                [pos for pos in T_instr_other if pos < prefix_limit],
            ]

            local_rows_t = torch.tensor(local_rows, dtype=torch.long, device=device)
            rate_positions_t = (
                torch.tensor(rate_positions, dtype=torch.long, device=device)
                if rate_positions else None
            )
            span_query_tensors = [
                torch.tensor(pos_list, dtype=torch.long, device=device)
                if pos_list else None
                for pos_list in span_query_positions
            ]
            span_key_tensors = [
                torch.tensor(pos_list, dtype=torch.long, device=device)
                if pos_list else None
                for pos_list in span_key_positions
            ]
            align_key_tensors = [
                torch.tensor(pos_list, dtype=torch.long, device=device)
                if pos_list else None
                for pos_list in align_key_positions
            ]

            for l in range(num_layers):
                att = attentions[l][0].float()  # [H_q, q_len, kv_len]
                chunk_rows = att.index_select(1, local_rows_t)  # [H_q, n_query, kv_len]

                if rate_positions_t is not None:
                    rate_rows = chunk_rows.index_select(1, rate_positions_t)  # [H_q, n_rate, kv_len]
                    for s_idx, key_t in enumerate(span_key_tensors):
                        if key_t is not None:
                            rate_att_sum[l, :, s_idx] += (
                                rate_rows.index_select(2, key_t).sum(dim=(1, 2))
                            )

                if has_instr:
                    for s_idx, query_t in enumerate(span_query_tensors):
                        key_t = align_key_tensors[s_idx]
                        if query_t is not None and key_t is not None:
                            align_att_sum[l, :, s_idx] += (
                                chunk_rows.index_select(1, query_t)
                                .index_select(2, key_t)
                                .sum(dim=(1, 2))
                            )

            del outputs
            del attentions

        if len(T_rate) > 0:
            A = rate_att_sum / max(len(T_rate), 1)
            denom = A.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(A, 1.0 / 3)
            U_sum += torch.where(denom < 1e-8, uniform, A / denom.clamp_min(1e-8))
        else:
            U_sum += 1.0 / 3
        U_cnt += 1

        if has_instr and has_span_queries:
            span_counts_t = torch.tensor(
                span_token_counts, dtype=torch.float32, device=accum_device
            ).clamp_min(1.0).view(1, 1, 3)
            a_sum += align_att_sum / span_counts_t
            a_cnt += 1

        if (rec_idx + 1) % log_every == 0 or rec_idx == 0 or rec_idx + 1 == len(records):
            elapsed = time.time() - start_time
            done = rec_idx + 1
            avg_sec = elapsed / done
            remaining = max(len(records) - done, 0) * avg_sec
            print(
                f"[step3_attention] processed {done}/{len(records)} records "
                f"({done / len(records) * 100:.1f}%) | "
                f"last_seq_len={seq_len} last_chunks={num_chunks} | "
                f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m"
                ,
                flush=True,
            )

    def safe_div(s, c):
        # c: [L, H], s: [L, H, 3]
        out = torch.zeros_like(s, dtype=torch.float32)
        mask = c > 0  # [L, H]
        out[mask] = (s[mask] / c[mask].unsqueeze(-1)).float()
        return out.cpu()

    U = safe_div(U_sum, U_cnt)
    a = safe_div(a_sum, a_cnt)
    u = torch.zeros_like(U)  # leakage dropped — always zero

    return {"U": U, "a": a, "u": u}
