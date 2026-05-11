import time
import torch
import torch.nn.functional as F


def _gpu_mem_summary(device):
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    try:
        dev_idx = torch.device(device).index
        if dev_idx is None:
            dev_idx = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)
        return (
            f"alloc={allocated:.2f}GiB reserved={reserved:.2f}GiB "
            f"peak={peak:.2f}GiB"
        )
    except Exception as e:
        return f"mem_query_failed:{e}"


def compute_span_fisher_for_proj(
    model, records, span_name, proj_type, param_names_by_layer, head_config, cfg
):
    """
    Returns float32 CPU tensor of shape [num_layers, H].
    H = num_attention_heads for q/o, num_key_value_heads for k/v.

    Correction: o_proj head partition lives in columns, not rows.
      - q/k/v: shape [H*head_dim, in_features] → reshape [H, head_dim, in_features]
               → sum squared over (head_dim, in_features) → [H]
      - o_proj: shape [out_features, H_q*head_dim] → reshape [out_features, H_q, head_dim]
               → sum squared over (out_features, head_dim) → [H_q]
    """
    device = cfg["device"]
    num_layers = head_config["num_layers"]
    head_dim = head_config["head_dim"]
    kv_head_dim = head_config["kv_head_dim"]

    if proj_type in ("q", "o"):
        H = head_config["num_attention_heads"]
        h_dim = head_dim
    else:
        H = head_config["num_key_value_heads"]
        h_dim = kv_head_dim

    accumulator = torch.zeros(num_layers, H, dtype=torch.float32)
    count = 0
    inspected = 0

    param_names = param_names_by_layer[proj_type + "_proj"]
    log_every = cfg.get("fisher_log_every", 10)

    print(
        f"    Starting {span_name}/{proj_type}: "
        f"{sum(1 for r in records if r['valid'] and r['span_masks'][span_name])} usable records, "
        f"{_gpu_mem_summary(device)}",
        flush=True,
    )

    for record in records:
        if not record["valid"]:
            continue
        span_indices = record["span_masks"][span_name]
        if len(span_indices) == 0:
            continue
        inspected += 1

        input_tensor = torch.tensor(
            [record["input_ids"]], dtype=torch.long, device=device
        )

        # No torch.no_grad() — gradients are required for Fisher computation
        outputs = model(input_ids=input_tensor, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        contributions = []
        for t in span_indices:
            if t == 0:
                continue
            target = record["input_ids"][t]
            lp = F.log_softmax(logits[0, t - 1, :], dim=-1)[target]
            contributions.append(-lp)

        if not contributions:
            continue

        loss = torch.stack(contributions).mean()
        loss.backward()

        for l, pname in enumerate(param_names):
            param = model.get_parameter(pname)
            if param.grad is None:
                continue
            grad = param.grad.float()

            if proj_type in ("q", "k", "v"):
                # Shape: [H * h_dim, in_features] → [H, h_dim, in_features]
                grad_r = grad.view(H, h_dim, -1)
                head_fisher = (grad_r ** 2).sum(dim=(1, 2))  # [H]
            else:
                # o_proj — columns carry head info
                # Shape: [out_features, H_q * h_dim] → [out_features, H_q, h_dim]
                out_features = grad.shape[0]
                grad_r = grad.view(out_features, H, h_dim)
                head_fisher = (grad_r ** 2).sum(dim=(0, 2))  # [H_q]

            accumulator[l] += head_fisher.cpu()

        model.zero_grad()
        count += 1
        if log_every and (count == 1 or count % log_every == 0):
            print(
                f"    Progress {span_name}/{proj_type}: "
                f"processed={count} inspected={inspected} "
                f"seq_len={len(record['input_ids'])} span_tokens={len(span_indices)} "
                f"loss={loss.item():.4f} {_gpu_mem_summary(device)}",
                flush=True,
            )

    if count > 0:
        accumulator /= count
    print(
        f"    Finished {span_name}/{proj_type}: processed={count}, "
        f"{_gpu_mem_summary(device)}",
        flush=True,
    )

    return accumulator


def compute_all_fisher(model, records, proj_param_names, head_config, cfg):
    """
    Single-pass Fisher: iterate records once, do one forward+backward per active span
    per record, and collect all proj types (q/k/v/o) from that one backward.
    This is ~4x faster than the original which did 16 separate full passes.
    """
    core_spans = ["tag", "match"]
    resp_spans = ["user", "item"] if cfg["USE_FISHER_RESPONSIBILITY"] else []
    all_spans = core_spans + resp_spans

    device = cfg["device"]
    num_layers = head_config["num_layers"]
    head_dim = head_config["head_dim"]
    kv_head_dim = head_config["kv_head_dim"]
    H_q = head_config["num_attention_heads"]
    H_kv = head_config["num_key_value_heads"]

    proj_types = ("q", "k", "v", "o")
    proj_H = {"q": H_q, "k": H_kv, "v": H_kv, "o": H_q}
    proj_h_dim = {"q": head_dim, "k": kv_head_dim, "v": kv_head_dim, "o": head_dim}

    # accumulators[span][proj_type] = float32 [num_layers, H]
    accumulators = {
        span: {pt: torch.zeros(num_layers, proj_H[pt], dtype=torch.float32) for pt in proj_types}
        for span in all_spans
    }
    counts = {span: 0 for span in all_spans}

    log_every = cfg.get("fisher_log_every", 10)
    valid_records = [r for r in records if r["valid"]]
    total = len(valid_records)

    t0 = time.time()
    for rec_i, record in enumerate(valid_records):
        input_tensor = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)

        # One forward per record; reuse logits across spans with retain_graph
        # Determine which spans are active for this record
        active_spans = [s for s in all_spans if len(record["span_masks"][s]) > 0]
        if not active_spans:
            continue

        # Forward once; we'll call backward multiple times with retain_graph
        outputs = model(input_ids=input_tensor, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab_size]

        for span_i, span_name in enumerate(active_spans):
            span_indices = record["span_masks"][span_name]
            contributions = []
            for t in span_indices:
                if t == 0:
                    continue
                target = record["input_ids"][t]
                lp = torch.nn.functional.log_softmax(logits[0, t - 1, :], dim=-1)[target]
                contributions.append(-lp)
            if not contributions:
                continue

            loss = torch.stack(contributions).mean()
            retain = span_i < len(active_spans) - 1  # keep graph for all but last span
            loss.backward(retain_graph=retain)

            # Collect squared gradients for all proj types at once
            for pt in proj_types:
                param_names = proj_param_names[pt + "_proj"]
                H = proj_H[pt]
                h_dim = proj_h_dim[pt]
                for l, pname in enumerate(param_names):
                    param = model.get_parameter(pname)
                    if param.grad is None:
                        continue
                    grad = param.grad.float()
                    if pt in ("q", "k", "v"):
                        grad_r = grad.view(H, h_dim, -1)
                        accumulators[span_name][pt][l] += (grad_r ** 2).sum(dim=(1, 2)).cpu()
                    else:
                        out_features = grad.shape[0]
                        grad_r = grad.view(out_features, H, h_dim)
                        accumulators[span_name][pt][l] += (grad_r ** 2).sum(dim=(0, 2)).cpu()

            model.zero_grad()
            counts[span_name] += 1

        if log_every and (rec_i == 0 or (rec_i + 1) % log_every == 0):
            elapsed = time.time() - t0
            print(
                f"  Fisher progress: record {rec_i+1}/{total} "
                f"seq_len={len(record['input_ids'])} "
                f"active_spans={active_spans} "
                f"elapsed={elapsed:.1f}s {_gpu_mem_summary(device)}",
                flush=True,
            )

    # Normalize by count per span
    fisher = {}
    for span_name in all_spans:
        fisher[span_name] = {}
        n = counts[span_name]
        for pt in proj_types:
            fisher[span_name][pt] = accumulators[span_name][pt] / max(1, n)

    print(f"\nTotal Fisher time: {time.time() - t0:.1f}s", flush=True)
    return fisher
