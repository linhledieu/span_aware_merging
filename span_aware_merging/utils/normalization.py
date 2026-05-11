import torch


def global_minmax_normalize(tensor_dict):
    """Normalize all tensors to [0,1] using global min/max across all values."""
    all_vals = torch.cat([v.reshape(-1).float() for v in tensor_dict.values()])
    g_min = all_vals.min().item()
    g_max = all_vals.max().item()
    if g_max == g_min:
        return {k: torch.zeros_like(v, dtype=torch.float32) for k, v in tensor_dict.items()}
    result = {}
    for k, v in tensor_dict.items():
        result[k] = (v.float() - g_min) / (g_max - g_min)
    return result


def normalize_responsibility_across_spans(R_user, R_item, R_match):
    """
    Inputs: three tensors of shape [num_layers, H].
    Returns three tensors of same shape summing to 1 at each (l, h) position.
    """
    denom = (R_user + R_item + R_match).clamp(min=1e-8)
    return R_user / denom, R_item / denom, R_match / denom
