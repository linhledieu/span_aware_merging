import json


def get_defaults():
    return {
        # Model paths
        "model_r_path": "",          # path to RecZero reasoning model dir
        "model_b_path": "",          # path to shared base model dir
        "stage1_delta_path": "",     # path to Stage 1 null-space projected task vector .pt

        # Calibration data paths
        "dr_path": "",               # reasoning calibration jsonl, 150 examples
        "di_path": "",               # instruction calibration jsonl, 365 examples
        "test_path": "",             # test jsonl

        # Tag strings in expected sequence order
        "tags": [
            "<analyze user>", "</analyze user>",
            "<analyze item>", "</analyze item>",
            "<match>", "</match>",
            "<rate>", "</rate>",
        ],

        # Compression-ordered span weights (inverted from JSD importance weights).
        # Higher weight → span dominates the coefficient objective → larger α → more TALLRec influence.
        # User/item spans are targets for compression, so they get high weights.
        # Match span should be preserved, so it gets low weight; format suppression
        # separately reduces alpha on format-critical heads.
        # Derivation: w_compress = (1 - w_jsd) / sum(1 - w_jsd)
        #   raw: user=0.767, item=0.772, match=0.461, total=2.0
        "w_user": 0.384,
        "w_item": 0.386,
        "w_match": 0.230,

        # Hyperparameters
        "delta_fmt": 0.8,            # max suppression factor for format-critical heads in [0, 1]
        "delta_coh": 0.3,            # deprecated: unused by format suppression path
        "user_head_scale": 1.0,      # scale applied to user-dominant head coefficients (ablate 0.6-1.0)
        "item_head_scale": 1.0,      # scale applied to item-dominant head coefficients
        "nu": 0.5,                   # attention modulation strength
        "rho": 10.0,                 # leakage penalty
        "fisher_beta": 1.0,          # exponential decay strength for Fisher-as-protection-amplifier
        "alpha_upper": 1.0,          # coefficient ceiling
        "lambda_global": 1.0,        # global merge scale
        "lambda_grid": [0.4, 0.6, 0.8, 1.0, 1.2, 1.5],  # grid for lambda search

        # Ablation flags
        "USE_JSD_WEIGHTS": True,
        "USE_ATTENTION_MODULATION": True,
        "USE_FORMAT_SUPPRESSION": True,
        "USE_FISHER_FLOORS": True,   # deprecated alias for USE_FORMAT_SUPPRESSION
        "USE_FISHER_RESPONSIBILITY": True,

        # Device and precision
        "device": "cuda:0",
        "dtype_str": "bfloat16",
        "attention_chunk_size": 8,
        "attention_log_every": 10,

        # Output directories
        "masks_dir": "outputs/masks",
        "fisher_dir": "outputs/fisher",
        "attention_dir": "outputs/attention",
        "responsibility_dir": "outputs/responsibility",
        "floors_dir": "outputs/floors",
        "coefficients_dir": "outputs/coefficients",
        "checkpoint_dir": "outputs/checkpoints",
        "eval_dir": "outputs/eval",
    }


def load_config(override_json_str=None):
    cfg = get_defaults()
    if override_json_str is not None:
        overrides = json.loads(override_json_str)
        cfg.update(overrides)
    # Apply flag resolutions immediately so all downstream code sees resolved values
    if not cfg["USE_JSD_WEIGHTS"]:
        cfg["w_user"] = 1 / 3
        cfg["w_item"] = 1 / 3
        cfg["w_match"] = 1 / 3
    if not cfg["USE_ATTENTION_MODULATION"]:
        cfg["nu"] = 0.0
    use_format_supp = cfg.get("USE_FORMAT_SUPPRESSION", cfg.get("USE_FISHER_FLOORS", True))
    cfg["USE_FORMAT_SUPPRESSION"] = bool(use_format_supp)
    cfg["USE_FISHER_FLOORS"] = cfg["USE_FORMAT_SUPPRESSION"]
    if not cfg["USE_FORMAT_SUPPRESSION"]:
        cfg["delta_fmt"] = 0.0
        cfg["delta_coh"] = 0.0
    return cfg
