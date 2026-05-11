#!/usr/bin/env bash

set -euo pipefail

GPU_INDEX="${GPU_INDEX:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-12000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

while true; do
  FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_INDEX}" | tr -d '[:space:]')"
  if [[ "${FREE_MB}" =~ ^[0-9]+$ ]] && [ "${FREE_MB}" -gt "${MIN_FREE_MB}" ]; then
    echo "GPU ${GPU_INDEX} is free enough: ${FREE_MB} MiB"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}" /data/uqlinh/exp3rt/venv/bin/python /home/uqlinh/RAIN-Merging/stage2_delta_loosen.py \
      --baseline_merged_model_path /data/uqlinh/rain_merging/stage3_output_20260423_195226_100/merged_model \
      --stage1_projected_path /data/uqlinh/rain_merging/stage1_output_20260423_195226/projected_task_vectors.pkl \
      --stage2_alpha_json /data/uqlinh/rain_merging/stage2_output_20260423_195226/alpha_true_forward_align_leak.json \
      --probe_data_path /home/uqlinh/ReasonMerge/Long-to-Short-via-Model-Merging/RAIN-Merging/data/instruction_calibration_set_from_reason_correct_direct_fail.jsonl \
      --output_dir /data/uqlinh/rain_merging/stage2_delta_loosen_spanaware_probe16_"$(date +%Y%m%d_%H%M%S)" \
      --global_lambda 1.0 \
      --delta_probe_eps 0.05 \
      --probe_eps_min 0.01 \
      --delta_max 0.20 \
      --tau_qual 0.01 \
      --tau_fmt 0.01 \
      --eta_q 1.0 \
      --eta_f 1.0 \
      --kappa_q 2.0 \
      --kappa_f 2.0 \
      --analyze_window 8 \
      --match_window 5 \
      --batch_size 1 \
      --device cuda:0 \
      --dtype bfloat16 \
      --quality_subset_mode auto \
      --format_subset_mode auto \
      --shortening_subset_mode auto \
      --max_probe_examples 16 \
      --seed 1234 \
      --epsilon_q_total 0.001 \
      --epsilon_f_total 0.0 \
      --joint_budget_subset_mode same_as_probe \
      --lambda_item_close 1.0 \
      --lambda_item_len 1.0 \
      --lambda_user_close 1.5 \
      --lambda_user_len 1.5 \
      --lambda_match_close 1.0 \
      --lambda_match_len 1.0 \
      --tau_user_close 0.02 \
      --tau_user_len 0.02 \
      --tau_match_close 0.02 \
      --tau_match_len 0.02 \
      --min_analyze_item_tokens_for_short 5 \
      --min_match_tokens_for_short 5
    break
  fi
  echo "GPU ${GPU_INDEX} free memory ${FREE_MB} MiB <= ${MIN_FREE_MB} MiB, sleeping ${SLEEP_SECONDS}s..."
  sleep "${SLEEP_SECONDS}"
done
