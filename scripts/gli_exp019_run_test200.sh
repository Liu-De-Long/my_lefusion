#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/LeFusion_v2/my_experiment
EXP=$REPO/experiments/20260811_exp019_gli_exp010_pseudomask_test200_comparison
OUT=$EXP/outputs
INFER=$REPO/LeFusion/inference/inference.py
PYTHON=/opt/conda/envs/lefusion/bin/python
STATUS=$OUT/formal_chain.log
LIMIT_SECONDS=13500
mkdir -p "$OUT"

log_phase() {
  printf '%s phase=%s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$STATUS"
}

elapsed() {
  printf '%s' "$(( $(date +%s) - START_EPOCH ))"
}

allow_new_stage() {
  local seconds
  seconds=$(elapsed)
  if (( seconds >= LIMIT_SECONDS )); then
    log_phase "deadline_no_new_stage_elapsed_${seconds}s"
    exit 75
  fi
}

run_pair() {
  local name=$1
  local config0=$2
  local config1=$3
  allow_new_stage
  log_phase "${name}_start"
  "$PYTHON" "$INFER" --config-name "$config0" >"$OUT/${name}_gpu0.log" 2>&1 &
  local pid0=$!
  "$PYTHON" "$INFER" --config-name "$config1" >"$OUT/${name}_gpu1.log" 2>&1 &
  local pid1=$!
  local status0=0 status1=0
  wait "$pid0" || status0=$?
  wait "$pid1" || status1=$?
  if (( status0 != 0 || status1 != 0 )); then
    log_phase "${name}_failed_gpu0_${status0}_gpu1_${status1}"
    exit 1
  fi
  log_phase "${name}_complete_elapsed_$(elapsed)s"
}

START_EPOCH=$(date +%s)
log_phase "formal_start_commit_$(git -C "$REPO" rev-parse HEAD)"

log_phase preflight_start
"$PYTHON" "$INFER" --config-name gli_exp019_exp010_test200_shard0 \
  output.root="$OUT/preflight/exp010" output.max_batches=1 >"$OUT/preflight_exp010.log" 2>&1 &
preflight0=$!
"$PYTHON" "$INFER" --config-name gli_exp019_direct_test200_shard1 \
  output.root="$OUT/preflight/direct" output.max_batches=1 >"$OUT/preflight_direct.log" 2>&1 &
preflight1=$!
wait "$preflight0"
wait "$preflight1"
"$PYTHON" "$INFER" --config-name gli_exp019_filtered_test200_shard0 \
  output.root="$OUT/preflight/filtered" output.max_batches=1 >"$OUT/preflight_filtered.log" 2>&1
log_phase "preflight_complete_elapsed_$(elapsed)s"

run_pair exp010 gli_exp019_exp010_test200_shard0 gli_exp019_exp010_test200_shard1
run_pair direct gli_exp019_direct_test200_shard0 gli_exp019_direct_test200_shard1
run_pair filtered gli_exp019_filtered_test200_shard0 gli_exp019_filtered_test200_shard1

allow_new_stage
log_phase metrics_start
"$PYTHON" "$REPO/scripts/gli_exp019_test200_metrics.py" \
  --manifest "$OUT/test200_manifest.json" \
  --exp010-shard "$OUT/inference/exp010/shard_0" --exp010-shard "$OUT/inference/exp010/shard_1" \
  --direct-shard "$OUT/inference/direct/shard_0" --direct-shard "$OUT/inference/direct/shard_1" \
  --filtered-shard "$OUT/inference/filtered/shard_0" --filtered-shard "$OUT/inference/filtered/shard_1" \
  --swav-weights /workspace/LeFusion_v2/model_cache/swav_800ep_pretrain.pth.tar \
  --device cuda:0 --output "$OUT/metrics" >"$OUT/metrics.log" 2>&1
log_phase "complete_elapsed_$(elapsed)s"
