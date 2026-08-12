#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/LeFusion_v2/my_experiment
EXP=$REPO/experiments/20260813_exp020_gli_exp010_pseudomask_conditioned_test200
OUT=$EXP/outputs
PYTHON=/opt/conda/envs/lefusion/bin/python
INFER=$REPO/LeFusion/inference/inference.py
MANIFEST=$REPO/experiments/20260811_exp019_gli_exp010_pseudomask_test200_comparison/outputs/test200_manifest.json
OVERLAY=/workspace/LeFusion_v2/dataset/brats2024_gli_exp020_test200_pseudomasks
KEEPALIVE=/workspace/LeFusion_v2/.gpu_keepalive_adaptive.py
GPU0_UUID=GPU-b8da5535-9296-908f-419b-949bb0215754
GPU1_UUID=GPU-3813e464-a69e-ed3e-b477-2c319dbbb1ed
START=$(date +%s)
LIMIT=14400
mkdir -p "$OUT"
export PYTHONPATH="/workspace/LeFusion_v2/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME=/workspace/LeFusion_v2/model_cache/torch

log_phase() { printf '%s phase=%s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$OUT/formal_chain.log"; }
elapsed() { printf '%s' "$(( $(date +%s) - START ))"; }

start_keepalive() {
  if pgrep -f "^/opt/conda/envs/lefusion/bin/python $KEEPALIVE$" >/dev/null; then return; fi
  cd /workspace/LeFusion_v2
  CUDA_VISIBLE_DEVICES=0 KEEPALIVE_GPU_UUID=$GPU0_UUID nohup "$PYTHON" "$KEEPALIVE" \
    >.gpu_keepalive_gpu0.log 2>&1 </dev/null &
  CUDA_VISIBLE_DEVICES=1 KEEPALIVE_GPU_UUID=$GPU1_UUID nohup "$PYTHON" "$KEEPALIVE" \
    >.gpu_keepalive_gpu1.log 2>&1 </dev/null &
}

stop_keepalive() {
  mapfile -t pids < <(pgrep -f "^/opt/conda/envs/lefusion/bin/python $KEEPALIVE$")
  [[ ${#pids[@]} -eq 2 ]] || { echo "expected two keepalive processes" >&2; exit 1; }
  kill -TERM "${pids[@]}"
  for _ in $(seq 1 30); do
    if ! kill -0 "${pids[0]}" 2>/dev/null && ! kill -0 "${pids[1]}" 2>/dev/null; then return; fi
    sleep 1
  done
  return 1
}

run_pair() {
  local name=$1 config0=$2 config1=$3 status0=0 status1=0
  log_phase "${name}_start"
  "$PYTHON" "$INFER" --config-name "$config0" >"$OUT/${name}_gpu0.log" 2>&1 & local pid0=$!
  "$PYTHON" "$INFER" --config-name "$config1" >"$OUT/${name}_gpu1.log" 2>&1 & local pid1=$!
  wait "$pid0" || status0=$?
  wait "$pid1" || status1=$?
  if (( status0 != 0 || status1 != 0 )); then
    log_phase "${name}_failed_gpu0_${status0}_gpu1_${status1}"; exit 1
  fi
  log_phase "${name}_complete_elapsed_$(elapsed)s"
}

trap start_keepalive EXIT
cd "$REPO"
stop_keepalive
log_phase "formal_start_commit_$(git rev-parse HEAD)"

log_phase preflight_start
"$PYTHON" "$INFER" --config-name gli_exp020_direct_test200_shard0 \
  dataset.batch_size=1 output.root="$OUT/preflight/direct" output.max_batches=1 >"$OUT/preflight_direct.log" 2>&1 & p0=$!
"$PYTHON" "$INFER" --config-name gli_exp020_filtered_test200_shard1 \
  dataset.batch_size=1 output.root="$OUT/preflight/filtered" output.max_batches=1 >"$OUT/preflight_filtered.log" 2>&1 & p1=$!
wait "$p0"; wait "$p1"
log_phase "preflight_complete_elapsed_$(elapsed)s"

run_pair direct gli_exp020_direct_test200_shard0 gli_exp020_direct_test200_shard1
run_pair filtered gli_exp020_filtered_test200_shard0 gli_exp020_filtered_test200_shard1
if (( $(elapsed) >= LIMIT )); then log_phase "deadline_before_metrics_elapsed_$(elapsed)s"; exit 75; fi

log_phase metrics_start
"$PYTHON" scripts/gli_exp020_test200_metrics.py \
  --manifest "$MANIFEST" \
  --exp010-shard "$REPO/experiments/20260811_exp019_gli_exp010_pseudomask_test200_comparison/outputs/inference/exp010/shard_0" \
  --exp010-shard "$REPO/experiments/20260811_exp019_gli_exp010_pseudomask_test200_comparison/outputs/inference/exp010/shard_1" \
  --direct-shard "$OUT/inference/direct/shard_0" --direct-shard "$OUT/inference/direct/shard_1" \
  --filtered-shard "$OUT/inference/filtered/shard_0" --filtered-shard "$OUT/inference/filtered/shard_1" \
  --direct-overlay-contract "$OVERLAY/direct/contract.json" \
  --filtered-overlay-contract "$OVERLAY/filtered/contract.json" \
  --swav-weights /workspace/LeFusion_v2/model_cache/swav_800ep_pretrain.pth.tar \
  --device cuda:0 --output "$OUT/metrics" >"$OUT/metrics.log" 2>&1
log_phase "complete_elapsed_$(elapsed)s"

