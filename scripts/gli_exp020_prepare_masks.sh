#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/LeFusion_v2/my_experiment
PYTHON=/opt/conda/envs/lefusion/bin/python
MANIFEST=$REPO/experiments/20260811_exp019_gli_exp010_pseudomask_test200_comparison/outputs/test200_manifest.json
CONFIG=/workspace/LeFusion_v2/my_experiment_exp016/experiments/20260808_exp016_gli_p64_multimodal_classifier/config.yaml
CHECKPOINT=/workspace/LeFusion_v2/my_experiment_exp016/experiments/20260808_exp016_gli_p64_multimodal_classifier/outputs/supervised/best.pt
CHECKPOINT_SHA=3f460bbd245fd69ba6e2c6cf71f806bc28e4ebb3862a2e3fc5c411b7e66f3617
THRESHOLD=$REPO/experiments/20260811_exp018_gli_exp016_pseudomask_lefusion/outputs/masks/threshold/threshold.json
ROOT=/workspace/LeFusion_v2/dataset/brats2024_gli_exp020_test200_pseudomasks
MULTIMODAL_ROOT=/workspace/LeFusion_v2/dataset/brats2024_gli_exp020_test200_multimodal_inputs
EXP=$REPO/experiments/20260813_exp020_gli_exp010_pseudomask_conditioned_test200
mkdir -p "$EXP/outputs"

GPU0_UUID=GPU-b8da5535-9296-908f-419b-949bb0215754
GPU1_UUID=GPU-3813e464-a69e-ed3e-b477-2c319dbbb1ed
KEEPALIVE=/workspace/LeFusion_v2/.gpu_keepalive_adaptive.py

start_keepalive() {
  if pgrep -f "^/opt/conda/envs/lefusion/bin/python $KEEPALIVE$" >/dev/null; then
    return
  fi
  local caller_dir=$PWD
  cd /workspace/LeFusion_v2
  CUDA_VISIBLE_DEVICES=0 KEEPALIVE_GPU_UUID=$GPU0_UUID nohup "$PYTHON" "$KEEPALIVE" \
    >.gpu_keepalive_gpu0.log 2>&1 </dev/null &
  CUDA_VISIBLE_DEVICES=1 KEEPALIVE_GPU_UUID=$GPU1_UUID nohup "$PYTHON" "$KEEPALIVE" \
    >.gpu_keepalive_gpu1.log 2>&1 </dev/null &
  cd "$caller_dir"
}

stop_keepalive() {
  mapfile -t pids < <(pgrep -f "^/opt/conda/envs/lefusion/bin/python $KEEPALIVE$")
  if [[ ${#pids[@]} -ne 2 ]]; then
    echo "expected exactly two keepalive processes, found ${#pids[@]}" >&2
    exit 1
  fi
  for pid in "${pids[@]}"; do
    grep -q "KEEPALIVE_GPU_UUID=" "/proc/$pid/environ" || {
      echo "keepalive environment gate failed for PID $pid" >&2; exit 1;
    }
  done
  kill -TERM "${pids[@]}"
  for _ in $(seq 1 30); do
    if ! kill -0 "${pids[0]}" 2>/dev/null && ! kill -0 "${pids[1]}" 2>/dev/null; then return; fi
    sleep 1
  done
  echo "keepalive did not stop" >&2
  exit 1
}

trap start_keepalive EXIT
cd "$REPO"
sha256sum "$MANIFEST" "$CHECKPOINT" | tee "$EXP/outputs/mask_input_hashes.txt"
# CPU-only replay of the frozen crop origins. Only the selected 200 test paths
# are materialized; the keepalive stays active throughout this phase.
"$PYTHON" scripts/brats_gli_rebuild_multimodal_p64.py \
  --raw-root /workspace/LeFusion-main/BraTS-2024-Complete/BraTS-GLI \
  --raw-split train \
  --source-dataset-root /workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches \
  --split-file "$REPO/experiments/20260805_exp004_gli_inference_closed_loop/splits_v2.json" \
  --output-root "$MULTIMODAL_ROOT" --include-split test \
  --selection-manifest "$MANIFEST" --workers 8 --resume \
  >"$EXP/outputs/multimodal_test200_rebuild.log" 2>&1
stop_keepalive
"$PYTHON" scripts/gli_pseudomask_pipeline.py export-direct \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --split test --selection-manifest "$MANIFEST" \
  --dataset-root "$MULTIMODAL_ROOT" \
  --output-root "$ROOT/direct" --device cuda:0 --batch-size 8 --num-workers 8 --resume \
  >"$EXP/outputs/direct_mask_export.log" 2>&1

# Filtering is CPU-only, so restore protection before starting it.
start_keepalive
"$PYTHON" scripts/gli_pseudomask_pipeline.py derive-filtered \
  --source-root /workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches \
  --direct-root "$ROOT/direct" --output-root "$ROOT/filtered" \
  --split-file "$REPO/experiments/20260805_exp004_gli_inference_closed_loop/splits_v2.json" \
  --threshold-json "$THRESHOLD" --split test --selection-manifest "$MANIFEST" --resume \
  >"$EXP/outputs/filtered_mask_derive.log" 2>&1

sha256sum "$ROOT/direct/contract.json" "$ROOT/filtered/contract.json" \
  | tee "$EXP/outputs/mask_contract_hashes.txt"
