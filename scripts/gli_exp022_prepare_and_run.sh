#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/LeFusion_v2/my_experiment
PYTHON=/opt/conda/envs/lefusion/bin/python
EXP=$REPO/experiments/20260815_exp022_gli_fig2_v2_test10_comparison
DATASET=/workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches
SPLIT=$REPO/experiments/20260805_exp004_gli_inference_closed_loop/splits_v2.json
MANIFEST=$EXP/outputs/selection/test10_manifest.json
MULTIMODAL=/workspace/LeFusion_v2/dataset/brats2024_gli_exp022_test10_multimodal_inputs
MASK_ROOT=/workspace/LeFusion_v2/dataset/brats2024_gli_exp022_test10_pseudomasks
CLASSIFIER_CONFIG=/workspace/LeFusion_v2/my_experiment_exp016/experiments/20260808_exp016_gli_p64_multimodal_classifier/config.yaml
CLASSIFIER=/workspace/LeFusion_v2/my_experiment_exp016/experiments/20260808_exp016_gli_p64_multimodal_classifier/outputs/supervised/best.pt
CLASSIFIER_SHA=3f460bbd245fd69ba6e2c6cf71f806bc28e4ebb3862a2e3fc5c411b7e66f3617
THRESHOLD=$REPO/experiments/20260811_exp018_gli_exp016_pseudomask_lefusion/outputs/masks/threshold/threshold.json
STAGE=$EXP/outputs/legacy_stage_inputs
LEGACY_STAGE=/workspace/LeFusion-main/LeFusion-main/outputs/exp022_v2_test10_stage
MODE=${1:-all}

if [[ "$MODE" != "all" && "$MODE" != "prepare" && "$MODE" != "inference" ]]; then
  echo "usage: $0 [all|prepare|inference]" >&2
  exit 2
fi

mkdir -p "$EXP/outputs/selection" "$EXP/outputs/logs"
cd "$REPO"

if [[ "$MODE" != "inference" ]]; then
"$PYTHON" scripts/gli_exp022_select_test10.py \
  --dataset-root "$DATASET" --split-file "$SPLIT" --output "$MANIFEST" \
  >"$EXP/outputs/logs/selection.log" 2>&1

"$PYTHON" scripts/brats_gli_rebuild_multimodal_p64.py \
  --raw-root /workspace/LeFusion-main/BraTS-2024-Complete/BraTS-GLI \
  --raw-split train --source-dataset-root "$DATASET" --split-file "$SPLIT" \
  --output-root "$MULTIMODAL" --include-split test \
  --selection-manifest "$MANIFEST" --workers 8 --resume \
  >"$EXP/outputs/logs/multimodal_rebuild.log" 2>&1

# The exp016 config intentionally stores its labeled-subset path relative to
# the original exp016 repository.  Run the current read-only exporter from
# that working directory so the checkpoint's frozen subset SHA is verified.
pushd /workspace/LeFusion_v2/my_experiment_exp016 >/dev/null
"$PYTHON" "$REPO/scripts/gli_pseudomask_pipeline.py" export-direct \
  --config "$CLASSIFIER_CONFIG" --checkpoint "$CLASSIFIER" \
  --expected-checkpoint-sha256 "$CLASSIFIER_SHA" --split test \
  --selection-manifest "$MANIFEST" --dataset-root "$MULTIMODAL" \
  --output-root "$MASK_ROOT/direct" --device cuda:0 --batch-size 8 \
  --num-workers 8 --resume >"$EXP/outputs/logs/direct_mask_export.log" 2>&1
popd >/dev/null

"$PYTHON" scripts/gli_pseudomask_pipeline.py derive-filtered \
  --source-root "$DATASET" --direct-root "$MASK_ROOT/direct" \
  --output-root "$MASK_ROOT/filtered" --split-file "$SPLIT" \
  --threshold-json "$THRESHOLD" --split test --selection-manifest "$MANIFEST" \
  --resume >"$EXP/outputs/logs/filtered_mask_derive.log" 2>&1

"$PYTHON" scripts/gli_exp022_prepare_legacy_inputs.py \
  --selection "$MANIFEST" --dataset-root "$DATASET" \
  --filtered-root "$MASK_ROOT/filtered" --output-root "$STAGE" \
  --legacy-stage-root "$LEGACY_STAGE" \
  >"$EXP/outputs/logs/legacy_input_prepare.log" 2>&1
fi

if [[ "$MODE" != "prepare" ]]; then
"$PYTHON" -m LeFusion.inference.inference \
  --config-name gli_exp022_filtered_v2_test10 \
  >"$EXP/outputs/logs/filtered_v2_inference.log" 2>&1
fi

if [[ "$MODE" != "inference" ]]; then
sha256sum "$MANIFEST" "$MASK_ROOT/direct/contract.json" \
  "$MASK_ROOT/filtered/contract.json" "$CLASSIFIER" "$THRESHOLD" \
  >"$EXP/outputs/input_contract_hashes.txt"
fi
