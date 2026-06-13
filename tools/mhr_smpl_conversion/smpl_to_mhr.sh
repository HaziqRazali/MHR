#!/usr/bin/env bash

: << BLOCK
Convert SMPL-X JSONs → MHR .npz files for a whole dataset.

# Usage (individual commands)
TEST_MODE=0 FORCE=0 ./smpl_to_mhr.sh --DATA_ROOT /data/haziq/mocap/data/fit3d
TEST_MODE=0 FORCE=0 ./smpl_to_mhr.sh --DATA_ROOT /data/haziq/mocap/data/fit3d
TEST_MODE=0 FORCE=0 ./smpl_to_mhr.sh --DATA_ROOT /data/haziq/mocap/data/fit3d --shard 0 --num_shards 4
TEST_MODE=0 FORCE=0 ./smpl_to_mhr.sh --DATA_ROOT /data/haziq/mocap/data/fit3d --shard 1 --num_shards 4

# tmux — video only (fast, keeps existing .npz)
for s in 0 1 2 3 4 5 6 7; do
  tmux new-window -n "shard$s" "cd ~/MHR/tools/mhr_smpl_conversion && TEST_MODE=0 FORCE_VIDEO=1 ./smpl_to_mhr.sh --DATA_ROOT /data/haziq/mocap/data/fit3d --shard $s --num_shards 8; read"
done

# tmux — full re-run from scratch (recomputes .npz with is_tracking=True)
tmux new-session -s mhr_convert
for s in 0 1 2 3; do
  tmux new-window -n "shard$s" "cd ~/MHR/tools/mhr_smpl_conversion && TEST_MODE=0 FORCE=0 ./smpl_to_mhr.sh --DATA_ROOT /data/haziq/mocap/data/fit3d --shard $s --num_shards 4; read"
done
ctrl+B then d to detach
ctrl+B then 0/1/2/3 to attach to a window
tmux attach -t mhr_convert
tmux ls

Expected input layout:
  $DATA_ROOT/{train,val}/<subject>/smplx/<action>.json

Output layout (mirrors run.sh convention):
  $DATA_ROOT/{train,val}/<subject>/mhr/<action>.npz

Environment variables:
  TEST_MODE=1  (default) → dry run, print commands only
  TEST_MODE=0            → actually run
  FORCE=1                → overwrite existing .npz + video (full re-run)
  FORCE=0    (default)   → skip if both .npz and .mp4 exist
  FORCE_VIDEO=1          → if .npz exists, re-render video only; else full pipeline
  VIDEO=1 (default)   → generate video; re-runs if .npz exists but .mp4 is missing
  VIDEO=0             → skip video rendering
BLOCK

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMPL_TO_MHR_PY="$SCRIPT_DIR/smpl_to_mhr.py"

TEST_MODE="${TEST_MODE:-1}"
FORCE="${FORCE:-0}"
FORCE_VIDEO="${FORCE_VIDEO:-0}"
VIDEO="${VIDEO:-1}"

DATA_ROOT="${DATA_ROOT:-/data/haziq/mocap/data/fit3d}"

SHARD=0
NUM_SHARDS=1

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --DATA_ROOT)
      DATA_ROOT="$2"; shift 2 ;;
    --shard)
      SHARD="$2"; shift 2 ;;
    --num_shards)
      NUM_SHARDS="$2"; shift 2 ;;
    *)
      echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT"
  exit 1
fi

if [[ ! -f "$SMPL_TO_MHR_PY" ]]; then
  echo "[ERROR] Script not found: $SMPL_TO_MHR_PY"
  exit 1
fi

echo "=============================================="
echo "SMPL-X → MHR batch conversion"
echo "DATA_ROOT  : $DATA_ROOT"
echo "SCRIPT     : $SMPL_TO_MHR_PY"
echo "TEST_MODE  : $TEST_MODE"
echo "FORCE      : $FORCE"
echo "VIDEO      : $VIDEO"
echo "SHARD      : $SHARD / $NUM_SHARDS"
echo "=============================================="

# ── Collect all smplx JSON files ──────────────────────────────────────────────
# Expected: $DATA_ROOT/{train,val}/<subject>/smplx/<action>.json
mapfile -t JSONS < <(
  find -L "$DATA_ROOT" -type f \
    \( -path "*/train/*/smplx/*.json" -o -path "*/val/*/smplx/*.json" \) \
  | sort
)

echo "[INFO] Total JSON files found: ${#JSONS[@]}"

if (( ${#JSONS[@]} == 0 )); then
  echo "[WARN] No JSON files matched. Check paths under $DATA_ROOT/(train|val)/*/smplx/*.json"
  exit 0
fi

# ── Process each JSON ─────────────────────────────────────────────────────────
for idx in "${!JSONS[@]}"; do
  json_path="${JSONS[$idx]}"

  # Shard filter
  if (( idx % NUM_SHARDS != SHARD )); then
    continue
  fi

  # Derive subject dir and action name from path
  # Layout: $DATA_ROOT/<split>/<subject>/smplx/<action>.json
  action="$(basename "$json_path" .json)"
  subj_dir="$(dirname "$(dirname "$json_path")")"   # .../train/<subject>
  split="$(basename "$(dirname "$subj_dir")")"
  subj="$(basename "$subj_dir")"

  out_dir="$subj_dir/mhr"
  out_npz="$out_dir/${action}.npz"

  echo "================================================"
  echo "IDX    : $idx"
  echo "SPLIT  : $split"
  echo "SUBJ   : $subj"
  echo "ACTION : $action"
  echo "JSON   : $json_path"
  echo "OUT    : $out_npz"
  echo "================================================"

  out_mp4="${out_npz%.npz}.mp4"
  _npz_ok=0;  [[ -f "$out_npz" ]] && _npz_ok=1
  _vid_ok=0;  [[ "$VIDEO" -eq 0 || -f "$out_mp4" ]] && _vid_ok=1

  if [[ "$FORCE" -ne 1 && "$FORCE_VIDEO" -ne 1 && "$_npz_ok" -eq 1 && "$_vid_ok" -eq 1 ]]; then
    echo "[SKIP] .npz and video both exist (use FORCE=1 to overwrite, FORCE_VIDEO=1 to re-render video only)"
    echo
    continue
  fi

  EXTRA_ARGS=()
  [[ "$VIDEO" -eq 0 ]] && EXTRA_ARGS+=(--no_video)

  if [[ "$_npz_ok" -eq 1 && ( "$_vid_ok" -eq 0 || "$FORCE_VIDEO" -eq 1 ) ]]; then
    echo "[INFO] .npz exists — rendering video only"
    EXTRA_ARGS+=(--video_only)
  fi

  if [[ "$TEST_MODE" -eq 1 ]]; then
    echo "[TEST_MODE] Would run:"
    echo "  conda run --no-capture-output -n mhr_new python \"$SMPL_TO_MHR_PY\" \\"
    echo "    --smplx_json \"$json_path\" \\"
    echo "    --out_npz    \"$out_npz\" ${EXTRA_ARGS[*]:+\"${EXTRA_ARGS[*]}\"}"
    echo
    continue
  fi

  mkdir -p "$out_dir"

  # Run from the script's directory so relative imports (conversion.py, assets/)
  # resolve correctly, matching the usage in the smpl_to_mhr.py docstring.
  (
    cd "$SCRIPT_DIR"
    conda run --no-capture-output -n mhr_new python "$SMPL_TO_MHR_PY" \
      --smplx_json "$json_path" \
      --out_npz    "$out_npz" \
      "${EXTRA_ARGS[@]}"
  )

  echo
done

echo "[DONE] $(date)"
