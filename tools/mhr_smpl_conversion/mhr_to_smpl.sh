#!/bin/bash
# TEST_MODE=0 FORCE=0 ./mhr_to_smpl.sh --data-root /home/haziq/datasets/mocap/data/laptop_webcam | tee mhr_to_smpl_new.log
set -euo pipefail
shopt -s nullglob

# ------------------------------------------------------------
# TEST MODE
# TEST_MODE=1 → DRY RUN
# TEST_MODE=0 → ACTUAL RUN
# ------------------------------------------------------------
TEST_MODE="${TEST_MODE:-1}"
DRY_RUN="$TEST_MODE"
FORCE="${FORCE:-0}"

# ------------------------------------------------------------
# Defaults (can be overridden)
# ------------------------------------------------------------
DEFAULT_DATA_ROOT="/media/haziq/Haziq/mocap/data/kit"
SCRIPT_PATH="/home/haziq/MHR/tools/mhr_smpl_conversion/mhr_to_smpl.py"

# ------------------------------------------------------------
# Resolve DATA_ROOT
# Priority: CLI arg > ENV var > default
# ------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[ERROR] Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "=============================================="
echo "MHR → SMPL-X conversion"
echo "DATA_ROOT : $DATA_ROOT"
echo "SCRIPT    : $SCRIPT_PATH"
echo "TEST_MODE : $TEST_MODE"
echo "FORCE     : $FORCE"
echo "=============================================="

# ------------------------------------------------------------
# Loop over split folders (train, val)
# ------------------------------------------------------------
for SPLIT_DIR in "$DATA_ROOT"/train "$DATA_ROOT"/val; do
    [ -d "$SPLIT_DIR" ] || {
        echo "[SKIP] Split dir missing: $SPLIT_DIR"
        continue
    }

    SPLIT_NAME=$(basename "$SPLIT_DIR")

    echo
    echo "=============================================="
    echo "Processing split: $SPLIT_NAME"
    echo "Split dir: $SPLIT_DIR"
    echo "=============================================="

    # ------------------------------------------------------------
    # Loop over subject folders
    # ------------------------------------------------------------
    for SUBJ_DIR in "$SPLIT_DIR"/*; do
        [ -d "$SUBJ_DIR" ] || continue

        SUBJ_NAME=$(basename "$SUBJ_DIR")
        MHR_ROOT="$SUBJ_DIR/sam3d"

        # Skip if no sam3d directory
        if [ ! -d "$MHR_ROOT" ]; then
            echo "[SKIP] $SPLIT_NAME/$SUBJ_NAME: no sam3d directory"
            continue
        fi

        # ------------------------------------------------------------
        # Loop over camera folders under mhr/<camera_name>
        # ------------------------------------------------------------
        for CAM_DIR in "$MHR_ROOT"/*; do
            [ -d "$CAM_DIR" ] || continue

            CAM_NAME=$(basename "$CAM_DIR")
            MHR_DIR="$CAM_DIR"
            SMPLX_DIR="$CAM_DIR"

            echo "=============================================="
            echo "Processing subject: $SPLIT_NAME/$SUBJ_NAME"
            echo "Camera:    $CAM_NAME"
            echo "MHR dir:   $MHR_DIR"
            echo "SMPLX dir: $SMPLX_DIR"
            echo "=============================================="

            # ------------------------------------------------------------
            # Create output directory
            # ------------------------------------------------------------
            if [ "$DRY_RUN" -eq 1 ]; then
                echo "[TEST_MODE] mkdir -p $SMPLX_DIR"
            else
                mkdir -p "$SMPLX_DIR"
            fi

            # ------------------------------------------------------------
            # Loop over .npz files
            # ------------------------------------------------------------
            FOUND=0
            for MHR_FILE in "$MHR_DIR"/*.npz; do
                [ -e "$MHR_FILE" ] || continue
                FOUND=1

                FILE_NAME=$(basename "$MHR_FILE" .npz)
                # Always remove trailing "_mhr_outputs" if present
                FILE_NAME="${FILE_NAME%_mhr_outputs}"

                OUT_JSON="$SMPLX_DIR/${FILE_NAME}_smplx.json"

                echo "[RUN] $SPLIT_NAME / $SUBJ_NAME / $CAM_NAME / $FILE_NAME"

                if [[ -f "$OUT_JSON" && "$FORCE" -ne 1 ]]; then
                    echo "[SKIP] JSON already exists (use FORCE=1 to overwrite): $OUT_JSON"
                    continue
                fi

                if [ "$DRY_RUN" -eq 1 ]; then
                    echo "[TEST_MODE] python $SCRIPT_PATH \\"
                    echo "            --mhr_path $MHR_FILE \\"
                    echo "            --out_json $OUT_JSON \\"
                    echo "            --show 0 \\"
                    echo "            --frame_id -1"
                else
                    python "$SCRIPT_PATH" \
                        --mhr_path "$MHR_FILE" \
                        --out_json "$OUT_JSON" \
                        --show 0 \
                        --frame_id -1
                fi
            done

            if [ "$FOUND" -eq 0 ]; then
                echo "[INFO] $SPLIT_NAME/$SUBJ_NAME/$CAM_NAME: no .npz files found"
            fi
        done
    done
done

echo "=============================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "✅ TEST_MODE=1 finished — no files were written"
else
    echo "✅ TEST_MODE=0 finished — all conversions done"
fi
echo "=============================================="
