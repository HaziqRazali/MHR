#!/bin/bash
#./mhr_to_smpl.sh | tee mhr_to_smpl.log
set -e

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_ROOT="/home/haziq/datasets/mocap/data/kit/train"
SCRIPT_PATH="/home/haziq/MHR/tools/mhr_smpl_conversion/mhr_to_smpl.py"

# ------------------------------------------------------------
# Loop over subject folders
# ------------------------------------------------------------
for SUBJ_DIR in "$DATA_ROOT"/*; do
    # Skip if not a directory
    [ -d "$SUBJ_DIR" ] || continue

    SUBJ_NAME=$(basename "$SUBJ_DIR")

    MHR_DIR="$SUBJ_DIR/mhr/cam1"
    SMPLX_DIR="$SUBJ_DIR/smplx/cam1"

    # Skip if no mhr/cam1 directory
    if [ ! -d "$MHR_DIR" ]; then
        echo "[SKIP] $SUBJ_NAME: no mhr/cam1 directory"
        continue
    fi

    # Create output directory
    mkdir -p "$SMPLX_DIR"

    echo "=============================================="
    echo "Processing subject: $SUBJ_NAME"
    echo "MHR dir:   $MHR_DIR"
    echo "SMPLX dir: $SMPLX_DIR"
    echo "=============================================="

    # ------------------------------------------------------------
    # Loop over .npz files
    # ------------------------------------------------------------
    for MHR_FILE in "$MHR_DIR"/*.npz; do
        # Handle case where glob matches nothing
        [ -e "$MHR_FILE" ] || continue

        FILE_NAME=$(basename "$MHR_FILE" .npz)
        OUT_JSON="$SMPLX_DIR/${FILE_NAME}.json"

        echo "[RUN] $SUBJ_NAME / cam1 / $FILE_NAME"

        python "$SCRIPT_PATH" \
            --mhr_path "$MHR_FILE" \
            --out_json "$OUT_JSON" \
            --show 0 \
            --frame_id -1
    done
done

echo "✅ All conversions finished."
