#!/bin/bash
#./mhr_to_smpl.sh | tee mhr_to_smpl.log
set -euo pipefail

# ------------------------------------------------------------
# DRY RUN (set to 0 to actually run)
# ------------------------------------------------------------
DRY_RUN=0

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
# /media/haziq/Haziq/mocap/data/kit/train/files_motions_1234/mhr/cam1/
DATA_ROOT="/media/haziq/Haziq/mocap/data/kit"
SCRIPT_PATH="/media/haziq/Haziq/mocap/my_scripts/MHR/tools/mhr_smpl_conversion/mhr_to_smpl.py"

echo "=============================================="
echo "MHR → SMPL-X conversion"
echo "DATA_ROOT : $DATA_ROOT"
echo "SCRIPT    : $SCRIPT_PATH"
echo "DRY_RUN   : $DRY_RUN"
echo "=============================================="

# ------------------------------------------------------------
# Loop over split folders (train, val)
# ------------------------------------------------------------
for SPLIT_DIR in "$DATA_ROOT"/train "$DATA_ROOT"/val; do
    # Skip if split directory doesn't exist
    if [ ! -d "$SPLIT_DIR" ]; then
        echo "[SKIP] Split dir missing: $SPLIT_DIR"
        continue
    fi

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
        # Skip if not a directory
        [ -d "$SUBJ_DIR" ] || continue

        SUBJ_NAME=$(basename "$SUBJ_DIR")

        MHR_DIR="$SUBJ_DIR/mhr/cam1"
        SMPLX_DIR="$SUBJ_DIR/smplx/cam1"

        # Skip if no mhr/cam1 directory
        if [ ! -d "$MHR_DIR" ]; then
            echo "[SKIP] $SPLIT_NAME/$SUBJ_NAME: no mhr/cam1 directory"
            continue
        fi

        echo "=============================================="
        echo "Processing subject: $SPLIT_NAME/$SUBJ_NAME"
        echo "MHR dir:   $MHR_DIR"
        echo "SMPLX dir: $SMPLX_DIR"
        echo "=============================================="

        # ------------------------------------------------------------
        # Create output directory
        # ------------------------------------------------------------
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "[DRY-RUN] mkdir -p $SMPLX_DIR"
        else
            mkdir -p "$SMPLX_DIR"
        fi

        # ------------------------------------------------------------
        # Loop over .npz files
        # ------------------------------------------------------------
        FOUND=0
        for MHR_FILE in "$MHR_DIR"/*.npz; do
            # Handle case where glob matches nothing
            [ -e "$MHR_FILE" ] || continue
            FOUND=1

            FILE_NAME=$(basename "$MHR_FILE" .npz)
            # Always remove trailing "_mhr_outputs" if present
            FILE_NAME="${FILE_NAME%_mhr_outputs}"

            OUT_JSON="$SMPLX_DIR/${FILE_NAME}.json"

            echo "[RUN] $SPLIT_NAME / $SUBJ_NAME / cam1 / $FILE_NAME"

            if [ "$DRY_RUN" -eq 1 ]; then
                echo "[DRY-RUN] python $SCRIPT_PATH \\"
                echo "           --mhr_path $MHR_FILE \\"
                echo "           --out_json $OUT_JSON \\"
                echo "           --show 0 \\"
                echo "           --frame_id -1"
            else
                python "$SCRIPT_PATH" \
                    --mhr_path "$MHR_FILE" \
                    --out_json "$OUT_JSON" \
                    --show 0 \
                    --frame_id -1
            fi
        done

        if [ "$FOUND" -eq 0 ]; then
            echo "[INFO] $SPLIT_NAME/$SUBJ_NAME: no .npz files found"
        fi
    done
done

echo "=============================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "✅ DRY RUN finished — no files were written"
else
    echo "✅ All conversions finished."
fi
echo "=============================================="
