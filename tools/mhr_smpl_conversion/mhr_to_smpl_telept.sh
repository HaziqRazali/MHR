#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

MHR_DIR="$1"   # timestamps folder

if [[ -z "${MHR_DIR:-}" ]]; then
  echo "Usage: $0 <timestamps_dir>"
  exit 1
fi

for npz in "$MHR_DIR"/*.npz; do
  json="${npz%.npz}.json"

  echo "[INFO] $npz → $json"

  python mhr_to_smpl.py \
    --mhr_path "$npz" \
    --out_json "$json"
done

#./mhr_to_smpl_telept.sh /home/haziq/datasets/telept/data/ipad/rgb_1764569430654/timestamps/
#./mhr_to_smpl_telept.sh /home/haziq/datasets/telept/data/ipad/rgb_1764569695903/timestamps/
#./mhr_to_smpl_telept.sh /home/haziq/datasets/telept/data/ipad/rgb_1764569971278/timestamps/

#python mhr_to_smpl.py \
#--mhr_path /home/haziq/datasets/telept/data/ipad/rgb_1764569430654/timestamps/ts_0001_01-38.357_f001913_data.npz \
#--out_json /home/haziq/datasets/telept/data/ipad/rgb_1764569430654/timestamps/ts_0001_01-38.357_f001913_data.json \
#--show 1 --frame_id 0
