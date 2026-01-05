ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=nb_frames \
  -of default=nokey=1:noprint_wrappers=1 \
  "/media/haziq/Haziq/mocap/data/self/train/haziq/mhr/laptop_webcam/20260104_001543_rendered.mp4"

ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=nb_frames \
  -of default=nokey=1:noprint_wrappers=1 \
  "/media/haziq/Haziq/mocap/data/self/train/haziq/videos/laptop_webcam/20260104_001543.mp4"

python - << 'EOF'
import numpy as np
npz = np.load("/media/haziq/Haziq/mocap/data/self/train/haziq/mhr/laptop_webcam/20260104_001543_mhr_outputs.npz", allow_pickle=True)
print("Number of frames in NPZ:", len(npz["frame_indices"]))
EOF

python - << 'EOF'
import json
p="/media/haziq/Haziq/mocap/data/self/train/haziq/smplx/laptop_webcam/20260104_001543.json"
with open(p) as f:
    j=json.load(f)
print("num_frames:", len(j["transl"]))
EOF

#######################################

ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=nb_frames \
  -of default=nokey=1:noprint_wrappers=1 \
  "/media/haziq/Haziq/mocap/data/self/train/haziq/mhr/laptop_webcam/20260104_002401_rendered.mp4"

ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=nb_frames \
  -of default=nokey=1:noprint_wrappers=1 \
  "/media/haziq/Haziq/mocap/data/self/train/haziq/videos/laptop_webcam/20260104_002401.mp4"

python - << 'EOF'
import numpy as np
npz = np.load("/media/haziq/Haziq/mocap/data/self/train/haziq/mhr/laptop_webcam/20260104_002401_mhr_outputs.npz", allow_pickle=True)
print("Number of frames in NPZ:", len(npz["frame_indices"]))
EOF

python - << 'EOF'
import json
p="/media/haziq/Haziq/mocap/data/self/train/haziq/smplx/laptop_webcam/20260104_002401.json"
with open(p) as f:
    j=json.load(f)
print("num_frames:", len(j["transl"]))
EOF
