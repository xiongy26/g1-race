#!/bin/bash
# 一次性资产下载: 天工 Tienkung2-Lite(TienKung-Lab) + 智元灵犀X1(agibot_x1_infer)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
RAW="https://raw.githubusercontent.com"

# ---------- 天工 ----------
TK_BASE="$RAW/Open-X-Humanoid/TienKung-Lab/main/legged_lab/assets/tienkung2_lite"
mkdir -p "$ROOT/assets/tienkung/meshes"
curl -sf --retry 3 "$TK_BASE/mjcf/tienkung.xml" -o "$ROOT/assets/tienkung/tienkung.xml"
TK_MESHES="pelvis hip_roll_l_link hip_pitch_l_link hip_yaw_l_link knee_pitch_l_link ankle_pitch_l_link ankle_roll_l_link \
hip_roll_r_link hip_pitch_r_link hip_yaw_r_link knee_pitch_r_link ankle_pitch_r_link ankle_roll_r_link \
shoulder_pitch_l_link shoulder_roll_l_link shoulder_yaw_l_link elbow_pitch_l_link \
shoulder_pitch_r_link shoulder_roll_r_link shoulder_yaw_r_link elbow_pitch_r_link"
for m in $TK_MESHES; do
  test -s "$ROOT/assets/tienkung/meshes/$m.STL" || curl -sf --retry 3 "$TK_BASE/meshes/$m.STL" -o "$ROOT/assets/tienkung/meshes/$m.STL"
done
curl -sf --retry 3 "$RAW/Open-X-Humanoid/TienKung-Lab/main/Exported_policy/walk.pt" -o "$ROOT/assets/tienkung/walk.pt"
echo "tienkung done: $(du -sh "$ROOT/assets/tienkung" | cut -f1)"

# ---------- 智元 X1 ----------
X1_BASE="$RAW/AgibotTech/agibot_x1_infer/main/src/module/sim_module/model"
mkdir -p "$ROOT/assets/x1/robot/xyber_x1" "$ROOT/assets/x1/environment" "$ROOT/assets/x1/meshes"
curl -sf --retry 3 "$X1_BASE/mjcf/xyber_x1_flat.xml" -o "$ROOT/assets/x1/xyber_x1_flat.xml"
curl -sf --retry 3 "$X1_BASE/mjcf/robot/xyber_x1/xyber_x1_serial.xml" -o "$ROOT/assets/x1/robot/xyber_x1/xyber_x1_serial.xml"
curl -sf --retry 3 "$X1_BASE/mjcf/environment/flat.xml" -o "$ROOT/assets/x1/environment/flat.xml"
X1_MESHES="base_link_simple lumbar_yaw lumbar_roll lumbar_pitch \
left_shoulder_pitch left_shoulder_roll left_shoulder_yaw left_elbow_pitch left_elbow_yaw left_wrist_pitch left_wrist_roll \
right_shoulder_pitch right_shoulder_roll right_shoulder_yaw right_elbow_pitch right_elbow_yaw right_wrist_pitch right_wrist_roll \
left_hip_pitch left_hip_roll left_hip_yaw left_knee_pitch left_ankle_pitch left_ankle_roll \
right_hip_pitch right_hip_roll right_hip_yaw right_knee_pitch right_ankle_pitch right_ankle_roll"
for m in $X1_MESHES; do
  test -s "$ROOT/assets/x1/meshes/$m.STL" || curl -sf --retry 3 "$X1_BASE/meshes/$m.STL" -o "$ROOT/assets/x1/meshes/$m.STL"
done
curl -sf --retry 3 "$RAW/AgibotTech/agibot_x1_infer/main/src/module/control_module/policy/rl_walk_leg.onnx" -o "$ROOT/assets/x1/policy.onnx"
echo "x1 done: $(du -sh "$ROOT/assets/x1" | cut -f1)"
