#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
arch="deim"
model="deim_dfine_l"
imgsz=960
fullname="${model}_fisheye8kv042vD8_${imgsz}"

# ----- Directory -----
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
root_dir=$(dirname "$project_dir")
deim_dir="${root_dir}/src/mon/vision/detect/deim"

# ----- Main -----
config="${deim_dir}/config/fisheye8k/${model}/${imgsz}/${fullname}.yaml"
weights_stem="${current_dir}/run/train/${arch}/${model}/${fullname}/best_stg2"
weights="${weights_stem}.pth"
onnx_file="${weights_stem}.onnx"

cd "${deim_dir}" || exit
python -W ignore i_export.py \
    --root "${current_dir}" \
    --arch "${arch}" \
    --model "${model}" \
    --config "${config}" \
    --fullname "${fullname}" \
    --weights "${weights}" \
    --exist-ok \
    --verbose \
    "$@"

engine_file="${current_dir}/run/train/${fullname}.engine"
trtexec --onnx="${onnx_file}" --saveEngine="${engine_file}" --fp32

# ----- Done -----
cd "${current_dir}" || exit
exit 0
