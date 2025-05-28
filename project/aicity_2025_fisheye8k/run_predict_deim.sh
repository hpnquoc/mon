#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
arch="deim"
model="deim_dfine_l"
fullname="${model}_coco802fisheye8kv2_1280"
data="fisheye8ktest"

# ----- Directory -----
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
root_dir=$(dirname "$project_dir")
run_dir="${project_dir}/run"
deim_dir="${root_dir}/src/mon/vision/detect/deim"

# ----- Main -----
config="${deim_dir}/config/${fullname}.yaml"
# save_dir="${current_dir}/run/predict/${arch}/${model}/${fullname}"
save_dir="${current_dir}/run/predict/${arch}/${fullname}"
weights="${current_dir}/run/train/${arch}/${model}/${fullname}/best_stg2.pth"

cd "${deim_dir}" || exit
python -W ignore i_predict.py \
    --root "${current_dir}" \
    --arch "${arch}" \
    --model "${model}" \
    --config "${config}" \
    --data "${data}" \
    --fullname "${fullname}" \
    --save-dir "${save_dir}" \
    --weights "${weights}" \
    --device "cuda:0" \
    --save-result \
    --save-image \
    --save-debug \
    --exist-ok \
    --verbose \
    "$@"

cd "${current_dir}" || exit
python -W ignore make_submission.py \
    --predict-dir "${save_dir}/${data}"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
