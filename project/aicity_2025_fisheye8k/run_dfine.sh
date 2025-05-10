#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
model=x  # n s m l x

# ----- Directory -----
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
root_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"
dfine_dir="${root_dir}/src/mon/vision/detect/dfine"

# ----- Main -----
cd "${runml_dir}" || exit
python -W ignore main.py \
    --root "${current_dir}" \
    --task "detect" \
    --mode "train" \
    --arch "dfine" \
    --save-image \
    --save-debug \
    --exist-ok \
    --verbose \
    --torchrun \
    "$@"

# cd "${dfine_dir}" || exit
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 i_train.py --config "${dfine_dir}/config/dfine_l_obj2fisheye8k.yaml" --root "${current_dir}"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
