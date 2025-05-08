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
dfine_dir="${mon_dir}/src/vision/detect/dfine"

# ----- Run -----
cd "${dfine_dir}" || exit
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=7777 \
  --nproc_per_node=4 \
  train.py \
  -c configs/dfine/dfine_hgnetv2_${model}_coco.yml \
  --use-amp \
  --seed=0


# ----- Done -----
cd "${current_dir}" || exit
