#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
task="detect"
mode="train"
arch=""
model=""

# ----- Directory -----
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
root_dir=$(dirname "$project_dir")
run_dir="${project_dir}/run"

# ----- Main -----
cd "${run_dir}" || exit
python -W ignore main.py \
    --root "${current_dir}" \
    --task "${task}" \
    --mode "${mode}" \
    --arch "${arch}" \
    --model "${model}" \
    --save-result \
    --save-image \
    --save-debug \
    --exist-ok \
    --verbose \
    --torchrun \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
