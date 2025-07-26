#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
task="depth"
mode="predict"
arch="daac"
model="daac_vits"
datasets=(

)
data=$(printf "%s, " "${datasets[@]}")
data=${data%, }  # Remove trailing ", "

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
project_dir=$(dirname "${current_dir}")
root_dir=$(dirname "${project_dir}")
run_dir="${project_dir}/run"

# ----- Main -----
cd "${run_dir}" || exit

device=$(get_device)

python -W ignore main.py \
    --root "${current_dir}" \
    --task "${task}" \
    --mode "${mode}" \
    --arch "${arch}" \
    --model "${model}" \
    --config 0 \
    --data "${data}" \
    --device "${device}" \
    --save-result \
    --save-image \
    --use-fullname \
    --save-nearby \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
