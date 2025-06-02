#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
task="detect"
mode="predict"
arch="dfine"
model="dfine_s"
data=(
    ### Real-World Set
    # "darkface"
    "darkface496"
    # "exdark"
    "exdark1200"
    "lolistreetval"
)
data_str=$(printf "%s, " "${data[@]}")
data_str=${data_str%, }  # Remove trailing ", "

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
    --data "${data_str}" \
    --save-result \
    --save-image \
    --save-debug \
    --use-fullname \
    --save-nearby \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
