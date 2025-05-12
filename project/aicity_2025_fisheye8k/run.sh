#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----

# ----- Directory -----
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
root_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"

# ----- Main -----
cd "${runml_dir}" || exit
python -W ignore main.py \
    --root "${current_dir}" \
    --task "detect" \
    --mode "train" \
    --save-image \
    --save-debug \
    --exist-ok \
    --verbose \
    --torchrun \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
