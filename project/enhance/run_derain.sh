#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Input -----
task="derain"
mode="predict"
arch=""
model=""
datasets=(
    "rain100"
    "rain100l"
    "rain100h"
    "rain800"
    "rain1200"
    "rain1400"
    "rain2800"
    "raincityscapes"
    "gtrain"
)
data=$(printf "%s, " "${datasets[@]}")
data=${data%, }  # Remove trailing ", "

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
project_dir=$(dirname "${current_dir}")
root_dir=$(dirname "${project_dir}")
run_dir="${project_dir}/run"

# ----- Validation -----
check_file() {
    [[ ! -f "$1" ]] && { echo "File not found: $1"; exit 1; }
}

check_dir() {
    [[ ! -d "$1" ]] && { echo "Directory not found: $1"; exit 1; }
}

create_dir() {
    [[ ! -d "$1" ]] && { echo "Creating directory: $1"; mkdir -p "$1"; }
}

# ----- Main -----
cd "${run_dir}" || exit
python -W ignore main.py \
    --root "${current_dir}" \
    --task "${task}" \
    --mode "${mode}" \
    --arch "${arch}" \
    --model "${model}" \
    --data "${data}" \
    --save-result \
    --save-image \
    --save-debug \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
