#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Input -----
task="detect"
mode="predict"
arch="deim"
model="deim_dfine_s"
datasets=(
    ### High-Level
    #"darkface"
    "exdark"
    #"lolistreetval"
)
#resize=$(echo "")
resize=$(echo "--resize")


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

for data in "${datasets[@]}"; do
    all_data=$(find "$current_dir/run/predict" -type d -path "*/${dataset}/pred" 2>/dev/null | sort | paste -sd ',' - | sed 's/,$//')

    python -W ignore main.py \
        --root "${current_dir}" \
        --task "${task}" \
        --mode "${mode}" \
        --arch "${arch}" \
        --model "${model}" \
        --data "${all_data}" \
        --save-result \
        --save-image \
        --save-debug \
        --use-fullname \
        --save-nearby \
        --exist-ok \
        --verbose \
        ${resize} \
        "$@"
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
