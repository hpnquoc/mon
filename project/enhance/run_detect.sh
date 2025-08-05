#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
task="detect"
mode="predict"
archs=(
    #"*"                                 # For all architectures
    ### Specific Architectures
    "zinf"
)
models=(
    #"*"                                 # * for all models
    ### Specific Models
    "zinf"
)
detector_arch="deim"
detector_model="deim_dfine_s"
datasets=(
    ### High-Level
    #"darkface"
    #"exdark"
    "lolistreetval"
)
#resize=$(echo "")
resize=$(echo "--resize")

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
project_dir=$(dirname "${current_dir}")
root_dir=$(dirname "${project_dir}")
run_dir="${project_dir}/run"

# ----- Main -----
cd "${run_dir}" || exit

for data in "${datasets[@]}"; do
    # Input
    all_data=""
    for arch in "${archs[@]}"; do
        for model in "${models[@]}"; do
            all_data+=$(find "$current_dir/run/predict" -type d -path "*/${arch}/${model}/${data}/pred" 2>/dev/null | sort | paste -sd ',' - | sed 's/,$//')
        done
    done

    device=$(get_device)

    python -W ignore main.py \
        --root "${current_dir}" \
        --task "${task}" \
        --mode "${mode}" \
        --arch "${detector_arch}" \
        --model "${detector_model}" \
        --data "${all_data}" \
        --device "${device}" \
        ${resize} \
        --save-result \
        --save-image \
        --save-debug \
        --use-fullname \
        --save-nearby \
        --exist-ok \
        --verbose \
        "$@"
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
