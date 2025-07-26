#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
#arch="*"                                # * for all architectures
#model="*"                               # * for all models
arch="zinf"
model="zinf"
detector="deim_dfine_s_coco80"
#detector="deim_dfine_s_widerface"
datasets=(
    ### High-Level
    #"darkface"
    #"exdark"
    "lolistreetval"
)
bbox_format="yolo"
#exist_ok=$(echo "")
exist_ok=$(echo "--exist-ok")

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
project_dir=$(dirname "${current_dir}")
root_dir=$(dirname "${project_dir}")
run_dir="${project_dir}/run"

declare -A target_jsons=(
    ["darkface"]="darkface/test/test.json"
    ["exdark"]="exdark/test/test.json"
    ["lolistreetval"]="lolistreet/val/ref.json"
)

declare -A remaps=(
    ["darkface"]=""
    ["exdark"]="exdark/remap_coco802exdark.yaml"
    ["lolistreetval"]=""
)

# ----- Main -----
cd "${run_dir}" || exit

for data in "${datasets[@]}"; do
    # Input
    declare -a input_dirs
    mapfile -t input_dirs < <(find "${current_dir}/run/predict" -type d -path "*/${arch}/${model}/${data}/pred" 2>/dev/null | sort)

    # Target
    target_json="${target_jsons[$data]:-${data}/test/test.json}"
    target_json="${current_dir}/data/${target_json}"
    remap="${remaps[${data}]:-""}"
    [[ "${remap}" != "" ]] && remap="${current_dir}/data/${remap}"

    for input_dir in "${input_dirs[@]}"; do
        data_subdir=$(dirname "${input_dir}")
        model_subdir=$(dirname "${data_subdir}")
        model_name=$(basename "${model_subdir}")
        arch_subdir=$(dirname "${model_subdir}")
        arch_name=$(basename "${arch_subdir}")

        label_dir="${data_subdir}/pred_${detector}/label"
        # Fallback label_dir if not found
        [[ ! -d "${label_dir}" ]] && label_dir="${data_subdir}/pred_${detector}"
        input_json="${data_subdir}/pred_${detector}.json"

        device=$(get_device)

        # Measure COCO
        python -W ignore metric_coco.py \
            --input-dir "${input_dir}" \
            --label-dir "${label_dir}" \
            --input-json "${input_json}" \
            --target-json "${target_json}" \
            --result-file "${current_dir}" \
            --remap "${remap}" \
            --arch "${arch_name}" \
            --model "${model_name}" \
            --data "${data}" \
            --device "${device}" \
            --bbox-format "${bbox_format}" \
            ${exist_ok} \
            "$@"
    done
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
