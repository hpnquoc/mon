#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Input -----
arch="fourierdiff"
model="fourierdiff"
detector="dfine_s_coco80"
datasets=(
    ### High-Level
    "darkface"
    "exdark"
    "lolistreetval"
)
device="cuda:0"
bbox_format="yolo"

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
    # Input
    input_dir="${current_dir}/run/predict/${arch}/${model}/${data}/pred"
    label_dir="${current_dir}/run/predict/${arch}/${model}/${data}/pred_${detector}/label"
    # Fallback label_dir if not found
    [[ ! -d "${label_dir}" ]] && label_dir="${current_dir}/run/predict/${arch}/${model}/${data}/pred_${detector}"
    # input_json="${label_dir}.json"
    input_json="${current_dir}/run/predict/${arch}/${model}/${data}/pred_${detector}.json"

    # Target
    if [ "${data}" == "darkface" ]; then
        target_json="${current_dir}/data/darkface/test/test.json"
        remap_classes=""
    elif [ "${data}" == "darkface496" ]; then
        target_json="${current_dir}/data/darkface/test496/test496.json"
        remap_classes=""
    elif [ "${data}" == "exdark" ]; then
        target_json="${current_dir}/data/exdark/test/test.json"
        remap_classes="${current_dir}/data/exdark/remap_coco802exdark.yaml"
    elif [ "${data}" == "exdark1200" ]; then
        target_json="${current_dir}/data/exdark/test1200/test1200.json"
        remap_classes="${current_dir}/data/exdark/remap_coco802exdark.yaml"
    elif [ "${data}" == "lolistreetval" ]; then
        target_json="${current_dir}/data/lolistreet/val/ref.json"
        remap_classes=""
    else
        target_json="${current_dir}/data/${data}/test/test.json"
        remap_classes=""
    fi

    # Measure COCO
    python -W ignore metric_coco.py \
        --input-dir "${input_dir}" \
        --label-dir "${label_dir}" \
        --input-json "${input_json}" \
        --target-json "${target_json}" \
        --result-file "${current_dir}" \
        --remap-classes "${remap_classes}" \
        --arch "${arch}" \
        --model "${model}" \
        --data "${data}" \
        --device "${device}" \
        --bbox-format "${bbox_format}" \
        --exist-ok \
        "$@"
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
