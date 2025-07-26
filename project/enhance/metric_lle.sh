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
datasets=(
    ### Unpaired
    "dicm"
    #"lime"
    #"mef"
    #"npe"
    #"vv"
    ### LOLs
    #"lolv1"
    #"lolv2real"
    #"lolv2syn"
    ### LSRW
    #"lsrw"
    ### FiveK
    #"fiveka"
    #"fivekb"
    #"fivekc"
    #"fivekd"
    #"fiveke"
    ### SICE
    #"sice"
    ### High-Level
    #"darkface"
    #"exdark"
    #"lolistreettest"
    #"lolistreetval"
)
imgsz=512
metrics=(
    "psnr"
    "ssimc"
    "lpips"
    "ilniqe"
    "niqe"
    "pi"
)

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
project_dir=$(dirname "${current_dir}")
root_dir=$(dirname "${project_dir}")
run_dir="${project_dir}/run"

declare -A input_subdirs=(
    ["fiveka"]="fivek/pred"
    ["fivekb"]="fivek/pred"
    ["fivekc"]="fivek/pred"
    ["fivekd"]="fivek/pred"
    ["fiveke"]="fivek/pred"
)
declare -A target_subdirs=(
    ["lolv2real"]="lolv2/real/test/ref"
    ["lolv2syn"]="lolv2/syn/test/ref"
    ["fiveka"]="fivek/test/ref_a"
    ["fivekb"]="fivek/test/ref_b"
    ["fivekc"]="fivek/test/ref_c"
    ["fivekd"]="fivek/test/ref_d"
    ["fiveke"]="fivek/test/ref_e"
    ["sice"]="sice/sice/test/ref"
    ["lolistreetval"]="lolistreet/val/ref"
    ["lolistreettest"]="lolistreet/test/ref"
)

# ----- Main -----
cd "${run_dir}" || exit

# Parse metrics arguments
metric_args=()
for metric in "${metrics[@]}"; do
    metric_args+=("--metric" "${metric}")
done

for data in "${datasets[@]}"; do
    # Input
    input_subdir="${input_subdirs[$data]:-${data}/pred}"
    declare -a input_dirs
    mapfile -t input_dirs < <(find "$current_dir/run/predict" -type d -path "*/${arch}/${model}/${input_subdir}" 2>/dev/null | sort)

    # Target
    target_subdir="${target_subdirs[$data]:-${data}/test/ref}"
    target_dir="${current_dir}/data/${target_subdir}"
    # Fallback target_dir if not found
    [[ ! -d "${target_dir}" ]] && target_dir="${root_dir}/data/enhance/${target_subdir}"

    # Determine IQA type
    # use_gt_mean=$([[ -d "${target_dir}" ]] && echo "--use-gt-mean" || echo "")
    use_gt_mean=$(echo "")

    for input_dir in "${input_dirs[@]}"; do
        data_subdir=$(dirname "${input_dir}")
        model_subdir=$(dirname "${data_subdir}")
        model_name=$(basename "${model_subdir}")
        arch_subdir=$(dirname "${model_subdir}")
        arch_name=$(basename "${arch_subdir}")

        device=$(get_device)

        python -W ignore metric_iqa.py \
            --input-dir "${input_dir}" \
            --target-dir "${target_dir}" \
            --result-file "${current_dir}" \
            --arch "${arch_name}" \
            --model "${model_name}" \
            --data "${data}" \
            --device "${device}" \
            --imgsz "${imgsz}" \
            --backend "pyiqa" \
            "${metric_args[@]}" \
            ${use_gt_mean} \
            "$@"
    done
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
