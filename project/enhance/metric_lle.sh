#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Input -----
arch="zerolinr"
model="zerolinr"
datasets=(
    ### Unpaired Set
    "dicm"
    "lime"
    "mef"
    "npe"
    "vv"
    ### LOLs Set
    "lolv1"
    "lolv2real"
    "lolv2syn"
    ### FiveK Set
    #"fiveka"
    #"fivekb"
    "fivekc"
    #"fivekd"
    "fiveke"
    ### SICE Set
    "sice"
    "sicegrad"
    "sicemix"
    ### Camera-Specific Set
    "sidsony"
    ### Real-World Set
    "darkcityscapes"
    # "darkface"
    "darkface496"
    # "exdark"
    "exdark1200"
    "lolistreettest"
    "lolistreetval"
    "nightcity"
)
device="cuda:0"
metrics=(
    "psnr"
    "ssimc"
    "psnry"
    "ssim"
    "ms_ssim"
    "lpips"
    "brisque"
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
    ["sicegrad"]="sice/grad/test/ref"
    ["sicemix"]="sice/mix/test/ref"
    ["sidsony"]="sid/sony/test/ref"
    ["exdark1200"]="exdark/test1200/ref"
    ["lolistreetval"]="lolistreet/val/ref"
    ["lolistreettest"]="lolistreet/test/ref"
)

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
    input_subdir="${input_subdirs[$dn]:-${data}/pred}"
    input_dir="${current_dir}/run/predict/${arch}/${model}/${input_subdir}"
    check_dir "${input_dir}"

    # Target
    target_subdir="${target_subdirs[$dn]:-${data}/test/ref}"
    target_dir="${current_dir}/data/${target_subdir}"
    # Fallback target_dir if not found
    [[ ! -d "${target_dir}" ]] && target_dir="${root_dir}/data/enhance/${target_subdir}"

    # Determine IQA type
    use_gt_mean=$([[ -d "${target_dir}" ]] && echo "--use-gt-mean" || echo "")

    # Run IQA evaluation
    metric_args=()
    for metric in "${metrics[@]}"; do
      metric_args+=("--metric" "${metric}")
    done
    python -W ignore metric_iqa.py \
        --input-dir "${input_dir}" \
        --target-dir "${target_dir}" \
        --result-file "${current_dir}" \
        --arch "${arch}" \
        --model "${model}" \
        --data "${data}" \
        --device "${device}" \
        --imgsz 512 \
        --backend "pyiqa" \
        "${metric_args[@]}" \
        ${use_gt_mean} \
        "$@"
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
