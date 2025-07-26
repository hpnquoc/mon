#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
arch="dav2"
model="dav2_vitb"
datasets=(
    ### Unpaired
    #"dicm"
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
    #"nightcity"
)

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
project_dir=$(dirname "${current_dir}")
root_dir=$(dirname "${project_dir}")
run_dir="${project_dir}/run"

declare -A input_subdirs=(
    ["lolv2real"]="lolv2/real/test/image"
    ["lolv2syn"]="lolv2/syn/test/image"
    ["fiveka"]="fivek/test/image"
    ["fivekb"]="fivek/test/image"
    ["fivekc"]="fivek/test/image"
    ["fivekd"]="fivek/test/image"
    ["fiveke"]="fivek/test/image"
    ["sice"]="sice/sice/test/image"
    ["lolistreetval"]="lolistreet/val/image"
    ["lolistreettest"]="lolistreet/test/image"
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

for data in "${datasets[@]}"; do
    # Input
    input_subdir="${input_subdirs[$data]:-${data}/test/image}"
    input_dir="${current_dir}/data/${input_subdir}_${model}"
    check_dir "${input_dir}"

    # Target
    target_subdir="${target_subdirs[$data]:-${data}/test/ref}"
    target_dir="${current_dir}/data/${target_subdir}_${model}"
    check_dir "${target_dir}"

    device=$(get_device)

    # Run evaluation
    python -W ignore metric_depth.py \
        --input-dir "${input_dir}" \
        --target-dir "${target_dir}" \
        --result-file "${current_dir}" \
        --arch "${arch}" \
        --model "${model}" \
        --data "${data}" \
        --device "${device}" \
        --imgsz 512 \
        --use-color \
        "$@"
done

# ----- Done -----
cd "${current_dir}" || exit
exit 0
