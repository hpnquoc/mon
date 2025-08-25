#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Import -----
source ./utils.sh

# ----- Input -----
task="lle"
mode="predict"
arch=""
model=""
datasets=(
    ### Unpaired
    "dicm"
    "lime"
    "mef"
    "npe"
    "vv"
    ### LOLs
    "lolv1"
    "lolv2real"
    "lolv2syn"
    ### LSRW
    "lsrw"
    ### SICE
    "sice"
    ### FiveK
    "fivek"
    #"fiveka"
    #"fivekb"
    #"fivekc"
    #"fivekd"
    #"fiveke"
    ### UHD
    "uhdll"
    ### High-Level
    "darkface"
    "exdark"
    "lolistreettest"
    "lolistreetval"
    "ydld"
)
data=$(printf "%s, " "${datasets[@]}")
data=${data%, }  # Remove trailing ", "

# ----- Directory & File -----
current_file=$(readlink -f "${0}")
current_dir=$(dirname "${current_file}")
project_dir=$(dirname "${current_dir}")
root_dir=$(dirname "${project_dir}")
run_dir="${project_dir}/run"

# ----- Main -----
cd "${run_dir}" || exit

device=$(get_device)

python -W ignore main.py \
    --root "${current_dir}" \
    --task "${task}" \
    --mode "${mode}" \
    --arch "${arch}" \
    --model "${model}" \
    --data "${data}" \
    --device "${device}" \
    --save-result \
    --save-image \
    --save-debug \
    --exist-ok \
    --verbose \
    "$@"

# ----- Done -----
cd "${current_dir}" || exit
exit 0
