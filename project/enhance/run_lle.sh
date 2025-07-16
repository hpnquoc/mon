#!/bin/bash

echo "${HOSTNAME}"
clear

# ----- Input -----
task="lle"
mode="predict"
arch=""
model=""
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
    ### LSRW Set
    "lsrw"
    ### FiveK Set
    "fivek"
    #"fiveka"
    #"fivekb"
    #"fivekc"
    #"fivekd"
    #"fiveke"
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
