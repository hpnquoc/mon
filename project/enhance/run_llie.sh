#!/bin/bash
echo "$HOSTNAME"
clear

# Directories
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
mon_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"
data_dir="${mon_dir}/data"

# Run
cd "${runml_dir}" || exit
python -W ignore main.py \
    --root "${current_dir}" \
    --task "llie" \
    --mode "predict" \
    --data "dicm, fusion, lime, mef, npe, vv, lol_v1, lol_v2_real, lol_v2_synthetic, fivek_e, sice, sice_grad, sice_mix_v2, sid_sony, loli_street_val, loli_street_test, nightcity" \
    --verbose \
    "$@"

# --data "dicm, fusion, lime, mef, npe, vv, lol_v1, lol_v2_real, lol_v2_synthetic, fivek_e, sice, sice_grad, sice_mix_v2, sid_sony, loli_street_val, loli_street_test, nightcity" \

# Done
cd "${current_dir}" || exit
