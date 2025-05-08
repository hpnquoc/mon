#!/bin/bash

echo "$HOSTNAME"
clear

# ----- Input -----
data="dicm, fusion, lime, mef, npe, vv, lol_v1, lol_v2_real, lol_v2_synthetic, fivek, sice, sice_grad, sice_mix, sid_sony, darkcityscapes, darkface, exdark, loli_street_test, loli_street_val, nightcity"

# ----- Directory -----
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
root_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"

# ----- Run -----
cd "${runml_dir}" || exit
python -W ignore main.py \
  --root "${current_dir}" \
  --task "lle" \
  --mode "predict" \
  --data "${data}" \
  --benchmark \
  --save-image \
  --save-debug \
  --exist-ok \
  --verbose \
  "$@"

# ----- Done -----
cd "${current_dir}" || exit
