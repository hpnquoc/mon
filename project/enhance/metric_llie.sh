#!/bin/bash
echo "$HOSTNAME"
clear

# Directories
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
mon_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"
data_dir="${mon_dir}/data/enhance"

# Input
arch="zero_mie"
model="zero_mie_ms"
data=(
    ### Unpaired Set
    "dicm"
    "lime"
    "mef"
    "npe"
    "vv"
    ### LOLs Set
    "lol_v1"
    "lol_v2_real"
    "lol_v2_synthetic"
    ### FiveK Set
    # "fivek_a"
    # "fivek_b"
    # "fivek_c"
    # "fivek_d"
    "fivek_e"
    ### SICE Set
    "sice"
    "sice_grad"
    "sice_mix"
    ### Camera-Specific Set
    "sid_sony"
    ### Real-World Set
    "darkcityscapes"
    "darkface"
    "exdark"
    "loli_street_val"
    "loli_street_test"
    "nightcity"
)
device="cuda:0"

# Run
cd "${runml_dir}" || exit
for (( i=0; i<${#data[@]}; i++ )); do
    # Input
    if [ "${data[i]}" == "fivek_a" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek"
    elif [ "${data[i]}" == "fivek_b" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek"
    elif [ "${data[i]}" == "fivek_c" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek"
    elif [ "${data[i]}" == "fivek_d" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek"
    elif [ "${data[i]}" == "fivek_e" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/fivek"
    else
        input_dir="${data_dir}/#predict/${arch}/${model}/${data[i]}"
        if ! [ -d "${input_dir}" ]; then
            input_dir="${current_dir}/run/predict/${arch}/${model}/${data[i]}"
        fi
    fi

    # Target
    if [ "${data[i]}" == "fivek_e" ]; then
        target_dir="${data_dir}/fivek/test/ref_a"
    elif [ "${data[i]}" == "fivek_b" ]; then
        target_dir="${data_dir}/fivek/test/ref_b"
    elif [ "${data[i]}" == "fivek_c" ]; then
        target_dir="${data_dir}/fivek/test/ref_c"
    elif [ "${data[i]}" == "fivek_d" ]; then
        target_dir="${data_dir}/fivek/test/ref_d"
    elif [ "${data[i]}" == "fivek_e" ]; then
        target_dir="${data_dir}/fivek/test/ref_e"
    elif [ "${data[i]}" == "loli_street_val" ]; then
        target_dir="${data_dir}/loli_street/val/ref"
    elif [ "${data[i]}" == "loli_street_test" ]; then
        target_dir="${data_dir}/loli_street/test/ref"
    else
        target_dir="${data_dir}/${data[i]}/test/ref"
        if ! [ -d "${target_dir}" ]; then
            target_dir="${data_dir}/${data[i]}/val/ref"
        fi
    fi

    # Measure FR-IQA
    if [ -d "${target_dir}" ]; then
        python -W ignore metric.py \
          --input-dir "${input_dir}" \
          --target-dir "${target_dir}" \
          --result-file "${current_dir}" \
          --arch "${arch}" \
          --model "${model}" \
          --data "${data[i]}" \
          --device "${device}" \
          --imgsz 512 \
          --metric "psnr" \
          --metric "ssimc" \
          --metric "psnry" \
          --metric "ssim" \
          --metric "ms_ssim" \
          --metric "lpips" \
          --metric "brisque" \
          --metric "ilniqe" \
          --metric "niqe" \
          --metric "pi" \
          --backend "pyiqa" \
          --use-gt-mean
    # Measure NR-IQA
    else
        python -W ignore metric.py \
          --input-dir "${input_dir}" \
          --target-dir "${target_dir}" \
          --result-file "${current_dir}" \
          --arch "${arch}" \
          --model "${model}" \
          --data "${data[i]}" \
          --device "${device}" \
          --imgsz 512 \
          --metric "psnr" \
          --metric "ssimc" \
          --metric "psnry" \
          --metric "ssim" \
          --metric "ms_ssim" \
          --metric "lpips" \
          --metric "brisque" \
          --metric "ilniqe" \
          --metric "niqe" \
          --metric "pi" \
          --backend "pyiqa"
    fi

done

# Done
cd "${current_dir}" || exit
