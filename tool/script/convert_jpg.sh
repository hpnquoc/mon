#!/bin/bash
echo "$HOSTNAME"
clear

# Remember to install ImageMagick first:
# sudo apt-get install imagemagick

# Directories
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")  # mon/tool/script/
tool_dir=$(dirname "$current_dir")      # mon/tool/
mon_dir=$(dirname "$tool_dir")          # mon/
data_dir="${mon_dir}/data"              # mon/data/

# Define the directory where you want to perform the recursive image conversion
directory="/media/longpham/hdd_01/50_archives/20_projects/hcmc_tss_2013_2017"
# directory="${data_dir}/mix_aicity_kodas"

# Replace "-" with "_" in directory names (recursively)
cd "${directory}" || exit
find . -type f -regex ".*\.\(bmp\|png\|webp\)" -exec mogrify -format jpg {} \; -print
find . -type f -regex ".*\.\(bmp\|png\|webp\)" -exec rm {} \; -print