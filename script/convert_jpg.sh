#!/bin/bash

# Remember to install ImageMagick first:
# sudo apt-get install imagemagick
# brew install imagemagick

# Functions

run_on_linux() {
    cd "${directory}" || exit
    find . -type f -regex ".*\.\(bmp\|heic\|png\|webp\)" -exec mogrify -format jpg {} \; -print
    find . -type f -regex ".*\.\(bmp\|heic\|png\|webp\)" -exec rm {} \; -print
}

run_on_darwin() {
    cd "${directory}" || exit
    find . -type f \( -iname "*.bmp" -o -iname "*.heic" -o -iname "*.png" -o -iname "*.webp" \) -exec mogrify -format jpg {} \; -print
    find . -type f \( -iname "*.bmp" -o -iname "*.heic" -o -iname "*.png" -o -iname "*.webp" \) -exec rm {} \; -print
}

# Main
clear
echo "$HOSTNAME"

# Directories
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")  # mon/tool/script/
tool_dir=$(dirname "$current_dir")      # mon/tool/
mon_dir=$(dirname "$tool_dir")          # mon/
data_dir="${mon_dir}/data"              # mon/data/

# Define the directory where you want to perform the recursive image conversion
# directory="${data_dir}/"
directory="/media/longpham/hdd_01/30_areas/family/photos"

case "$OSTYPE" in
    linux*)
        run_on_linux
        ;;
    darwin*)
        run_on_darwin
        ;;
    win*)
        echo -e "\nWindows"
        ;;
    msys*)
        echo -e "\nMSYS / MinGW / Git Bash"
        ;;
    cygwin*)
        echo -e "\nCygwin"
        ;;
    bsd*)
        echo -e "\nBSD"
        ;;
    solaris*)
        echo -e "\nSolaris"
        ;;
    *)
        echo -e "\nunknown: $OSTYPE"
        ;;
esac
