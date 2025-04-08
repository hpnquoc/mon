#!/bin/bash

# Remember to install ffmpeg first:
# sudo apt-get install ffmpeg
# brew install ffmpeg

# Functions

run_on_linux() {
    cd "${directory}" || exit
    for i in $(find . -type f -regex ".*\.\(mp4\|MP4\|avi\|m4v\|mkv\|mov\|mpeg\|mpg\|wmv\)" | sort -h); do
        echo "Processing file: ${i}"
        ffmpeg \
            -i "$i" \
            -c:v libx265 \
            -c:a copy \
            -tag:v hvc1 \
            "${i%.*}_convert.mp4";
        rm "$i"
        mv "${i%.*}_convert.mp4" "${i%.*}.mp4"
    done
}

run_on_darwin() {
    cd "${directory}" || exit
    for i in $(find . -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.flv" -o -iname "*.wmv" -o -iname "*.webm" -o -iname "*.mpeg" -o -iname "*.mpg" -o -iname "*.3gp" -o -iname "*.m4v" \) | sort -h); do
        echo "Processing file: ${i}"
        ffmpeg \
            -i "$i" \
            -c:v libx265 \
            -c:a copy \
            -tag:v hvc1 \
            "${i%.*}_convert.mp4";
        rm "$i"
        mv "${i%.*}_convert.mp4" "${i%.*}.mp4"
    done
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

# Define the directory where you want to perform the recursive video conversion
# directory="${data_dir}/"
directory="/media/longpham/hdd_01/30_areas/family/photos/2009"

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
