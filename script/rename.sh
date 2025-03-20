#!/bin/bash

# Main
clear
echo "$HOSTNAME"

# Define the directory where you want to perform the recursive renaming
directory="/media/longpham/hdd_01/40_resources/programming_language/ios/ray_wenderlich"


# Replace "-" with "_" in filenames (recursively)
find "$directory" -depth -name '*-*' | while IFS= read -r file; do
    newname=$(echo "$file" | tr '-' '_')
    mv -v "$file" "$newname"
done

# Replace "-" with "_" in directory names (recursively)
find "$directory" -depth -type d -name '*-*' | while IFS= read -r dir; do
    newname=$(echo "$dir" | tr '-' '_')
    mv -v "$dir" "$newname"
done
