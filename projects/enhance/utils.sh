#!/bin/bash

# ----- Validation -----
check_file() {
    [[ ! -f "$1" ]] && { echo "File not found: $1"; }
}

check_dir() {
    [[ ! -d "$1" ]] && { echo "Directory not found: $1"; }
}

create_dir() {
    [[ ! -d "$1" ]] && { echo "Creating directory: $1"; mkdir -p "$1"; }
}

# ----- Device -----
get_device() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "cpu"
        return
    fi

    GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
    if [ -z "$GPU_INFO" ]; then
        echo "cpu"
        return
    fi

    lowest_score=9999
    least_used_gpu=0
    while IFS=, read -r index mem_used mem_total; do
        index=$(echo "$index" | tr -d '[:space:]')
        mem_used=$(echo "$mem_used" | tr -d '[:space:]')
        mem_total=$(echo "$mem_total" | tr -d '[:space:]')
        if [ -z "$mem_used" ] || [ -z "$mem_total" ] || [ "$mem_total" -eq 0 ]; then
            # echo "Debug: Skipping GPU $index, invalid memory: used=$mem_used, total=$mem_total" >&2
            continue
        fi

        mem_usage_percent=$(( (mem_used * 100) / mem_total ))
        process_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c "GPU $index" 2>/dev/null || echo 0)
        if ! [[ "$process_count" =~ ^[0-9]+$ ]]; then
            # echo "Debug: Invalid process_count for GPU $index, setting to 0" >&2
            process_count=0
        fi

        normalized_process=$(( process_count * 10 ))
        score=$(( mem_usage_percent + normalized_process ))
        if [ "$score" -lt "$lowest_score" ]; then
            lowest_score=$score
            least_used_gpu=$index
        fi
    done <<< "$GPU_INFO"
    echo "cuda:$least_used_gpu"
}


# ----- Misc -----
unique_array() {
    local -a temp_array=()
    local -A seen
    local value
    local output_array_name="$1"
    shift

    for value in "$@"; do
        if [[ ! -v seen["$value"] ]]; then
            seen["$value"]=1
            temp_array+=("$value")
        fi
    done

    # Assign to the output array
    declare -ga "$output_array_name=(${temp_array[@]@Q})"
}
