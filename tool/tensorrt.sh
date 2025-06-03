#!/bin/bash

# https://medium.com/@moshiur.faisal01/install-tensorrt-with-command-line-wrapper-trtexec-on-ununtu-20-04-lts-3e44f4f36a2b

clear
echo "$HOSTNAME"

# ----- Input -----


# ----- Globals -----
# Directory
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")  # mon/tool/
root_dir=$(dirname "$current_dir")      # mon/

# ----- Check -----
check_cuda() {
    if command -v nvcc >/dev/null 2>&1; then
        # echo "CUDA is installed. Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c 2-)"
        return 0
    elif command -v nvidia-smi >/dev/null 2>&1; then
        # echo "NVIDIA driver is installed. CUDA Version: $(nvidia-smi | grep CUDA | awk '{print $6}')"
        return 0
    else
        # echo "CUDA is not installed or not detected."
        return 1
    fi
}

# ----- Setup -----
install_tensorrt() {
    echo -e "\nInstall TensorRT"

    if check_cuda; then
        echo "CUDA is installed. Proceeding with TensorRT installation."
    else
        echo "CUDA is not installed. Please install CUDA first."
        exit 1
    fi

    # Install TensorRT
    sudo apt-get update
    sudo apt-get install tensorrt
    sudo apt-get install onnx-graphsurgeon
    sudo apt autoremove

    # Build trtexec
    cd /usr/src/tensorrt/samples/trtexec || exit
    sudo make CUDA_INSTALL_DIR=/usr/local/cuda/bin TRT_LIB_DIR=/usr/src/tensorrt/bin
    sudo cp /usr/src/tensorrt/bin/trtexec /usr/local/bin/
}

# ----- Main -----
install_tensorrt

# ----- Done -----
exit 0
