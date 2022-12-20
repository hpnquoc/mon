#!/bin/bash

# Install:
# chmod +x install.sh
# conda init bash
# ./install.sh

script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
root_dir=$(dirname "$current_dir")


# Add `conda-forge` channel
echo -e "\nAdd 'conda-forge' channel:"
conda config --append channels conda-forge

# Add `nvidia` channel
echo -e "\nAdd 'nvidia' channel:"
conda config --append channels nvidia

# Install `mamba`
echo -e "\nInstall 'mamba':"
conda install -c conda-forge mamba --y

# Update 'base' env
echo -e "\nUpdate 'base' environment:"
mamba update --a --y
pip install --upgrade pip


case "$OSTYPE" in
  linux*)
    echo -e "\nLinux / WSL"
    # Create `one` env
    env_yml_path="${current_dir}/environment.yml"
    if { mamba env list | grep 'one'; } >/dev/null 2>&1; then
      echo -e "\nUpdate 'one' environment:"
      mamba env update --name one -f "${env_yml_path}"
    else
      echo -e "\nCreate 'one' environment:"
      mamba env create -f "${env_yml_path}"
    fi
    eval -e "$(mamba shell.bash hook)"
    mamba activate one
    pip install --upgrade pip
    # Install `torch-tensorrt`
    echo -e "\nInstall 'torch-tensorrt':"
    pip install torch-tensorrt==1.3.0 --find-links https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.3.0
    # Install `mish-cuda`
    # echo "Install 'mish-cuda':"
    # pip install git+https://github.com/JunnYu/mish-cuda.git
    # Remove `cv2/plugin` folder
    rm -rf $CONDA_PREFIX/lib/python3.9/site-packages/cv2/qt/plugins
    ;;
  darwin*)
    echo -e "\nMacOS"
    # Create `one` env
    env_yml_path="${current_dir}/environment_mac.yml"
    if { mamba env list | grep 'one'; } >/dev/null 2>&1; then
      echo -e "\nUpdate 'one' environment:"
      mamba env update --name one -f "${env_yml_path}"
    else
      echo -e "\nCreate 'one' environment:"
      mamba env create -f "${env_yml_path}"
    fi
    eval "$(mamba shell.bash hook)"
    mamba activate one
    pip install --upgrade pip
    # Remove `cv2/plugin` folder:
    rm -rf $CONDA_PREFIX/lib/python3.9/site-packages/cv2/qt/plugins
    ;;
  win*)
    echo -e "\nWindows"
    # Create `one` env
    env_yml_path="${current_dir}/environment.yml"
    if { mamba env list | grep 'one'; } >/dev/null 2>&1; then
      echo -e "\nUpdate 'one' environment:"
      mamba env update --name one -f "${env_yml_path}"
    else
      echo -e "\nCreate 'one' environment:"
      mamba env create -f "${env_yml_path}"
    fi
    eval "$(mamba shell.bash hook)"
    mamba activate one
    pip install --upgrade pip
    # Remove `cv2/plugin` folder:
    rm -rf $CONDA_PREFIX/lib/python3.9/site-packages/cv2/qt/plugins
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


# Set environment variables
# shellcheck disable=SC2162
data_dir="/data"
if [ ! -d "$data_dir" ];
then
  data_dir="${root_dir}/data"
fi
read -e -i "$data_dir" -p "Enter DATA_DIR=" input
data_dir="${input:-$data_dir}"
if [ "$data_dir" != "" ]; then
  export DATA_DIR="$data_dir"
  mamba env config vars set data_dir="$data_dir"
  echo -e "\nDATA_DIR has been set to $data_dir."
else
  echo -e "\nDATA_DIR has NOT been set."
fi
if [ -d "$root_dir" ];
then
  echo -e "\nDATA_DIR=$data_dir" > "${root_dir}/pycharm.env"
fi


# Setup resilio sync
rsync_dir="${root_dir}/.sync"
mkdir -p "${rsync_dir}"
cp "IgnoreList" "${rsync_dir}/IgnoreList"


# Setup mamba
mamba init
