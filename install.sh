#!/bin/bash

# Usage:
# chmod +x install.sh
# ./install.sh

# ----- Check -----
check_gui_support() {
  if [ -n "$DISPLAY" ]; then
    # echo "GUI supported: X11 display server detected (DISPLAY=$DISPLAY)" >&2
    # echo "x11"
    return 0
  elif [ -n "$WAYLAND_DISPLAY" ]; then
    # echo "GUI supported: Wayland display server detected (WAYLAND_DISPLAY=$WAYLAND_DISPLAY)" >&2
    # echo "wayland"
    return 0
  else
    # echo "GUI not supported: No display server detected." >&2
    # echo "none"
    return 1
  fi
}

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

# ----- Utils -----
get_env_yaml_path() {
  # echo -e "\nGetting environment YAML path"
  if check_cuda; then
    echo "${current_dir}/env/cuda.yaml"
  else
    echo "${current_dir}/env/cpu.yaml"
  fi
}

# ----- Setup -----
add_channels() {
  echo -e "\nAdding channels"
  conda config --append channels conda-forge
  conda config --append channels nvidia
  conda config --append channels pytorch
  echo -e "... Done"
}

update_base_env() {
  echo -e "\nUpdating 'base' environment"
  conda update -n base -c defaults conda --y
  conda update --a --y
  pip install --upgrade pip
  pip install --upgrade pipx
  pip install --upgrade poetry
  echo -e "... Done"
}

create_mon_env_linux() {
  echo -e "\nCreating 'mon' environment:"
  # Install gcc and g++
  if sudo -n true 2>/dev/null; then
    sudo apt-get install gcc g++
  else
    apt-get install gcc g++
  fi
  # Create `mon` env
  env_yaml_path=$(get_env_yaml_path)
  conda env create -f "${env_yaml_path}"
  echo "conda activate mon" >> ~/.bashrc
  source ~/.bashrc
  echo -e "... Done"
  # Cleanup
  rm -rf $CONDA_PREFIX/lib/python3.12/site-packages/cv2/qt/plugins
}

create_mon_env_darwin() {
  echo -e "\nCreating 'mon' environment:"
  # Must be called before installing PyTorch Lightning
  export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
  export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
  # Create `mon` env
  env_yaml_path=$(get_env_yaml_path)
  conda env create -f "${env_yaml_path}"
  echo "conda activate mon" >> ~/.bash_profile
  source ~/.bash_profile
  echo -e "... Done"
  # Cleanup
  rm -rf $CONDA_PREFIX/lib/python3.12/site-packages/cv2/qt/plugins
}

create_mon_env() {
  case "$OSTYPE" in
    linux*)
        create_mon_env_linux
        ;;
    darwin*)
        create_mon_env_darwin
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
}

install_ffmpeg() {
  echo -e "\nInstalling FFMPEG"
  case "$OSTYPE" in
    linux*)
      if sudo -n true 2>/dev/null; then
          sudo apt-get install ffmpeg
          sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
      else
          apt-get install ffmpeg
          apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
      fi
      ;;
    darwin*)
      brew install ffmpeg
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
}

install_mon_package() {
  echo -e "\nInstall mon library"
  eval "$(conda shell.bash hook)"
  conda activate mon
  rm -rf poetry.lock
  if check_gui_support; then
    poetry install --extras "docs gui"
  else
    poetry install --extras "docs"
  fi
  rm -rf poetry.lock
  conda update --a --y
  conda clean  --a --y
}

setup_data_dir() {
  echo -e "\nSetting DATA_DIR"
  root_dir=$1
  data_dir="${root_dir}/data"
  read -e -i "$data_dir" -p "Enter DATA_DIR=" input
  data_dir="${input:-$data_dir}"
  if [ "$data_dir" != "" ]; then
      export DATA_DIR="$data_dir"
      conda env config vars set data_dir="$data_dir"
      echo -e "\nDATA_DIR has been set to $data_dir."
  else
      echo -e "\nDATA_DIR has NOT been set."
  fi
  echo -e "... Done"
}

setup_resilio_sync() {
  rsync_dir="${root_dir}/.sync"
  mkdir -p "${rsync_dir}"
  cp "env/IgnoreList" "${rsync_dir}/IgnoreList"
  echo -e "... Done"
}

# ----- Main -----
clear
echo "$HOSTNAME"

# Directory
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
root_dir=$current_dir

add_channels
update_base_env
create_mon_env
install_ffmpeg
install_mon_package

# Setup environment variables
# shellcheck disable=SC2162
export PYTHONDONTWRITEBYTECODE=1
# setup_data_dir "$root_dir"
setup_resilio_sync "$root_dir"
