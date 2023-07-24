#!/bin/bash

echo "$HOSTNAME"

machine=$HOSTNAME
task=$1
read -e -i "$task" -p "Task [train, test, predict, ensemble]: " task

machine=$(echo $machine | tr '[:upper:]' '[:lower:]')
task=$(echo $task | tr '[:upper:]' '[:lower:]')

# Initialization
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
bin_dir=$(dirname "$current_dir")
root_dir=$(dirname "$bin_dir")
yolov8_dir="${root_dir}/src/lib/yolov8"

cd "${yolov8_dir}" || exit

# Train
if [ "$task" == "train" ]; then
  echo -e "\nTraining"
  if [ "$machine" == "lp-labdesktop-01-ubuntu" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8l-det-coco.pt" \
      --data "data/aic23-autocheckout-synthetic-117.yaml" \
      --project "${root_dir}/run/train/aic23/ablation" \
      --name "yolov8l-aic23-autocheckout-synthetic-117-640" \
      --epochs 50 \
      --batch 16 \
      --imgsz 640 \
      --workers 8 \
      --device 0 \
      --save \
      --exist-ok \
      --pretrained
  elif [ "$machine" == "vsw-ws02" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8x6-det-coco.pt" \
      --data "data/aic23-autocheckout-mix-117.yaml" \
      --project "${root_dir}/run/train/aic23" \
      --name "yolov8x6-aic23-autocheckout-mix-117-1920-02" \
      --epochs 50 \
      --batch 4 \
      --imgsz 1920 \
      --workers 8 \
      --device 0,1 \
      --save \
      --exist-ok \
      --pretrained
  elif [ "$machine" == "vsw-ws03" ]; then
    python train.py \
      --task "detect" \
      --model "${root_dir}/zoo/yolov8/yolov8x-det-coco.pt" \
      --data "data/aic23-autocheckout-mix-117.yaml" \
      --project "${root_dir}/run/train/aic23" \
      --name "yolov8x-aic23-autocheckout-mix-117-1920" \
      --epochs 50 \
      --batch 8 \
      --imgsz 1920 \
      --workers 8 \
      --device 0,1 \
      --save \
      --exist-ok \
      --pretrained
  fi
fi

# Test
if [ "$task" == "test" ]; then
  echo -e "\nTesting"
fi

# Predict
if [ "$task" == "predict" ]; then
  echo -e "\nPredicting"
  python predict.py \
  	--task "detect" \
  	--model "${root_dir}/run/train/delftbikes/yolov8x6-delftbikes-1920/weights/best.pt" \
  	--data "data/delftbikes.yaml" \
  	--project "${root_dir}/run/predict/delftbikes/" \
  	--name "submission" \
  	--source "${root_dir}/data/vipriors/delftbikes/test/images" \
  	--imgsz 2560 \
  	--conf 0.0001 \
  	--iou 0.5 \
  	--max-det 1000 \
  	--augment \
  	--device 0 \
  	--exist-ok \
  	--save-txt \
  	--save-conf \
  	--overlap-mask \
  	--box
fi

# Ensemble
if [ "$task" == "ensemble" ]; then
  echo -e "\nPredicting"
  python ensemble.py \
  	--task "detect" \
  	--model "${root_dir}/run/train/delftbikes/yolov8x6-delftbikes-2160/weights/best.pt" \
  	--model "${root_dir}/run/train/delftbikes/yolov8x-delftbikes-2160/weights/best.pt" \
  	--model "${root_dir}/run/train/delftbikes/yolov8x6-delftbikes-1920/weights/best.pt" \
  	--model "${root_dir}/run/train/delftbikes/yolov8x-delftbikes-1920/weights/best.pt" \
  	--model "${root_dir}/run/train/delftbikes/yolov8n-delftbikes-1920/weights/best.pt" \
  	--model "${root_dir}/run/train/delftbikes/yolov8x6-delftbikes-1280/weights/best.pt" \
  	--model "${root_dir}/run/train/delftbikes/yolov8x-delftbikes-1280/weights/best.pt" \
  	--model "${root_dir}/run/train/delftbikes/yolov8s-delftbikes-1280/weights/best.pt" \
  	--model "${root_dir}/run/train/delftbikes/yolov8n-delftbikes-1280/weights/best.pt" \
  	--data "data/delftbikes.yaml" \
  	--project "${root_dir}/run/predict/delftbikes/" \
  	--name "submission" \
  	--source "${root_dir}/data/vipriors/delftbikes/test/images" \
  	--imgsz 3440 \
  	--conf 0.00001 \
  	--iou 0.5 \
  	--max-det 2000 \
  	--augment \
  	--device 0 \
  	--exist-ok \
  	--save-txt \
  	--save-conf \
  	--overlap-mask \
  	--box
fi

cd "${root_dir}" || exit
