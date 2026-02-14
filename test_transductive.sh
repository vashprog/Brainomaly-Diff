#!/bin/bash

folder=$1
iter=$2

if [[ $# -lt 2 ]]; then
  echo "✅ Correct usage:"
  echo "bash test_transductive.sh <dataset_folder> <model_iteration>"
  echo "Example: bash test_transductive.sh MedicalData 100000"
  exit
fi

echo "==============================================="
echo " Running Brainomaly-Diffusion Transductive Test"
echo " Dataset : $folder"
echo " Iter    : $iter"
echo "==============================================="

python main.py --mode testAUCTransductive \
  --dataset MedicalData \
  --image_size 192 \
  --image_dir data/${folder} \
  --sample_dir ${folder}/samples \
  --log_dir ${folder}/logs \
  --model_save_dir ${folder}/models \
  --result_dir ${folder}/result \
  --test_iters ${iter} \
  --batch_size 1 \
  --num_workers 2 \
  --use_tensorboard False
