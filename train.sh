FOLDER=$1

if [[ $# -eq 0 ]]; then
  echo "Correct usage: bash train.sh <dataset_folder_name>"
  exit
fi

python main.py --mode train --dataset MedicalData --image_size 192 \
  --image_dir data/MedicalData \
  --sample_dir MedicalData/samples \
  --log_dir MedicalData/logs \
  --model_save_dir MedicalData/models \
  --result_dir MedicalData/result \
  --num_iters 150000 \
  --batch_size 2 \
  --use_tensorboard False