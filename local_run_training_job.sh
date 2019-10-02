now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="$1_$now"
JOB_DIR=$2
REGION=$3

gcloud ai-platform local train \
  --package-path gcloud_trainer/trainer/ \
  --module-name trainer.gcloud_aip_trainer \
  --job-dir $JOB_DIR \
  --python-version 3.5 \
  -- \
  --data-path gs://bird-classifier-cnn/training_data.npz \
  --categories-path gs://bird-classifier-cnn/cat.txt \
