now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="$1_$now"
JOB_DIR="$2/$JOB_NAME"
REGION=$3

gcloud ai-platform jobs submit training $JOB_NAME \
  --staging-bucket gs://bird-classifier-cnn \
  --package-path gcloud_trainer/trainer/ \
  --module-name trainer.gcloud_aip_trainer \
  --region $REGION \
  --python-version 3.5 \
  --runtime-version 1.14 \
  --job-dir $JOB_DIR \
  --scale-tier basic-gpu \
  --stream-logs \
  -- \
  --config-file gs://bird-classifier-cnn/config/cloud_config.yml
