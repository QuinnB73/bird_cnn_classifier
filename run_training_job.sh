now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="$1_$now"
JOB_DIR="$2/$JOB_NAME"
REGION=$3
CONFIG_NUM=$4

gcloud ai-platform jobs submit training $JOB_NAME \
  --staging-bucket gs://bird-classifier-cnn \
  --package-path gcloud_trainer/trainer/ \
  --module-name trainer.gcloud_aip_trainer \
  --region $REGION \
  --python-version 3.5 \
  --runtime-version 1.14 \
  --job-dir $JOB_DIR \
  --scale-tier custom \
  --master-machine-type complex_model_m_p100 \
  --stream-logs \
  -- \
  --config-file gs://bird-classifier-cnn/config/cloud_config_$CONFIG_NUM.yml
