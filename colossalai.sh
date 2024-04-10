export MODEL_NAME="./model/stable-diffusion-v1-4"
export INSTANCE_DIR="./instance"
export OUTPUT_DIR="./output"
# export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node 1 train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of my cat." \
  --resolution=512 \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=20
