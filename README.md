# Stable Diffusionv 1.4 Dreambooth fine-tuning with ColossalAI

## Experiment Environment

- **Operating System:** Ubuntu 20.04
- **Python Version:** 3.8
- **CUDA Version:** 11.6
  - *Cuda version might be very important here, I cannot reproduce on CUDA 11.8 & 12.2*
- **Key Libraries:** 
  - **PyTorch:** 1.13.1
  - **Transformers:** 4.39.3
  - **Diffusers:** 0.8.0
    - *Diffusers version might be very important here, I cannot reproduce on the latest version*
  - **ColossalAI:** 0.3.6
  - See more in `./requirements.txt`
- **GPU Model:** NVIDIA RTX 3090 (24GB) * 1
- **CPU:** 24 vCPU AMD EPYC 7642 48-Core Processor

## Preparation

### 1. Download LDM model

```bash
sudo apt install git-lfs
git lfs install

mkdir model
cd model
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
```

### 2. Install Pytorch

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 3. Install Apex

```bash
git clone https://github.com/NVIDIA/apex
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```

### 4. Install requirements & Colossal-AI

```bash
pip install -r requirements.txt
pip install colossalai
```

## Data used in Experiment 

For the purpose of fine-tuning via Dreambooth, a very small dataset was used, comprising 4 nearly identical images of a silver gradient cat, placed in the `./instance/` directory. 

## Acknowledgments

This experiment was developed based on an official example provided by ColossalAI, with the primary effort being the adjustment of code and runtime parameters to ensure successful execution with limited resources. Due to the dependency on specific versions of CUDA and even C++ builders, I cannot guarantee the robustness of the code across all environments. Should you encounter any issues while using it, please feel free to contact me for assistance.

## Run

The script `./run.sh` is used to the training process for `stable-diffusion-v1-4` model:

```bash
export MODEL_NAME="./model/stable-diffusion-v1-4"
export INSTANCE_DIR="./instance"
export OUTPUT_DIR="./output"

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
```
