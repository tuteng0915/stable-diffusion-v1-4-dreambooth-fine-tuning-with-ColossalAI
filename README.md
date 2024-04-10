# Stable Diffusionv 1.4 Dreambooth fine-tuning with ColossalAI

https://github.com/tuteng0915/stable-diffusion-dreambooth-fine-tuning-with-ColossalAI

This project demonstrates the fine-tuning of the Stable Diffusion v1.4 model for personalized image generation using Dreambooth

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

## Data used in Experiment 

For the purpose of fine-tuning via Dreambooth, a very small dataset was used, comprising 4 nearly identical images of a silver gradient cat, placed in the `./instance/` directory. 

## Acknowledgments

This experiment was developed based on an official example provided by ColossalAI, with the primary effort being the adjustment of code and runtime parameters to ensure successful execution with limited resources. Due to the dependency on specific versions of CUDA and even C++ builders, I cannot guarantee the robustness of the code across all environments. Should you encounter any issues while using it, please feel free to contact me.

Initially, my endeavor to delve into fine-tuning diffusion models led me to explore the examples provided for training LDM (Latent Diffusion Models) within ColossalAI. However, this journey was fraught with obstacles. A notable challenge arose from the integration efforts between Lightning and ColossalAI, where it became apparent that Lightning was attempting to work with parts of ColossalAI that had been deprecated. The option to downgrade ColossalAI to resolve these integration issues presented a new set of problems. The intricate dependency between ColossalAI and Lightning introduces additional complexity, making the debugging process particularly challenging. It is my hope that the maintainers of ColossalAI can address these issues by updating the relevant examples.


## Setting & Implement


### 1. **Single GPU Consideration:**

To accommodate the limited computational resources of a single GPU setup, the `--nproc_per_node` is set to 1, and `train_batch_size` is also set to 1. This configuration ensures that the training process is optimized for environments with restricted hardware capabilities.

### 2. **Mixed Precision Training:**

The experiment leverages FP16 (mixed precision) to manage memory usage more effectively. Corrections were made in the code to ensure uniform data types across all models:

In the code of ColossalAI, only vae and text_encoder were adjusted, but the data accuracy of unet was not adjusted, resulting in mismatch in subsequent data accuracy.

- Model components, including the VAE, text encoder, and UNet, are transferred to the appropriate device with the correct data type:

```python
  vae.to(get_accelerator().get_current_device(), dtype=weight_dtype)
  text_encoder.to(get_accelerator().get_current_device(), dtype=weight_dtype)
  unet.to(get_accelerator().get_current_device(), dtype=weight_dtype)
```

- To further avoid the problem of inconsistent data precision, it is needed to ensure that the precision of lantent and text_encode is converted correctly before passing to UNet:

```python
  noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
  noisy_latents = noisy_latents.to(get_accelerator().get_current_device(), dtype=weight_dtype)

  encoder_hidden_states = text_encoder(batch["input_ids"])[0]
  encoder_hidden_states = encoder_hidden_states.to(get_accelerator().get_current_device(), dtype=weight_dtype)
```

### 3. **Code Organization:**

To streamline the main Python script and enhance clarity, the dataset and argument parser were moved into a separate `utils` module. To ensure a focused and tested approach, several untested options and features were removed, including:

- Removal of Push to Hugging Face Functionality.

- Simplification of Text Encoder Options: I restricted the text encoder options exclusively to CLIPTextModel, in alignment with the architecture of the Stable Diffusion v1.4 model.

- Discontinuation of Booster Plugin: torch_ddp remains as the only chosen strategy.

### 4. **Model Saving:**

Originally, only the UNet's parameters were saved during the fine-tuning process, which aligns with our objective as UNet is the focus of fine-tuning. However, the saving functionality provided by ColossalAI posed challenges for the subsequent evaluation process. To address this, I opted for a straightforward solution by directly copying the other unchanged files of the Stable Diffusion v1.4 model to the output_dir. This approach, albeit not space-efficient, simplifies the implementation and facilitates easier access to the model for evaluation purposes.

```python
    dir_list = ["feature_extractor", "safety_checker", "scheduler", "text_encoder", "tokenizer", "vae"]
    if local_rank == 0:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        
        for d in dir_list:
            shutil.copytree(os.path.join(args.pretrained_model_name_or_path, d), os.path.join(args.output_dir, d))

        shutil.copy(os.path.join(args.pretrained_model_name_or_path, "model_index.json"), args.output_dir)
        
        os.makedirs(os.path.join(args.output_dir, "unet"), exist_ok=True)
        shutil.copy(os.path.join(args.pretrained_model_name_or_path, "unet/config.json"), os.path.join(args.output_dir, "unet"))

        booster.save_model(unet, os.path.join(os.path.join(args.output_dir, "unet"), "diffusion_pytorch_model.bin"))
        logger.info(f"Saving model checkpoint to {args.output_dir} on rank {local_rank}")
```

### 5. **Evaluation:**

Dreambooth lacks an automated method for evaluation. Therefore, after the fine-tuning process, the model was manually evaluated by generating images using the same prompt. The outcomes of this evaluation are discussed in the subsequent section.

```python
    pipeline = DiffusionPipeline.from_pretrained("./output", safety_checker=None).to("cuda")
    for i in range(1, 5):
        image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save("./result/output-"+str(i)+".png")

    del pipeline

    pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, safety_checker=None).to("cuda")
    for i in range(1, 5):
        image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save("./result/baseline-"+str(i)+".png")
```


## Result





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
