import torch
import os
import json

model_id = "./output"

from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, LMSDiscreteScheduler

tokenizer = AutoTokenizer.from_pretrained(model_id + "/tokenizer")

text_encoder = CLIPTextModel.from_pretrained(model_id + "/text_encoder")

vae = AutoencoderKL.from_pretrained(model_id + "/vae")

unet = UNet2DConditionModel.from_pretrained(model_id + "/unet")

scheduler_config_path = os.path.join(model_id, "scheduler", "scheduler_config.json")
with open(scheduler_config_path, 'r') as f:
    config = json.load(f)
scheduler = DDPMScheduler(**config)


def run_pipeline(prompt, num_inference_steps=50):
    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_inputs.input_ids).last_hidden_state
    
    latents = torch.randn((1, unet.in_channels, unet.sample_size, unet.sample_size), device="cuda")

    timesteps = torch.linspace(num_inference_steps-1, 0, num_inference_steps, dtype=torch.long, device="cuda")

    for i, timestep in enumerate(timesteps):
        latent_model_input = latents
        sigma = scheduler.sigmas[timestep]

        with torch.no_grad():
            noise_pred = unet(latent_model_input, text_embeddings, timestep=timestep.unsqueeze(0).repeat(latent_model_input.shape[0]))
        
        latents = scheduler.step(noise_pred, timesteps[i], latents, sigma).prev_sample

    images = vae.decode(latents)

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()

    return images

prompt = "a photo of my cat."
images = run_pipeline(prompt)

