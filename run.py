import torch
from diffusers import DiffusionPipeline


pipeline = DiffusionPipeline.from_pretrained("./output", safety_checker=None).to("cuda")

prompt = "a photo of my cat."

for i in range(1,5):
    image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save("./result/output-"+str(i)+".png")

del pipeline
    
pipeline = DiffusionPipeline.from_pretrained("./model/stable-diffusion-v1-4",safety_checker=None).to("cuda")

prompt = "a photo of my cat."

for i in range(1,121):
    image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save("./result/baseline-"+str(i)+".png")
