
import hashlib
import math
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam

from utils.argparser import parse_args
from utils.datasets import DreamBoothDataset, PromptDataset

disable_existing_loggers()
logger = get_dist_logger()

def main(args):
    if args.seed is None:
        colossalai.launch_from_torch(config={})
    else:
        colossalai.launch_from_torch(config={}, seed=args.seed)

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if get_accelerator().get_current_device() == "cuda" else torch.float32
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            pipeline.to(get_accelerator().get_current_device())

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not local_rank == 0,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha256(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline

    if local_rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        logger.info(f"Loading tokenizer from {args.tokenizer_name}", ranks=[0])
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        logger.info("Loading tokenizer from pretrained model", ranks=[0])
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
        # import correct text encoder class

    # Load models and create wrapper for stable diffusion

    logger.info(f"Loading text_encoder from {args.pretrained_model_name_or_path}", ranks=[0])

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    logger.info(f"Loading AutoencoderKL from {args.pretrained_model_name_or_path}", ranks=[0])
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )

    if args.externel_unet_path is None:
        logger.info(f"Loading UNet2DConditionModel from {args.pretrained_model_name_or_path}", ranks=[0])
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, low_cpu_mem_usage=False
        )
    else:
        logger.info(f"Loading UNet2DConditionModel from {args.externel_unet_path}", ranks=[0])
        unet = UNet2DConditionModel.from_pretrained(
            args.externel_unet_path, revision=args.revision, low_cpu_mem_usage=False
        )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.train_batch_size * world_size

    # Use Booster API to use Gemini/Zero with ColossalAI

    booster_kwargs = {}
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()

    booster = Booster(plugin=plugin, **booster_kwargs)

    # config optimizer for colossalai zero
    optimizer = HybridAdam(
        unet.parameters(), lr=args.learning_rate, initial_scale=2**5, clipping_norm=args.max_grad_norm
    )

    # load noise_scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # prepare dataset
    logger.info(f"Prepare dataset from {args.instance_data_dir}", ranks=[0])
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        test=args.test_run,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(get_accelerator().get_current_device(), dtype=weight_dtype)
    text_encoder.to(get_accelerator().get_current_device(), dtype=weight_dtype)
    unet.to(get_accelerator().get_current_device(), dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    unet, optimizer, _, _, lr_scheduler = booster.boost(unet, optimizer, lr_scheduler=lr_scheduler)

    # Train!
    total_batch_size = args.train_batch_size * world_size

    logger.info("***** Running training *****", ranks=[0])
    logger.info(f"  Num examples = {len(train_dataset)}", ranks=[0])
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}", ranks=[0])
    logger.info(f"  Num Epochs = {args.num_train_epochs}", ranks=[0])
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}", ranks=[0])
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}", ranks=[0])
    logger.info(f"  Total optimization steps = {args.max_train_steps}", ranks=[0])

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not local_rank == 0)
    progress_bar.set_description("Steps")
    global_step = 0

    torch.cuda.synchronize()
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            torch.cuda.reset_peak_memory_stats()
            # Move batch to gpu
            for key, value in batch.items():
                batch[key] = value.to(get_accelerator().get_current_device(), non_blocking=True)

            # Convert images to latent space
            optimizer.zero_grad()

            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = noisy_latents.to(get_accelerator().get_current_device(), dtype=weight_dtype)
            
            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            encoder_hidden_states = encoder_hidden_states.to(get_accelerator().get_current_device(), dtype=weight_dtype)

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            logger.info(f"max GPU_mem cost is {torch.cuda.max_memory_allocated()/2**20} MB", ranks=[0])
            # Checks if the accelerator has performed an optimization step behind the scenes
            progress_bar.update(1)
            global_step += 1
            logs = {
                "loss": loss.detach().item(),
                "lr": optimizer.param_groups[0]["lr"],
            }  # lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step % args.save_steps == 0:
                torch.cuda.synchronize()
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                booster.save_model(unet, os.path.join(save_path, "diffusion_pytorch_model.bin"))
                if local_rank == 0:
                    if not os.path.exists(os.path.join(save_path, "config.json")):
                        shutil.copy(os.path.join(args.pretrained_model_name_or_path, "unet/config.json"), save_path)
                    logger.info(f"Saving model checkpoint to {save_path}", ranks=[0])
            if global_step >= args.max_train_steps:
                break
    torch.cuda.synchronize()
    
    
    dir_list = ["feature_extractor", "safety_checker", "scheduler", "text_encoder", "tokenizer", "vae"]
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        
        for d in dir_list:
            shutil.copytree(os.path.join(args.pretrained_model_name_or_path, d), os.path.join(args.output_dir, d))

        shutil.copy(os.path.join(args.pretrained_model_name_or_path, "model_index.json"), args.output_dir)
        
        os.makedirs(os.path.join(args.output_dir, "unet"), exist_ok=True)
        shutil.copy(os.path.join(args.pretrained_model_name_or_path, "unet/config.json"), os.path.join(args.output_dir, "unet"))

        booster.save_model(unet, os.path.join(os.path.join(args.output_dir, "unet"), "diffusion_pytorch_model.bin"))
        logger.info(f"Saving model checkpoint to {args.output_dir} on rank {local_rank}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
        
    # Inference
    os.makedirs("result", exist_ok=True)
    import torch
    from diffusers import DiffusionPipeline

    prompt = args.instance_prompt
    
    pipeline = DiffusionPipeline.from_pretrained("./output", safety_checker=None).to("cuda")
    for i in range(1, 5):
        image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save("./result/output-"+str(i)+".png")

    del pipeline

    pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, safety_checker=None).to("cuda")
    for i in range(1, 5):
        image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save("./result/baseline-"+str(i)+".png")

    import matplotlib.pyplot as plt
    from PIL import Image
    import os

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    instances_images = [f'./instance/instance-{i}.png' for i in range(1, 5)]
    baseline_images = [f'./result/baseline-{i}.png' for i in range(1, 5)]
    results_images = [f'./result/output-{i}.png' for i in range(1, 5)]

    def plot_image_row(image_filenames, row_index, title):
        for col_index, img_path in enumerate(image_filenames):
            img = Image.open(img_path)
            img = img.resize((128, 128))
            axes[row_index, col_index].imshow(img)
            axes[row_index, col_index].set_title(title if col_index==0 else "", fontsize=18)
            axes[row_index, col_index].axis('off')

    plot_image_row(instances_images, 0, 'Instance')
    plot_image_row(baseline_images, 1, 'Baseline')
    plot_image_row(results_images, 2, 'Result')

    plt.tight_layout(pad=3.0)
    plt.savefig("result.png")
