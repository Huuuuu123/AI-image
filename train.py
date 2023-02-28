from clip_loss import clip_loss
from Dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler,StableDiffusionPipeline,DDIMScheduler
from diffusers import UNet2DModel
import torch.nn.functional as F

import torch
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import wandb
import torchvision
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from crop_face import crop_face
from torchvision import transforms
from datasets import load_dataset
from transformers import CLIPFeatureExtractor, CLIPTextModel,CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
def train (img_folder = None):
    wandb.init(project="AI-image", config=locals())
    #
    guidance_scale = 0.1
    n_cuts = 4
    device = "cuda"
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    pipe.enable_attention_slicing()
    batch_size = 1
    lr = 1e-6
    grad_accumulation_steps = 2
    save_model_every= 5 #epoch
    # img_folder = "./imgs"
    # model
    text_encoder = pipe.text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
    )
    vae = pipe.vae
    unet = pipe.unet
    # model = UNet2DModel(
    #     sample_size=32,  # the target image resolution
    #     in_channels=4,  # the number of input channels, 3 for RGB images
    #     out_channels=4,  # the number of output channels
    #     layers_per_block=2,  # how many ResNet layers to use per UNet block
    #     block_out_channels=(128, 256, 512),  # More channels -> more parameters
    #     down_block_types=(
    #
    #         "DownBlock2D",
    #         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    #         "AttnDownBlock2D",
    #     ),
    #     up_block_types=(
    #         "AttnUpBlock2D",
    #         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
    #
    #         "UpBlock2D",  # a regular ResNet upsampling block
    #     ),
    # )
    # model.load_state_dict(torch.load("./modelv1.pth"))
    # model.to("cuda")
    # Set the noise scheduler
    transform_1 = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为 0.5
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),  # 0~1
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    def transform_data(examples, transforms=transform_1):
        
        images = [transforms(crop_face(image)) for image in examples["image"]]
        input_ids = [tokenizer(
            "an Anime photo",
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids for example in examples]
        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        ).input_ids

        return {"images":images,"input_ids":input_ids}
    if img_folder is None:
        datasets =load_dataset("hipete12/anime_char_image",split="train")
        datasets.set_transform(transform_data)

    else :
        datasets = CustomDataset(img_folder)

    train_dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    
    noise_scheduler = pipe.scheduler
    sampling_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    sampling_scheduler.set_timesteps(num_inference_steps=50)
    # Training loop
    import bitsandbytes as bnb
    # optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    losses = []
    
    for epoch in range(100):
        unet.train()
        for step, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)/batch_size) :
            clean_images = batch["images"].to(device)
            with torch.no_grad():
                latents = 0.18215 * pipe.vae.encode(clean_images).latent_dist.mean
            # Sample noise to add to the images
            latents.to(device)
            noise = torch.randn(latents.shape).to(latents.device)
            bs = latents.shape[0]
    
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=latents.device
            ).long()
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
            # Get the model prediction
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)[0]
            # cond_loss=torch.tensor([0]*bs)
            # image = pipe.decode_latents(noisy_images)
    
    
        
            # # Get the predicted x0:
            # x0 = noise_scheduler.step(noise_pred, timesteps, noise).pred_original_sample
            # x0 = pipe.decode_latents(x0)
            #
            # # Calculate loss
            # cond_loss += clip_loss(x0, batch["tags"]) * guidance_scale / n_cuts
            
            # Calculate the loss
            # loss = F.mse_loss(noise_pred, noise) + cond_loss
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())
            wandb.log({"loss": loss})
            # Update the model parameters with the optimizer
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # if (step + 1) % log_samples_every == 0:
            #     x = torch.randn(8, 3, 32, 32).to(device)  # Batch of 8
            #     for i, t in tqdm(enumerate(sampling_scheduler.timesteps)):
            #         model_input = sampling_scheduler.scale_model_input(x, t)
            #         with torch.no_grad():
            #             noise_pred = model(model_input, t)["sample"]
            #         x = sampling_scheduler.step(noise_pred, t, x).prev_sample
            #     grid = torchvision.utils.make_grid(x, nrow=4)
            #     im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
            #     im = Image.fromarray(np.array(im * 255).astype(np.uint8))
            #     wandb.log({'Sample generations': wandb.Image(im)})
    
            # Occasionally save model
                # if (step + 1) % save_model_every == 0:
            #     torch.save(model.state_dict(), "model"+f'step_{step + 1}'+".pth")
                # image_pipe.save_pretrained(model_save_name + f'step_{step + 1}')
            # sample loop
        with torch.no_grad():
            x = torch.randn(1, 4, 32, 32).to(device)  # Batch of 8
            for i, t in tqdm(enumerate(sampling_scheduler.timesteps)):
                model_input = sampling_scheduler.scale_model_input(x, t)
              
                noise_pred = unet(x, t, encoder_hidden_states)[0]
                x = sampling_scheduler.step(noise_pred, t, x).prev_sample
            decoded_images = pipe.vae.decode(x / 0.18215).sample
            grid = torchvision.utils.make_grid(decoded_images, nrow=4)
            im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
            im = Image.fromarray(np.array(im * 255).astype(np.uint8))
            wandb.log({'Sample generations': wandb.Image(im)})
            # save model
            torch.save(unet.state_dict(), "new_model"+f'epoch_{epoch + 1}'+".pth")
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(losses)
    axs[1].plot(np.log(losses))
    plt.show()
    
    torch.save(unet.state_dict(), "modelv2.pth")
    pipeline = StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,

    )
    pipeline.save_pretrained("./pipeline")
    model_name = f"ai-image"
    # Code to upload a pipeline saved locally to the hub
    from huggingface_hub import HFUploadApi

    api = HFUploadApi(api_token="YOUR_API_TOKEN")
    api.upload_new_version(
        model_id="ai-image",
        file_path="./pipeline",
        release_notes="RELEASE_NOTES",
        model_version="0"
    )
if __name__ == "__main__":
    train()