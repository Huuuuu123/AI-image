from clip_loss import clip_loss
from Dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler, StableDiffusionPipeline, DDIMScheduler,PNDMScheduler
from diffusers import UNet2DModel,DiffusionPipeline
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


def train(img_folder=None):
    wandb.init(project="AI-image", config=locals())
    #
    # guidance_scale = 0.1
    n_cuts = 4
    device = "cuda"
    # model_id = "mrm8488/ddpm-ema-anime-256"
    # pipe = DiffusionPipeline.from_pretrained(model_id).to(device)
    batch_size = 4
    lr = 1e-4
    grad_accumulation_steps = 2
    save_model_every = 5  # epoch
    # img_folder = "./imgs"
    # model
    model = UNet2DModel(
        sample_size=64,  # the target image resolution
        in_channels=3 , # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64,64,128, 256, 512),  # More channels -> more parameters
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",# a regular ResNet upsampling block
        ),
    )
    # model.load_state_dict(torch.load("./modelepoch_38.pth"))
    model.to("cuda")
    # pipe.to("cuda")
    # Set the noise scheduler
    transform_1 = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为 0.5
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),  # 0~1
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def transform_data(examples, transforms=transform_1):
        
        images = [transforms(crop_face(image)) for image in examples["image"]]
        return {"images": images}
    
    if img_folder is None:
        datasets = load_dataset("hipete12/anime_char_image", split="train")
        datasets.set_transform(transform_data)
    
    else:
        datasets = CustomDataset(img_folder,size=64)
    
    train_dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    
    noise_scheduler = PNDMScheduler.from_config("./scheduler_config.json")
    sampling_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    sampling_scheduler.set_timesteps(num_inference_steps=50)
    # Training loop
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    losses = []
    
    for epoch in range(100):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader) / batch_size):
            clean_images = batch["images"].to(device)
            # with torch.no_grad():
            #     latents = 0.18215 * pipe.vae.encode(clean_images).latent_dist.mean
            # # Sample noise to add to the images
            # latents.to(device)
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps).to(device)
            
            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
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
        x = torch.randn(2, 3, 64, 64).to(device)  # Batch of 8
        for i, t in tqdm(enumerate(sampling_scheduler.timesteps)):
            model_input = sampling_scheduler.scale_model_input(x, t)
            with torch.no_grad():
                noise_pred = model(model_input, t)["sample"]
            x = sampling_scheduler.step(noise_pred, t, x).prev_sample
            # decoded_images = pipe.vae.decode(x / 0.18215).sample
        grid = torchvision.utils.make_grid(x, nrow=2)
        im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
        im = Image.fromarray(np.array(im * 255).astype(np.uint8))
        wandb.log({'Sample generations': wandb.Image(im)})

        # save model
        if (epoch + 1) % 5 ==0:
            torch.save(model.state_dict(), "model" + f'epoch_{epoch + 1}' + ".pth")
        loss_last_epoch = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        wandb.log({"loss": np.log(loss_last_epoch)})
        print(f"Epoch:{epoch + 1}, loss: {np.log(loss_last_epoch)}")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(losses)
    axs[1].plot(np.log(losses))
    plt.show()
    
    torch.save(model.state_dict(), "modelv2.pth")


if __name__ == "__main__":
    train(img_folder = "./imgs")