from clip_loss import clip_loss
from Dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler,StableDiffusionPipeline
from diffusers import UNet2DModel
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
#
guidance_scale = 0.1
n_cuts = 4
device = "cuda"
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
batch_size = 4
# model
model = UNet2DModel(
    sample_size=32,  # the target image resolution
    in_channels=4,  # the number of input channels, 3 for RGB images
    out_channels=4,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 256, 512),  # More channels -> more parameters
    down_block_types=(
        
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
model.load_state_dict(torch.load("./modelv1.pth"))
model.to("cuda")
# Set the noise scheduler
dataset = CustomDataset("data_0000")
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

losses = []

for epoch in range(30):
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
        noisy_images = noise_scheduler.add_noise(latents, noise, timesteps).to(device)

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
        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 5 == 0:
        loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
        print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
        
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(losses)
axs[1].plot(np.log(losses))
plt.show()

torch.save(model.state_dict(), "modelv2.pth")