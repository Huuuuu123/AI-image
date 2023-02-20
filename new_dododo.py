from diffusers import AutoencoderKL
import torch
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
# model_id = "runwayml/stable-diffusion-v1-5"
model_id = "hakurei/waifu-diffusion"
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16)
prompt ="1girl,bangs, barbara (genshin impact), bare shoulders, belt, blue eyes, book, choker, cross, detached sleeves, dress, genshin impact, hair between eyes, hat, light brown hair, long hair, long sleeves, looking at viewer, nun, one eye closed, sidelocks, simple background, smile, solo, standing, standing on one leg, strapless, strapless dress, sushi 171, twintails, vision (genshin impact), waving, white background, white dress"
pipe = pipe.to("cuda")
batch_size= 2
generator =[ torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


image = pipe([prompt]*batch_size, generator=generator, num_inference_steps=20).images

from PIL import Image


def image_grid(imgs, rows=1, cols=2):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

image_grid(image).save("girl.png")