import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import gradio as gr
generator = torch.Generator("cuda").manual_seed(0)
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to('cuda')

prompt = "mksks style,masterpiece, best quality, ultra-detailed, illustration, 1girl, witch hat, purple eyes, blonde hair, wielding a purple staff blasting purple energy, purple beam, purple effects, dragons, chaos"
negative_prompt=" lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck"
with autocast("cuda"):

    image = pipe(prompt, guidance_scale=6,
                 negative_prompt=negative_prompt,
                 num_images_per_prompt= 2,
                 generator = generator


                 ).images

for i in range(2):
    image[i].save(f"test{i}.png")

# todo 引入 CLIP指导模型生成
# todo 完善 UI界面
