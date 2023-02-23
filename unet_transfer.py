from diffusers import DDPMPipeline
from diffusers import UNet2DModel
import torch
model_id = "mrm8488/ddpm-ema-anime-v2-128"

# load model and scheduler
pipeline = DDPMPipeline.from_pretrained(model_id)

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

unet = pipeline.unet
pretrained_dict = unet.state_dict()
model_dict = model.state_dict()
i = 0
prefixes = ["up_blocks.2","up_blocks.5","down_blocks.0","down_blocks.2","down_blocks.5",
          "mid_block","conv_norm_out","conv_out","conv_in","time_embedding"]
error=["up_blocks.2.resnets.2.norm1.weight","up_blocks.2.resnets.2.norm1.bias","up_blocks.2.resnets.2.conv1.weight",
       "up_blocks.2.resnets.2.conv_shortcut.weight","up_blocks.5.resnets.0.norm1.weight","up_blocks.5.resnets.0.norm1.bias",
       "up_blocks.5.resnets.0.conv1.weight","up_blocks.5.resnets.0.conv_shortcut.weight","conv_in.weight",
       "conv_in.bias","conv_out.weight","conv_out.bias"]
# prefixes = ["up_blocks.2","up_blocks.5","down_blocks.0","down_blocks.2","down_blocks.5"
#           ]
# error=["up_blocks.2.resnets.2.norm1.weight","up_blocks.2.resnets.2.norm1.bias","up_blocks.2.resnets.2.conv1.weight",
#        "up_blocks.2.resnets.2.conv_shortcut.weight","up_blocks.5.resnets.0.norm1.weight","up_blocks.5.resnets.0.norm1.bias",
#        "up_blocks.5.resnets.0.conv1.weight","up_blocks.5.resnets.0.conv_shortcut.weight"]
for name ,para in pretrained_dict.items():
    for j,prefix in enumerate(prefixes):
        if name.startswith(prefix):
            if name in error:
                break
            if j == 0 :
                new_name = name.replace(prefix,"up_blocks.1")
            elif j == 1:
                new_name = name.replace(prefix,"up_blocks.2")
            elif j == 2:
                break
            elif j == 3:
                new_name = name.replace(prefix, "down_blocks.1")
            elif j == 4:
                # new_name = name.replace(prefix, "down_blocks.2")
                break
            else :
                new_name= name
            # if new_name  not in model_dict:
            #     break
            model_dict[new_name] =para
model.load_state_dict(model_dict)
# for  i in model_dict.keys():
#     print(i)
torch.save(model.state_dict(), "modelv1.pth")
