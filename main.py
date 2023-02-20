import wandb
import numpy as np
import torch, torchvision
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from diffusers import DDPMPipeline
from diffusers import DDIMScheduler
from datasets import load_dataset
from matplotlib import pyplot as plt

import torch
import numpy as np
tensor = torch.randn(2, 2)
print(tensor.type())

# torch.long() 将tensor转换为long类型
long_tensor = tensor.long()
print(long_tensor.type())

# torch.half()将tensor转换为半精度浮点类型
half_tensor = tensor.half()
print(half_tensor.type())

test_data_x = torch.tensor(np.ones(shape = (540,19,1000)))

print(test_data_x.type)

import torch
print(torch.cuda.is_available())