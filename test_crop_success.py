from crop_face import  crop_face
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from crop_face import crop_face
from tqdm.auto import tqdm
## set
data_dir = "data_0000"


##
file_list = os.listdir(data_dir)
file_list = [os.path.join("data_0000", f) for f in file_list if f.endswith('.jpg')]
failures =[]
cnt = 0
for i in tqdm(file_list):
    image = Image.open(i)
    try :
        crop_face(image)
    except Exception as e:
        failures.append(i)
        cnt += 1

print("failures:", failures)
print("cnt:", cnt)
#
# crop_face(Image.open('./data_0000\\5867332.jpg')).save("1.jpg")

