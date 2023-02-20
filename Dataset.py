import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from crop_face import crop_face


class CustomDataset(Dataset):
    def __init__(self, data_dir, size=256):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.file_list = [f for f in self.file_list if f.endswith('.jpg')]
        self.file_list = sorted(self.file_list)
        self.crop_face = crop_face
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为 0.5
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),  # 0~1
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        example = {}
        
        # Load image
        image_path = os.path.join(self.data_dir, self.file_list[idx])
        image = Image.open(image_path)
        image = self.crop_face(image)
        image = self.transforms(image)
        example["images"] = image
        
        # Load label
        label_path = os.path.join(self.data_dir, self.file_list[idx][:-4] + '.txt')
        with open(label_path, 'r') as f:
            label = f.read()
        
        # Convert label to tensor
        example["tags"] = label
        
        return example


# Instantiate the dataset and dataloader
if __name__ == "__main__":
    dataset = CustomDataset('data_0000')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print((next(iter(dataloader))["images"][1]))
