import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from crop_face import crop_face


class CustomDataset(Dataset):
    def __init__(self, data_dir, instance_prompt,tokenizer,size=256,have_label=False):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.file_list = [f for f in self.file_list if f.endswith('.jpg')]
        self.file_list = sorted(self.file_list)
        self.crop_face = crop_face
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.have_label = have_label
        self.transforms = transforms.Compose(
            [
                transforms.Resize((size, size)),
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
        try :
            image = Image.open(image_path)
        except :
            image = Image.open("./imgs/hat2.jpg")
        # image = self.crop_face(image)
        image = image.convert("RGB")
        image = self.transforms(image)
        example["images"] = image
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        # Load label
        if self.have_label :
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
