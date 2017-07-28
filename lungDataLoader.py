import os
import numpy as np
import torch
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader
from skimage import io, color
from PIL import Image
import random
# import matplotlib.pyplot as plt

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

target_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

data_dir = '/Users/zhaohe/workspace/data/lung_training_data'


class LungDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.IMAGE_EXT = ['.png', 'jpeg', 'jpg', 'bmp']
        train_list = {}
        
        for x in ['train', 'val']:
            train_list[x] = []
            for dirpath, dnames, fnames in os.walk(os.path.join(self.root, x)):
                for f in fnames:
                    if f.endswith(tuple(self.IMAGE_EXT)):
                        train_list[x].append(os.path.join(dirpath, f))

        self.images = train_list['train']
        self.targets = train_list['val']
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.images[idx])
        target = Image.open(self.targets[idx])
        
        # just in case it is single channel
        image = image.convert('RGB')
        target = target.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)
            
        return [image, target]

    
    def view(self, idx):
        fig = plt.figure()
        img, target = self.__getitem__(idx)
        
        img = img.numpy()
        target = target.numpy()
        print(img.shape)
        
        ## For plt Dim must be (H,W,C)
        plt.imshow(np.transpose(img, (1,2,0)))
        plt.show()
         
        plt.imshow(np.transpose(target, (1,2,0)) * 255)
        plt.show()

lung424 = LungDataset(root=data_dir, data_transform=data_transform, target_transform=target_transform)

LungDataLoader = DataLoader(lung424, shuffle=True, batch_size=5)

if __name__ == '__main__':
    lung424 = LungDataset(root=data_dir, data_transform=data_transform, target_transform=target_transform)
    lung424.view(0)


