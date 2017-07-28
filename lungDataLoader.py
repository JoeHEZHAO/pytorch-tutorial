import os
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from skimage import io
import matplotlib.pyplot as plt

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomSizedCrop(224),
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
        image = io.imread(self.images[idx])
        target = io.imread(self.targets[idx])
       
        w,h = image.shape
        img = np.empty((3, w, h), dtype=np.float32)
        targ = np.empty((3, w, h), dtype=np.float32)

        for i in range(3):
            img[i,:,:] = image
            targ[i,:,:] = target 
        
        if self.transform:
            img = self.transform(img)
            targ = self.transform(targ)
            
        return [img, targ]

    
    def view(self, idx):
        fig = plt.figure()
        img, target = self.__getitem__(idx)
#         print(np.max(img[0,:,:]))
        
        ## For plt Dim must be (H,W,C)
        plt.imshow(np.transpose(img, (1,2,0)))
        plt.show()
        plt.imshow(np.transpose(target, (1,2,0)))
        plt.show()


if __name__ == '__main__':
	pass