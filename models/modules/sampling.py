from torch.utils import data
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision import transforms
import numpy as np
 
class FaceDataset(Dataset):
    def __init__(self,data_path, is_train=True):
        self.dataset = []
        # f1 = os.listdir(os.path.join(data_path, "negative"))
        # f2 = os.listdir(os.path.join(data_path, "positive"))
        # f3 = os.listdir(os.path.join(data_path, "part"))
        l1 = open(os.path.join(data_path, "negative.txt")).readlines()
        for l1_filename in l1:
            self.dataset.append([os.path.join(data_path, l1_filename.split(" ")[0]), l1_filename.split(" ")[1:6]])
            # print(self.dataset)
        # exit()
        l2 = open(os.path.join(data_path, "positive.txt")).readlines()
        for l2_filename in l2:
            self.dataset.append([os.path.join(data_path, l2_filename.split(" ")[0]), l2_filename.split(" ")[1:6]])
        l3 = open(os.path.join(data_path, "part.txt")).readlines()
        for l3_filename in l3:
            self.dataset.append([os.path.join(data_path, l3_filename.split(" ")[0]), l3_filename.split(" ")[1:6]])
        # print(self.dataset.shape())
 
    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, item):
        data = self.dataset[item]
        # print(data[0])
        img_tensor = self.trans(Image.open(data[0]))
        category = torch.tensor(float(data[1][0])).reshape(-1)
        # print(category.shape,"9999999999999999999999")
        offset = torch.tensor([float(data[1][1]), float(data[1][2]), float(data[1][3]), float(data[1][4])])
 
        return img_tensor, category, offset
 
    def data_transforms(self,x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(x)
 
#测试是否可用
if __name__ == '__main__':
    data_path = r"E:\CelebA\MTCN\12"
    mydata = FaceDataset(data_path)
    data = data.DataLoader(mydata, 2, shuffle=True)
    for i, (x1, y1, y2) in enumerate(data):
        print(x1)
        print(x1.shape)
        print(y1)
        print(y2.shape)
        print()
        exit()