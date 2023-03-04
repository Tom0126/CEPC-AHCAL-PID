from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from  typing import Any
from Config.config import *
import numpy as np
import torch

class ImageSet(Dataset):
    def __init__(self,img_path,label_path,mean=0.07068362266069922,std=1.6261502913850978
                 ,mean_std_static=False,transform=None) -> None:
        super().__init__()
        datasets=np.load(img_path)
        labels=np.load(label_path)
        self.datasets = datasets.astype(np.float32)
        # use the mean&std of the train set
        # if mean_std_static:
        #     mean=mean
        #     std=std
        # else:
        #     mean = np.average(datasets)
        #     std = np.std(datasets)
        # self.datasets = (datasets-mean)/std
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index: Any):
        img = self.datasets[index]
        label = self.labels[index]
        if self.transform != None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.datasets)

def data_loader(img_path, label_path,mean=None,std=None,mean_std_static:bool=False,
                batch_size:int=32, shuffle:bool=False, num_workers:int=4):

    transforms_train = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Grayscale(),
        # transforms.Resize((int(img_size*1.2),int(img_size*1.2))),
        # transforms.RandomCrop((img_size, img_size)),
        # transforms.RandomVerticalFlip(),
        # transforms.Normalize(mean=(-22.8044,), std=(10.6840,)),
        # transforms.Normalize(mean=(-9.783645194698,), std=(17.847375921229,)),
    ])
    dataset_train = ImageSet(img_path,label_path,mean=mean,std=std,mean_std_static=mean_std_static,transform=transforms_train)
    loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers, drop_last=True)

    return loader_train


class PIDSet(Dataset):
    def __init__(self, imgs, mean=0.07068362266069922, std=1.6261502913850978
                 , mean_std_static=True, transform=None) -> None:
        super().__init__()
        # datasets = np.load(img_path)
        # datasets = datasets.astype(np.float32)

        self.datasets=imgs.astype(np.float32)

        # use the mean&std of the train set
        if mean_std_static:
            mean = mean
            std = std
        else:
            mean = np.average(self.datasets)
            std = np.std(self.datasets)

        self.datasets = (self.datasets - mean) / std
        self.transform = transform

    def __getitem__(self, index: Any):
        img = self.datasets[index]

        if self.transform != None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.datasets)


def pid_data_loader(imgs, mean=0.07068362266069922, std=1.6261502913850978, mean_std_static: bool = True,
                batch_size: int = 128, shuffle: bool = False, num_workers: int = 0):
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_pid = PIDSet(imgs,mean=mean, std=std, mean_std_static=mean_std_static,
                             transform=transforms_train)

    loader_pid = DataLoader(dataset=dataset_pid, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers, drop_last=False)

    return loader_pid


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img_path = 'File/e_test.npy'
    label_path='File/e_label_test.npy'
    loader = data_loader(img_path,label_path, num_workers=0,)
    for i, (img,label) in enumerate(loader):
        print('img:{} label:{}'.format(img.shape,label.shape))
