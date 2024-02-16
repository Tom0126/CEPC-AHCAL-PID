from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from  typing import Any
from Config.config import *
import numpy as np
import torch

class ImageSet(Dataset):
    def __init__(self,
                 img_path,
                 label_path,
                 mean,std,
                 mean_std_static,
                 transform=None) -> None:
        super().__init__()

        datasets=np.load(img_path,allow_pickle=True)
        labels=np.load(label_path,allow_pickle=True)

        #standardize the train set
        if mean_std_static:
            mean=mean
            std=std
        else:
            mean = np.average(datasets)
            std = np.std(datasets)
        datasets = (datasets-mean)/std

        self.datasets = datasets.astype(np.float32)
        self.labels = labels.astype(np.longlong)
        self.transform = transform
    
    def __getitem__(self, index: Any):
        img = self.datasets[index]
        label = self.labels[index]
        if self.transform != None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.datasets)


class GNNImageSet(Dataset):
    def __init__(self,
                 img_path,
                 label_path,
                 mean,
                 std,
                 mean_std_static,
                 max_nodes,
                 transform=None) -> None:
        super().__init__()

        datasets = np.load(img_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True)

        # standardize the train set
        if mean_std_static:
            mean = mean
            std = std
        else:
            mean = np.average(datasets)
            std = np.std(datasets)
        datasets = (datasets - mean) / std

        self.datasets = datasets.astype(np.float32)
        self.labels = labels.astype(np.longlong)
        self.transform = transform

        self.max_nodes=max_nodes

    def __getitem__(self, index: Any):

        img = self.datasets[index]

        img_indices = np.nonzero(img)

        e = img[img_indices[0], img_indices[1], img_indices[2]]

        img = [i.astype(np.float32) for i in img_indices]
        img.append(e)
        img = np.vstack(img)



        if self.max_nodes > img.shape[-1]:

            paddings=np.zeros((img.shape[-2], (self.max_nodes-img.shape[-1])), dtype=np.float32)

            img=np.concatenate([img, paddings], axis=1)

        else:

            choice = np.random.choice(img.shape[-1], self.max_nodes, replace=False)
            img=img[:, choice]



        label = self.labels[index]


        if self.transform != None:
            img = self.transform(img)


        return img, label

    def __len__(self):
        return len(self.datasets)


def data_loader(img_path,
                label_path,
                mean=0.0,
                std=1.0,
                mean_std_static:bool=True,
                batch_size:int=256,
                shuffle:bool=False,
                num_workers:int=0,
                **kwargs):

    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = ImageSet(img_path,
                             label_path,
                             mean=mean,
                             std=std,
                             mean_std_static=mean_std_static,
                             transform=transforms_train)

    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=True)




    return loader_train


def data_loader_gnn(img_path,
                    label_path,
                    mean=0.0,
                    std=1.0,
                    mean_std_static: bool = True,
                    batch_size: int = 512,
                    shuffle: bool = False,
                    num_workers: int = 0,
                    max_nodes:int=32,
                    **kwargs):

    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = GNNImageSet(img_path,
                                label_path,
                                mean=mean,
                                std=std,
                                mean_std_static=mean_std_static,
                                transform=transforms_train,
                                max_nodes=max_nodes
                                )

    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=True)

    return loader_train


if __name__ == "__main__":

    img_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/tutorial/Test/imgs.npy'
    label_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/tutorial/Test/labels.npy'

    loader = data_loader_gnn(img_path,label_path, num_workers=0,mean_std_static=True, max_nodes=8)
    for i, (img,label) in enumerate(loader):
        print('img:{} label:{}'.format(img.shape,label.shape))
        if i==0:
            break
