import torch
from Agg.Datasets import  get_server_public_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
class sample_dataset(Dataset):
    def __init__(self,images,targets):
        self.images = images
        self.targets = targets
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        img = self.images[item]
        label = self.targets[item]
        return img,label

class FedMD():
    def __init__(self,dataname='CIFAR100'):
        self.dataname = dataname
        self.public_dataset = get_server_public_dataset(dataname)

    def update(self,public_activations):
        ratio = 1.0 / len(public_activations)
        global_act = torch.zeros_like(public_activations[0])
        for pub_act in public_activations:
            global_act += pub_act * ratio
        return global_act
    def get_public_dataset(self):
        if self.dataname == 'MiniGC':
            pub_loader = self.public_dataset
            return pub_loader
        else:
            pub_loader = DataLoader(self.public_dataset,batch_size=250,shuffle=False)
            for x,y in pub_loader:
                return sample_dataset(x,y)