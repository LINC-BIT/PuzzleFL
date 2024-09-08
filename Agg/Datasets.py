import torch
import torchvision
# from dgl.data import MiniGCDataset
from torch.utils.data import Dataset
from torchvision import transforms

from ClientTrain.dataset.Cifar100 import Cifar100Task
from ClientTrain.dataset.fc100 import FC100Task
from ClientTrain.dataset.miniimagenet import MiniImageTask
# from ClientTrain.dataset.online_shoping import OnlineTask
# import dgl
def collate(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs)

def collate_label(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

def get_server_dataset(samples,name=None,size=None,t=0):
    transform = None
    if name == 'MNIST':
        dataloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=True, download=False,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               # only 1 channel
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=samples, shuffle=True
        )
        for data,_ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name=='CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_stddev = (0.2023, 0.1994, 0.2010)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ])
        training_set = torchvision.datasets.CIFAR10(root='./files', train=True, download=True,
                                                    transform=transform)
        dataloader = torch.utils.data.DataLoader(
            training_set,
            batch_size=samples,
            shuffle=True,
        )
        for data,_ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name=='CIFAR100':
        task = Cifar100Task('/data/lpyx/FedAgg/data/cifar100', task_num=1)
        train, test = task.getTaskDataSet()
        train_dataset = train[0]
        dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=samples,
                                                   shuffle=True)
        # mean = [0.5071, 0.4867, 0.4408]
        # std = [0.2675, 0.2565, 0.2761]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
        for data,_ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name=='FC100':
        task = FC100Task('/data/lpyx/FPKD_other/data/FC100', task_num=1)
        train, test = task.getTaskDataSet()
        train_dataset = train[0]
        dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=samples,
                                                 shuffle=True)
        # mean = [0.5071, 0.4867, 0.4408]
        # std = [0.2675, 0.2565, 0.2761]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
        for data, _ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name=='miniimagenet':
        task = MiniImageTask(root='/data/lpyx/FedFPKD/data/mini-imagenet/',json_path="/data/lpyx/FedFPKD/data/mini-imagenet/classes_name.json",task_num=1)
        train, test = task.getTaskDataSet()
        train_dataset = train[0]
        dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=samples,
                                                 shuffle=True)
        # mean = [0.5071, 0.4867, 0.4408]
        # std = [0.2675, 0.2565, 0.2761]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
        for data, _ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name == 'online_shopping':
        online = OnlineTask()
        train, test = online.getTaskDataSet()
        train_dataset = train[0]
        dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=samples,
                                                 shuffle=True)
        for data, _, _ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name == 'MiniGC':

        train_dataset = MiniGCDataset(samples, 10, 20)
        dataset = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=20,
                                                 shuffle=True,collate_fn=collate)



    else:
        dataset = ServerDataset([torch.rand(size) for i in range(samples)])

    return dataset

def get_server_task_dataset(samples,name=None,t=0):
    transform = None
    if name == 'MNIST':
        dataloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=True, download=False,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               # only 1 channel
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=samples, shuffle=True
        )
        for data, _ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_stddev = (0.2023, 0.1994, 0.2010)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ])
        training_set = torchvision.datasets.CIFAR10(root='./files', train=True, download=True,
                                                    transform=transform)
        dataloader = torch.utils.data.DataLoader(
            training_set,
            batch_size=samples,
            shuffle=True,
        )
        for data, _ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name == 'CIFAR100':
        task = Cifar100Task('/data/lpyx/FedAgg/data/cifar100', task_num=10)
        train, test = task.getTaskDataSet()
        train_dataset = train[t]
        dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=samples,
                                                 shuffle=True)
        # mean = [0.5071, 0.4867, 0.4408]
        # std = [0.2675, 0.2565, 0.2761]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
        for datas, labels in dataloader:
            dataset = ServerTaskDataset(datas,labels)
            break
    elif name =='FC100':
        task = FC100Task('/data/lpyx/FPKD_other/data/FC100', task_num=10)
        train, test = task.getTaskDataSet()
        train_dataset = train[t]
        dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=samples,
                                                 shuffle=True)
        # mean = [0.5071, 0.4867, 0.4408]
        # std = [0.2675, 0.2565, 0.2761]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
        for datas, labels in dataloader:
            dataset = ServerTaskDataset(datas, labels)
            break
    elif name=='miniimagenet':
        task = MiniImageTask(root='/data/lpyx/FedFPKD/data/mini-imagenet/',json_path="/data/lpyx/FedFPKD/data/mini-imagenet/classes_name.json",task_num=10)
        train, test = task.getTaskDataSet()
        train_dataset = train[t]
        dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=samples,
                                                 shuffle=True)
        # mean = [0.5071, 0.4867, 0.4408]
        # std = [0.2675, 0.2565, 0.2761]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
        for datas, labels in dataloader:
            dataset = ServerTaskDataset(datas, labels)
            break
    elif name == 'MiniGC':
        train_dataset = MiniGCDataset(samples, 10, 20)
        dataset = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=20,
                                              shuffle=True, collate_fn=collate_label)
    else:
        dataset = ServerDataset([torch.rand(size) for i in range(samples)])

    return dataset

def get_server_public_dataset(name,t=0):
    transform = None
    if name == 'MNIST':
        dataloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files/', train=True, download=False,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               # only 1 channel
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=samples, shuffle=True
        )
        for data, _ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_stddev = (0.2023, 0.1994, 0.2010)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ])
        training_set = torchvision.datasets.CIFAR10(root='./files', train=True, download=True,
                                                    transform=transform)
        dataloader = torch.utils.data.DataLoader(
            training_set,
            batch_size=samples,
            shuffle=True,
        )
        for data, _ in dataloader:
            dataset = ServerDataset(data)
            break
    elif name == 'CIFAR100':
        task = Cifar100Task('/data/lpyx/FedAgg/data/cifar100', task_num=10)
        train, test = task.getTaskDataSet()
        dataset = train[t]
        # mean = [0.5071, 0.4867, 0.4408]
        # std = [0.2675, 0.2565, 0.2761]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
    elif name =='FC100':
        task = FC100Task('/data/lpyx/FPKD_other/data/FC100', task_num=10)
        train, test = task.getTaskDataSet()
        dataset = train[t]
        # mean = [0.5071, 0.4867, 0.4408]
        # std = [0.2675, 0.2565, 0.2761]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
    elif name=='miniimagenet':
        task = MiniImageTask(root='/data/lpyx/FedFPKD/data/mini-imagenet/',json_path="/data/lpyx/FedFPKD/data/mini-imagenet/classes_name.json",task_num=10)
        train, test = task.getTaskDataSet()
        dataset = train[t]
        # mean = [0.5071, 0.4867, 0.4408]
        # std = [0.2675, 0.2565, 0.2761]
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )
    elif name == 'MiniGC':
        train_dataset = MiniGCDataset(200, 10, 20)
        dataset = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=20,
                                              shuffle=True, collate_fn=collate_label)


    else:
        dataset = ServerDataset([torch.rand(size) for i in range(samples)])

    return dataset

class ServerDataset(Dataset):
    def __init__(self, samples,transform = None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, item):
        img = self.samples[item]
        if self.transform is not None:
            img = self.transform(img)
        return img

class ServerTaskDataset(Dataset):
    def __init__(self, samples,labels,transform = None):
        self.samples = samples
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, item):
        img = self.samples[item]
        label = self.labels[item]
        if self.transform is not None:
            img = self.transform(img)
        return img,label



