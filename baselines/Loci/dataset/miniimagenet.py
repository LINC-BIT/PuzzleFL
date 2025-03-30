import os
import json
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
from ClientTrain.models.Nets import RepTail
import torch.nn as nn
import numpy as np
class MiniImageTask():
    def __init__(self,root,json_path,task_num=1):
        self.root = root
        self.csv_name = {
            'train':"new_train.csv",
            'test': "new_val.csv"
        }
        self.task_num = task_num
        self.json_path = json_path
        self.data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(32),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "test": transforms.Compose([transforms.Resize(36),
                                       transforms.CenterCrop(32),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    def getTaskDataSet(self):
        trainDataset = MyMiniImageDataSet(root_dir=self.root, csv_name=self.csv_name['train'], json_path = self.json_path,task=self.task_num)
        train_task_datasets = [MiniImageDataset(data, transform=self.data_transform['train']) for data in trainDataset.task_datasets]
        testDataset = MyMiniImageDataSet(root_dir=self.root, csv_name=self.csv_name['test'], json_path = self.json_path,task=self.task_num)
        test_task_datasets = [MiniImageDataset(data, transform=self.data_transform['test']) for data in testDataset.task_datasets]
        return train_task_datasets,test_task_datasets
class MiniImageDataset(Dataset):
    def __init__(self,data,transform):
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int):
        imgpath, target = self.data[item]['image'], self.data[item]['label']

        img = Image.open(imgpath)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
class MyMiniImageDataSet():
    """自定义数据集"""
    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 task:int):
        images_dir = os.path.join(root_dir, "images")
        self.task=task
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)
        samples = self.total_num//task
        self.task_datasets = []
        for i in range(task):
            task_dataset = []
            for j in range(samples):
                zipped = {}
                zipped['image'] = self.img_paths[i*samples+j]
                zipped['label'] = self.img_label[i*samples+j]
                task_dataset.append(zipped)
            self.task_datasets.append(task_dataset)
def eval(model,dataloaders):
    total_loss = 0
    total_acc = 0
    total_num = 0
    model = model.cuda()
    # Loop batches
    model.eval()
    with torch.no_grad():
        for images, targets in dataloaders:
            images = images.cuda()
            targets = targets.cuda()
            # Forward
            output = model.forward(images,t=-1)
            _, pred = output.max(1)
            hits = (pred == targets).float()
            # Log
            total_acc += hits.sum().data.cpu().numpy()
            total_num += len(images)

    return total_acc / total_num
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))
class SixCNN(nn.Module):
    def __init__(self, inputsize, outputsize=100,nc_per_task = 10):
        super().__init__()
        self.outputsize = outputsize
        ncha, size, _ = inputsize
        self.conv1 = nn.Conv2d(ncha, 32, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(size, 3, padding=1)  # 32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 32
        s = s // 2  # 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        s = s // 2  # 8
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        #         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        #         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s // 2  # 4
        self.fc1 = nn.Linear(s * s * 128, 1024, bias=False)  # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        # self.fc2 = nn.Linear(256, 100)
        self.relu = torch.nn.ReLU()
        self.last = nn.Linear(1024, outputsize, bias=False)
        self.nc_per_task = nc_per_task
    def forward(self, x, t=-1, pre=False, is_cifar=True, avg_act=False):
        if x.size(1) != 3:
            bsz = x.size(0)
            x = x.view(bsz, 3, 32, 32)
        act1 = self.relu(self.conv1(x))
        act2 = self.relu(self.conv2(act1))
        h = self.drop1(self.MaxPool(act2))
        act3 = self.relu(self.conv3(h))
        act4 = self.relu(self.conv4(act3))
        h = self.drop1(self.MaxPool(act4))
        act5 = self.relu(self.conv5(h))
        act6 = self.relu(self.conv6(act5))
        h = self.drop1(self.MaxPool(act6))
        h = h.view(x.shape[0], -1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        output = self.last(h)
        if is_cifar and t != -1:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.outputsize:
                output[:, offset2:self.outputsize].data.fill_(-10e10)
        return output
class testmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # SENET
        # self.net = models.resnext50_32x4d()
        self.net = models.DenseNet()
        # self.net = models.mobilenet_v2()
        # models.resnet152()
        # self.net = models.shufflenet_v2_x0_5()
        # self.net.load_state_dict(torch.load('../pre_train/shufflenetv2_x0.5-f707e7126e.pth'))
        self.net.fc = torch.nn.Linear(1024,10)
        self.weight_keys =[]
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x):
        out = self.net(x)
        return out
if __name__ == '__main__':
    m = MiniImageTask(root='/data/lpyx/FedFPKD/data/mini-imagenet/',json_path="/data/lpyx/FedFPKD/data/mini-imagenet/classes_name.json",task_num=10)
    t,te =m.getTaskDataSet()
    train_dataset = t[0]
    val_dataset = te[0]

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=1,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=256,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=val_dataset.collate_fn)
    model = SixCNN([3, 32, 32], 10).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    ce = torch.nn.CrossEntropyLoss()
    for epoch in range(30):
        model.train()
        for images, targets in train_loader:
            images = images.cuda()
            targets = targets.cuda()
            outputs = model(images)
            loss = ce(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("acc:", eval(model, val_loader))





