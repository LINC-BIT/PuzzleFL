import torch
from torch import nn
from torch.utils.data import Dataset
import os
import random
from torchvision import transforms, datasets
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
from ClientTrain.AggModel.resnet import Resnet18,WideResnet
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def make_dataset(
    directory,
    class_to_idx,
    extensions = None,
    is_valid_file = None,
):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            add_class_len = 1300 - len(fnames)
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
            if add_class_len!=0:
                add_class_samples = random.sample(fnames,add_class_len)
                for fname in sorted(add_class_samples):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
    return instances
def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

class ImageNetTask():
    def __init__(self,root,task_num=1):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.root = root
        self.task_num = task_num
        self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        self.transform = transforms.Compose([
                transforms.Resize(36),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                normalize,
            ])


    def getTaskDataSet(self):
        trainDataset = myimagnet(root=self.root, split='train', transform=self.transform_train, download=False, task_num=self.task_num)
        train_task_datasets = [ImageNetDataset(data, self.transform_train) for data in trainDataset.task_datasets]
        testDataset = myimagnet(root=self.root, split='val', download=False, task_num=self.task_num)
        test_task_datasets = [ImageNetDataset(data, self.transform) for data in testDataset.task_datasets]
        return train_task_datasets,test_task_datasets

class ImageNetDataset(Dataset):
    def __init__(self,data,transform):
        self.data = data
        self.transform = transform
        self.n_class_num = 10

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        path, target = self.data[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = target % 10
        return sample, target

class myimagnet(datasets.ImageNet):
    def __init__(self, root: str, split: str = 'train', download= None,task_num=1, **kwargs) -> None:
        super(myimagnet,self).__init__(root,split,download,**kwargs)
        if split == 'train':
            classes, class_to_idx = self._find_classes(self.split_folder)
            samples = make_dataset(self.split_folder, class_to_idx, IMG_EXTENSIONS, None)
            self.samples = samples
            self.targets = [s[1] for s in samples]
            n_per_class = 1300
            n_per_task = n_per_class * 1000//task_num
            self.task_datasets = []
            for i in range(task_num):
                task_dataset = samples[i*n_per_task:(i+1)*n_per_task]
                self.task_datasets.append(task_dataset)
        else:
            n_per_class = 50
            n_per_task = n_per_class * 1000 // task_num
            self.task_datasets = []
            for i in range(task_num):
                task_dataset = self.samples[i * n_per_task:(i + 1) * n_per_task]
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
data = ImageNetTask('/data/lpyx/data/ImageNet',task_num=100)
trains,tests = data.getTaskDataSet()
train_dataset = trains[20]
test_dataset = tests[20]
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=256,
                                           shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=512,
                                           shuffle=False)
model = SixCNN([3,32,32],10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
ce = torch.nn.CrossEntropyLoss()
for epoch in range(30):
    model.train()
    for images, targets in train_dataloader:
        images = images.cuda()
        targets = targets.cuda()
        outputs = model(images)
        loss = ce(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("acc:",eval(model,test_dataloader))
