import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets,models
from PIL import Image
from copy import deepcopy
class ClassDataset(Dataset):
    def __init__(self,data,idxs,cur_class):
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        self.data = data
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        self.class_dataset = []
        error_num = 50
        for idx in idxs:
            if self.data[idx][1]  == cur_class:
                temp = list(deepcopy(self.data[idx]))
                temp[1] = 1
                self.class_dataset.append(temp)
            elif error_num >=0:
                temp = list(deepcopy(self.data[idx]))
                temp[1] = 0
                self.class_dataset.append(temp)
                error_num -= 1
        

    def __len__(self) -> int:
        return len(self.class_dataset)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.class_dataset[index][0], self.class_dataset[index][1]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        return img, target
    
    

class ClientClassDataset(Dataset):
    def __init__(self,data,idxs,cur_class, c_tid, client_tid, error_num=50):
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        self.data = data
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        self.class_dataset = []
        
        error_cls = None
        for idx in idxs:
            raw_lable = self.data[idx][1]
            client_label = raw_lable - 10*client_tid + 10*c_tid
            
            if client_label == cur_class:
                temp = list(deepcopy(self.data[idx]))
                temp[1] = client_label
                self.class_dataset.append(temp)
            
            elif error_num > 0:
                if error_cls is None:
                    temp = list(deepcopy(self.data[idx]))
                    temp[1] = client_label
                    self.class_dataset.append(temp)
                    error_num -= 1

                    error_cls = client_label

                elif error_cls == client_label:
                    temp = list(deepcopy(self.data[idx]))
                    temp[1] = client_label
                    self.class_dataset.append(temp)
                    error_num -= 1
                    
        

    def __len__(self) -> int:
        return len(self.class_dataset)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.class_dataset[index][0], self.class_dataset[index][1]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        return img, target
