import sys, time, os
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from ClientTrain.utils import *
from torch.utils.tensorboard import SummaryWriter
import quadprog

from util.prune import InfoPrune
sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn
from util.classData import ClassDataset, ClientClassDataset
from torch.utils.data import DataLoader


from torch.utils.data import Dataset
from torchvision import transforms, datasets,models

# ZUO
def fisher_matrix_diag_for_one_class(t, dataloader, model,device):
    # Init
    fisher = {}
    # for n, p in model.feature_net.named_parameters():
    for n, p in model.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    offset1, offset2 = compute_offsets(t, 10)
    all_num = 0
    
    for images,target in dataloader:
        images = images.to(device)
        # print('Raw target:', target)
        target = (target - 10 * t).to(device)
        all_num += images.shape[0]
        # Forward and backward
        model.zero_grad()
        outputs = model.forward(images, t)[:, offset1: offset2]
        # print("target:", target)
        # print("offset1:", offset1)
        try:
            loss = criterion(outputs, target)
            loss.backward()
        except:
            print("Paus")
        # Get gradients
        # for n, p in model.feature_net.named_parameters():
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += images.shape[0] * p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        # for n, _ in model.feature_net.named_parameters():
        for n, _ in model.named_parameters():
            fisher[n] = fisher[n] / all_num
    return fisher

# ZUO
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
        img, target = self.class_dataset[index][0], self.class_dataset[index][1]
        return img, target
    

def compute_offsets(task, nc_per_task, is_cifar=True):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

def class_fisher_matrix_diag(dataloader, model,device):
    # Init
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    all_num = 0
    for images,target in dataloader:
        images = images.to(device)
        target = (target).to(device)
        all_num += images.shape[0]
        # Forward and backward
        model.zero_grad()
        outputs = model.forward(images, 0)
        loss = criterion(outputs, target)
        loss.backward()
        # Get gradients
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += images.shape[0] * p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n, _ in model.named_parameters():
            fisher[n] = fisher[n] / all_num
    return fisher




def fisher_matrix_diag(t,dataloader, model,device):
    # Init
    fisher = {}
    for n, p in model.feature_net.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    offset1, offset2 = compute_offsets(t, 10)
    all_num = 0
    for images,target in dataloader:
        images = images.to(device)
        target = (target - 10 * t).to(device)
        all_num += images.shape[0]
        # Forward and backward
        model.zero_grad()
        outputs = model.forward(images, t)[:, offset1: offset2]
        loss = criterion(outputs, target)
        loss.backward()
        # Get gradients
        for n, p in model.feature_net.named_parameters():
            if p.grad is not None:
                fisher[n] += images.shape[0] * p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n, _ in model.feature_net.named_parameters():
            fisher[n] = fisher[n] / all_num
    return fisher
def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1
def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1
def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
def MultiClassCrossEntropy(logits, labels, t,T=2):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    label = torch.softmax(labels / T, dim=1)
        # print('outputs: ', outputs)
        # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * label, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)

    # print('OUT: ', outputs)
    return outputs

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return
class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, tr_dataloader,nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None, kd_model=None,client_task=None, idx=None):
        self.idx = idx
        self.model = model
        self.model_old = model
        self.kd_state_dict = deepcopy(model.state_dict()) # TODO: Deepcopy
        self.device = args.device
        self.fisher = None
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_min = lr_min * 1 / 3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.args = args
        self.ce = torch.nn.CrossEntropyLoss()
        self.softloss = torch.nn.KLDivLoss()
        self.optimizer = self._get_optimizer()
        self.lamb = args.lamb
        self.e_rep = args.local_rep_ep
        self.old_task=-1
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.first_train = True
        self.task_class_number = 10
        self.send_para = 0.10   # TODO: 这里要改低， 普遍 分数为 0.112
        self.recive_para = 0.1
        self.task_class_model = []
        self.client_task = client_task
        # for i,task in enumerate(client_task):
        #     cur_task_class = []
        #     for j in range(self.task_class_number):
        #         cur_task_class.append(task*self.task_class_number + j)
        for i in range(100):
            ct = self.client_task.index(i//self.task_class_number)
            self.task_class_model.append({'model':None,'mask':None,'class':i,'score':None, 'client_class':ct*self.task_class_number+i%self.task_class_number})
                                         
            
        self.extra_sample_num = 50  # zuo: 构建类别数据集时，除了指定类别的样本外，再额外增加10个另外类别的样本
        
        return
    
    def judge_class(self,cur_class):
        cur_datas = self.tr_dataloader.dataset.dataset
        cur_idxs = self.tr_dataloader.dataset.idxs
        cur_class_score = 0
        
        for idx in cur_idxs:
            if cur_datas[idx][1] == cur_class:
                cur_class_score+=1
                
        return cur_class_score/len(cur_idxs)
    
    def new_kd(self):
        self.cur_kd = deepcopy(self.model)
        self.cur_kd.load_state_dict(self.kd_state_dict)
    
    def extract_task_class(self,t):
        
        for i in range(self.task_class_number):
            cur_class = self.client_task[t] * self.task_class_number + i    # 获取当前客户端第t个任务的第i个类别的索引；即样本真实类别
            _client_class = t*self.task_class_number + i                # 获得样本真实标签在当前客户端上的顺位标签_client_class
            
            data_loader_for_extract = self._get_class_dataloader(_client_class, t, train=True)
            
            if data_loader_for_extract is not None:
                #### 剪枝
                cur_class_info = fisher_matrix_diag_for_one_class(t, data_loader_for_extract, deepcopy(self.cur_kd),self.device)
                model_prune = InfoPrune(device=self.device,info=cur_class_info)
                prune_kd_state_dict, pruned_model = model_prune.prune_one_model(deepcopy(self.cur_kd), None, 0)
                self.task_class_model[cur_class]['model'] = prune_kd_state_dict
                self.task_class_model[cur_class]['mask'] = model_prune.mask
                self.task_class_model[cur_class]['score'] = self.judge_class(cur_class)
            else:
                self.task_class_model[cur_class]['score'] = 0   # TODO: 这里本来可以设置None，但是为了方便后续的判断，这里设置为0       
            
    def aggregation(self,other_clients, t):
        for other_client in other_clients:
            cur_class = other_client['class']
            oth_model_state = other_client['model']
            oth_model = deepcopy(self.cur_kd)
            oth_model.load_state_dict(oth_model_state)
            cur_class_state = self.task_class_model[cur_class]['model']
            cur_mask = self.task_class_model[cur_class]['mask']
            cur_class_model = deepcopy(self.cur_kd)
            cur_class_model.load_state_dict(cur_class_state)
            ## Train
            cur_class_model.train()
            cur_optimizer = torch.optim.Adam(cur_class_model.parameters(), lr=0.0005)
            # Forward current model
            cur_class_model.to(self.device)
            oth_model.to(self.device)
            for e in range(5):
                for images,targets in self.tr_dataloader:
                    images = images.to(self.device)
                    targets = (targets % 10).to(self.device)
                    cur_outputs = cur_class_model.forward(images, 0)
                    oth_outputs = oth_model.forward(images, 0)
                    loss = MultiClassCrossEntropy(cur_outputs, oth_outputs,0,T=2)
                    cur_optimizer.zero_grad()
                    loss.backward()
                    for name, param_layer in cur_class_model.named_parameters():
                        if 'bias' not in name and  param_layer.grad is not None:
                            param_layer.grad *= cur_mask[name]
                    cur_optimizer.step()
            self.task_class_model[cur_class]['model'] = cur_class_model.state_dict()
            self.task_class_model[cur_class]['score'] += 0.01   

    def update_local_class_knowledge(self, t):

        for i in range(self.task_class_number):
            cur_class = self.client_task[t] * self.task_class_number + i    # 获取当前客户端第t个任务的第i个类别的索引；即样本真实类别
            _client_class = t*self.task_class_number + i                # 获得样本真实标签在当前客户端上的顺位标签_client_class
            
            data_loader_for_extract = self._get_class_dataloader(_client_class, t, train=True)
            
            if data_loader_for_extract is not None:
                grads = torch.zeros(self.grad_dims).to(self.device)  # 存储梯度
                store_grad(self.model.parameters(), grads, self.grad_dims, cur_class)
       
                old_class_state = self.task_class_model[cur_class]['model']
                old_class_model = deepcopy(self.cur_kd)
                old_class_model.load_state_dict(old_class_state)
                old_class_model.to(self.device)

                for images,targets in data_loader_for_extract:
                    images = images.to(self.device)
                    targets = (targets % 10).to(self.device)
                    cur_outputs = old_class_model.forward(images, 0)
                    oth_outputs = old_class_model.forward(images, 0)
                    loss = MultiClassCrossEntropy(cur_outputs, oth_outputs,0,T=2)

                old_grads = torch.zeros(self.grad_dims).to(self.device)
                store_grad(old_class_model.parameters(), old_grads, self.grad_dims, cur_class)

                a = grads.unsqueeze(0)  
                b = old_grads.unsqueeze(0)  
                dotp = torch.mm(a, b.T)  

                if (dotp < 0).sum() != 0:
                    project2cone2(grads.unsqueeze(1), old_grads.unsqueeze(1), self.margin)
                    overwrite_grad(self.model.parameters(), grads, self.grad_dims)

        # 使用优化器更新当前模型
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        optimizer.zero_grad()

        for images, targets in data_loader_for_extract:
            images = images.to(self.device)
            targets = (targets % 10).to(self.device)
            cur_outputs = self.model.forward(images, 0)
            loss = self.ce(cur_outputs, targets)
            loss.backward()
        optimizer.step()


    def send_class(self, t):
        s_classes = []
        for i in range(self.task_class_number):
            cur_class = self.client_task[t] * self.task_class_number + i 
            i = cur_class
            
            if self.task_class_model[i]['score'] is not None:
                if self.task_class_model[i]['score'] > self.send_para:
                    s_classes.append(i)
        return s_classes
    
    def recive_class(self, t):
        r_classes = []
        for i in range(self.task_class_number):
            cur_class = self.client_task[t] * self.task_class_number + i 
            i = cur_class
             
            if self.task_class_model[i]['score'] is not None:
                if self.task_class_model[i]['score'] < self.recive_para:
                    r_classes.append(i)
        return r_classes
    
    def get_task_class(self,class_index, t=None):
        class_info = []
        for i in class_index:
            class_info.append(self.task_class_model[i])     
        return class_info       
            
    def set_model(self,model):
        self.model = model

    def set_fisher(self,fisher):
        self.fisher = fisher
        
    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader
        
        
    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        if 'vit' in self.args.model:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)
        # self.momentum = 0.9
        # self.weight_decay = 0.0001
        #
        # optimizer =  torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum,
        #                       weight_decay=self.weight_decay)
        return optimizer
    
    def train(self, t):
        if t!=self.old_task:
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            freeze_model(self.model_old)  # Freeze the weights
            self.old_task=t
            self.first_train = True
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        # Loop epochs
        # self.train_kd(t)
        for e in range(self.nepochs):
            # Train
            self.train_epoch_rep(t, e)
            train_loss, train_acc = self.eval(t)
            if e % self.e_rep == self.e_rep -1:
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                    e + 1,  train_loss, 100 * train_acc), end='')
        # Fisher ops
        fisher_old = {}
        if t>0:
            for n, _ in self.model.feature_net.named_parameters():
                fisher_old[n] = self.fisher[n].clone()
        self.fisher = fisher_matrix_diag(t,self.tr_dataloader, self.model,self.device)
        if t > 0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n, _ in self.model.feature_net.named_parameters():
                self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (
                        t + 1)  # Checked: it is better than the other option

        self.first_train=False
        return train_loss, train_acc

    def train_kd(self,t):
        print(self.eval(t,train=True,model=self.cur_kd))
        self.cur_kd.train()
        self.cur_kd = self.cur_kd.to(self.device)
        
        kd_optimizer = torch.optim.Adam(self.cur_kd.parameters(), lr=0.0005)
        
        if 'vit' in self.args.model:
            kd_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0.05)
        
        for e in range(50):
        # Forward current model
            for images,targets in self.tr_dataloader:
                images = images.to(self.device)
                targets = (targets - 10 * t).to(self.device)
                offset1, offset2 = compute_offsets(t, 10)
                kd_outputs = self.cur_kd.forward(images, t)[:, offset1:offset2]
                loss = self.ce(kd_outputs, targets)
                kd_optimizer.zero_grad()
                loss.backward()
                kd_optimizer.step()
        print(self.eval(t,train=True,model=self.cur_kd))
        


    def train_epoch(self,t):
        self.model.train()
        for images,targets in self.tr_dataloader:
            images = images.to(self.device)
            targets = (targets - 10 * t).to(self.device)
            # Forward current model
            offset1, offset2 = compute_offsets(t, 10)
            outputs = self.model.forward(images,t)[:,offset1:offset2]
            loss = self.ce(outputs, targets)
            ## 根据这个损失计算梯度，变换此梯度
            # Backward
            loss.backward()
            self.optimizer.step()
        return
    
    def _get_class_dataloader(self, client_class, task, train=False):
        # ZUO : get class dataloader
        if train:
            cur_datas = self.tr_dataloader.dataset.dataset
            cur_idxs = self.tr_dataloader.dataset.idxs
        else:
            cur_datas = self.ts_dataloader.dataset.dataset
            cur_idxs = self.ts_dataloader.dataset.idxs
            
        
        one_class_dataset = ClientClassDataset(cur_datas, cur_idxs, client_class, task, self.client_task[task], error_num=self.extra_sample_num if train else 0)
        
        
        if train:
            if len(one_class_dataset) > self.extra_sample_num:
                cur_class_dataloader = DataLoader(one_class_dataset, batch_size=32,shuffle=train) 
                
                data_info = {}
                for data in one_class_dataset:
                    if data[1] not in data_info:
                        data_info[data[1]] = 1
                    else:
                        data_info[data[1]] += 1
                print(f"{'TRAIN' if train else 'TEST'} Class Dataset info is {data_info}")
            
                return cur_class_dataloader
            else:
                return None
        else:
            if len(one_class_dataset) > 0:
                cur_class_dataloader = DataLoader(one_class_dataset, batch_size=32,shuffle=train) 
                
                data_info = {}
                for data in one_class_dataset:
                    if data[1] not in data_info:
                        data_info[data[1]] = 1
                    else:
                        data_info[data[1]] += 1
                print(f"{'TRAIN' if train else 'TEST'} Class Dataset info is {data_info}")
            
                return cur_class_dataloader
            else:
                return None
        
        
    def train_epoch_rep(self, t, epoch, kd_lambda = 0.0):
        self.model.train()
        # Loop batches
        for images,targets in self.tr_dataloader:
            # Forward current model
            images = images.to(self.device)
            targets = (targets - 10 * t).to(self.device)
            pre_loss = 0
            offset1, offset2 = compute_offsets(t, 10)   # 当前任务 offsets 
            cur_outputs = self.model.forward(images, t)[:, offset1: offset2]
            
            pre_loss = 0                
            pre_number = 0
            
            for i in range(self.task_class_number):
                real_class = self.client_task[t] * self.task_class_number + i 
                
                pre_class = self.task_class_model[real_class]       # 拿到指定类别的模型信息
                if pre_class['model'] is not None and pre_class['score'] > 0:
                    pre_class_state = pre_class['model']
                    pre_class_model = deepcopy(self.cur_kd)
                    pre_class_model.load_state_dict(pre_class_state)
                    
                    ### zuo: 类别模型输出logits
                    pre_client_class = pre_class['client_class']
                    pre_tid = pre_client_class//self.task_class_number  # 获取client_class所在的旧任务id
                    pre_offset1, pre_offset2 = compute_offsets(pre_tid, 10)
                    pre_output = pre_class_model.forward(images, pre_tid)[:, pre_offset1: pre_offset2]
                    
                    # 蒸馏损失
                    tempr = 2
                    pre_loss += nn.KLDivLoss()(nn.functional.log_softmax(cur_outputs / tempr, dim=1),
                                   nn.functional.softmax(pre_output / tempr, dim=1)) * (tempr ** 2)
                    pre_number += 1
            
                    if pre_loss == 0:
                        print("---> KD loss is 0!")
                
            loss = self.criterion(t, cur_outputs, targets) + pre_loss
            
            ## 根据这个损失计算梯度，变换此梯度
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def eval(self, t,train=True,model=None):
        total_loss = 0
        total_acc = 0
        total_num = 0
        if train:
            dataloaders = self.tr_dataloader
        if model is None:
            model= self.model
        # Loop batches
        model.eval()
        with torch.no_grad():
            for images,targets in dataloaders:
                images = images.to(self.device)
                targets = (targets - 10*t).to(self.device)
                # Forward
                offset1, offset2 = compute_offsets(t, 10)
                output = model.forward(images,t)[:,offset1:offset2]

                loss = self.criterion(t, output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.feature_net.named_parameters(), self.model_old.feature_net.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return self.ce(output, targets) + self.lamb  * loss_reg
    
    def pre_critertion(self,pre_model,pre_mask):
        loss_reg = 0
        for (name, param), (_, param_old) in zip(self.model.named_parameters(), pre_model.named_parameters()):
            if 'last' not in name:
                loss_reg += torch.sum(pre_mask[name] * (param_old - param).pow(2)) / 2
        return loss_reg


def LongLifeTrain(args, appr, aggNum, writer,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    taskcla = []
    for i in range(10):
        taskcla.append((i, 10))
    # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    t = aggNum // args.round
    print('cur task:'+ str(t))
    r = aggNum % args.round
    # for t, ncla in taskcla:

    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    task = t

    # Train
    loss,_ = appr.train(task)
    print('-' * 100)

    return appr.model.state_dict(), loss, 0

def LongLifeTest(args, appr, t, testdatas, aggNum, writer):
    acc = np.zeros((1, t), dtype=np.float32)
    lss = np.zeros((1, t), dtype=np.float32)
    t = aggNum // args.round
    r = aggNum % args.round
    for u in range(t + 1):
        xtest = testdatas[u][0].cuda()
        ytest = (testdatas[u][1] - u * 10).cuda()
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss,
                                                                              100 * test_acc))
        acc[0, u] = test_acc
        lss[0, u] = test_loss
    # Save
    mean_acc = np.mean(acc[0, :t+1])
    mean_lss = np.mean(lss[0, :t])
    print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[0, :t+1])))
    print('Average loss={:5.1f}'.format(np.mean(lss[0, :t+1])))
    print('Save at ' + args.output)
    if r == args.round - 1:
        writer.add_scalar('task_finish_and _agg', mean_acc, t + 1)
    # np.savetxt(args.agg_output + 'aggNum_already_'+str(aggNum)+args.log_name, acc, '%.4f')
    return mean_lss, mean_acc
