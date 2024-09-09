import sys, time, os
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

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


class Appr(object):
    def __init__(self, model, tr_dataloader, nepochs=100, lr=0.001, lr_min=1e-6, idx=None, args=None, client_task=None, cell_neighbors=None):
        self.model = model
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.old_task=-1
        self.args = args
        self.device = args.device
        
        # self.client_task = client_task  # 存储了当前客户端的任务序列
        # self.cell_neighbors = cell_neighbors    # 与当前客户端同一个cell的客户端序号
    
    def set_model(self,model):
        self.model = model

    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader
        
    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
     
        return optimizer
    
    def train(self, t):
        if t!=self.old_task:
            self.old_task=t
            
        self.optimizer = self._get_optimizer(self.lr)
        
        # Loop epochs
        for e in range(self.nepochs):
            self.train_epoch(t)
            train_loss, train_acc = self.eval(t)
            if e % self.nepochs == self.nepochs -1:
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                    e + 1,  train_loss, 100 * train_acc), end='')
        # Fisher ops
        return train_loss, train_acc

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
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return

    def eval(self, t,train=True):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        
        if train:
            dataloaders = self.tr_dataloader

        # Loop batches
        with torch.no_grad():
            for images,targets in dataloaders:
                images = images.to(self.device)
                targets = (targets - 10*t).to(self.device)
                
                # Forward
                offset1, offset2 = compute_offsets(t, 10)
                output = self.model.forward(images,t)[:,offset1:offset2]

                loss = self.ce(output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num
   
    
def LongLifeTrain(args, appr, aggNum, idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    taskcla = []
    for i in range(10):
        taskcla.append((i, 10))
    t = aggNum // args.round
    print('cur task:'+ str(t))
    r = aggNum % args.round

    print('*' * 100)
    print('*' * 100)

    # Get data
    task = t

    # Train
    loss,_ = appr.train(task)
    print('-' * 100)
    return appr.model.state_dict(),loss,0