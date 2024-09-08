import sys, time, os
from typing import Union

import numpy as np
import torch
from copy import deepcopy
from FCL.FedKNOW.packnet.PacknetViT import PackNetViT
from FCL.FedKNOW.packnet.PacknetCNN import PackNetCNN

# from FCL.FedKNOW.funcs import freeze_model, compute_offsets, MultiClassCrossEntropy, store_grad, project2cone2, overwrite_grad
from FCL.FedKNOW.funcs import freeze_model, compute_offsets, MultiClassCrossEntropy, store_grad, project2cone2, overwrite_grad

sys.path.append('..')
import torch.nn as nn


class Appr(object):
    def __init__(self, model, packnet: Union[PackNetCNN, PackNetViT], packmodel, tr_dataloader, nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None, cid=None, ts_dataloader=None):
        self.cid = cid  # client id
        self.args = args
        self.device = args.device
        self.num_classes = args.num_classes
        self.model = model              # model of current task
        self.model_old = model          # model of last task
        self.pack = packnet             # mask, pruning   PackNet
        self.packmodel = packmodel      # model subsets
        self.nepochs = nepochs          # local_ep
        self.tr_dataloader = tr_dataloader
        self.ts_dataloader = ts_dataloader
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer(self.model, lr=args.lr)
        self.pack_optimizer = self._get_optimizer(self.packmodel, args.lr)
        self.e_rep = args.local_rep_ep
        self.old_task = -1
        self.grad_dims = []
        self.num_classes = args.num_classes     #TODO: 这里都是 10 

        # unused params
        self.fisher = None
        self.lr_min = lr_min * 1 / 3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.lamb = args.lamb

        # 记录特征提取部分模型各层参数维度
        for param in self.model.feature_net.parameters():
            self.grad_dims.append(param.data.numel())
        # self.select_grad_num = args.select_grad_num

        # metric
        self.model_iters = -1
        self.packmodel_iters = -1

    def _get_optimizer(self, model, lr=None):
        if lr is None: lr = self.args.lr
        
        if self.args.model in ['tiny_vit', 'tiny_pit']:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
        return optimizer
    
    def set_model(self, model):
        self.model = model

    def set_fisher(self, fisher):
        self.fisher = fisher

    def set_trData(self, tr_dataloader):
        self.tr_dataloader = tr_dataloader

    def set_tsData(self, ts_dataloader):
        self.ts_dataloader = ts_dataloader

    def train(self, t):
        """
        Args:
             t (int): task id
        """
        self.model.to(self.device)
        self.model_old.to(self.device)
        self.packmodel.to(self.device)
        oldpackmodel = deepcopy(self.packmodel)     # a copy of server global model

        # when new task: let current model as old model
        if t != self.old_task:
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            freeze_model(self.model_old)  # old model only support computing
            self.old_task = t
            # 用来记录当前任务的batch数
            self.packmodel_iters = -1
            self.model_iters = -1

        self.optimizer = self._get_optimizer(self.model, self.args.lr)
        
        # self.pack.on_init_end(self.packmodel, t)        # 将bias和normalize参数 设定为 requires_grad = False
        

        # train model
        if len(self.pack.masks) > t:
            self.pack.masks.pop()

        # local training n epochs
        for e in range(self.nepochs):
            # 与GEM算法不同的地方
            if e < self.e_rep:
                for name, para in self.model.named_parameters():    # fix body
                    if 'feature_net' in name:
                        para.requires_grad = False
                    else:
                        para.requires_grad = True
            else:
                for name, para in self.model.named_parameters():     # fix head
                    if 'feature_net' in name:
                        para.requires_grad = True
                    else:
                        para.requires_grad = False

            # only train reper for the first task
            if t == 0:
                self.train_epoch_rep(t, e, oldpackmodel)
            else:
                if e < self.e_rep:      # first train head
                    self.train_epoch_head(t)
                else:                   # then extract features
                    self.train_epoch_rep(t, e, oldpackmodel)

        # train task_specific params and pruning params
        self.pack.on_init_end(self.packmodel, t)        # 将bias和normalize参数 设定为 requires_grad = False
        for e in range(self.pack.total_epochs()):
            self.train_packnet(t)
            self.pack.on_epoch_end(self.packmodel.feature_net, e, t)

        train_loss, train_acc = 0, 0    # 这里都设置为 0， 因为 不需要每次测试Fedknow的 loss和准确率
        
        return self.fisher, train_loss, train_acc

    def train_packnet(self, t):
        """ Train weight subset of remaining parameters or fine-tune the pruned parameters"""
        self.packmodel.train()
        for iter, (images, targets) in enumerate(self.tr_dataloader):
            self.packmodel_iters += 1
            images = images.to(self.device)
            targets = (targets - self.num_classes * t).to(self.device)
            offset1, offset2 = compute_offsets(t, self.num_classes)
            outputs = self.packmodel.forward(images, t)[:, offset1:offset2]
            loss = self.ce(outputs, targets)
            self.pack_optimizer.zero_grad()
            loss.backward()

            # 当t<e_rep时，将所有旧任务模型参数梯度置零；当t>e_rep时，将当前任务之外的参数梯度置零。
            self.pack.on_after_backward(self.packmodel.feature_net, t)
            self.pack_optimizer.step()

            # 记录训练metric
            # acc, avg_loss = self._metric(outputs, targets, loss)
            # self.args.writer.add_scalar(f'Appr-{self.cid}/PackModel/Task-{t}/Train Acc', np.round(acc, 3), self.packmodel_iters + 1)
            # self.args.writer.add_scalar(f'Appr-{self.cid}/PackModel/Task-{t}/Train Loss', np.round(avg_loss, 3), self.packmodel_iters + 1)

            # 记录test metric
            # self.eval(model_name='packmodel', t=t, train=False)

    def train_epoch_head(self, t):
        """ Train and make the head close to the old_model """
        self.model.train()
        for iter, (images, targets) in enumerate(self.tr_dataloader):
            self.model_iters += 1
            images = images.to(self.device)
            targets = (targets - self.num_classes * t).to(self.device)

            # make head close to the last model
            offset1, offset2 = compute_offsets(t, self.num_classes)
            preLabels = self.model_old.forward(images, t, pre=True)[:, 0: offset1]
            preoutputs = self.model.forward(images, t, pre=True)[:, 0: offset1]
            self.optimizer.zero_grad()
            self.model.zero_grad()
            memoryloss=MultiClassCrossEntropy(preoutputs, preLabels, t, T=1)
            memoryloss.backward()
            self.optimizer.step()

            # train model head of the current task
            self.optimizer.zero_grad()
            self.model.zero_grad()
            outputs = self.model.forward(images, t)[:, offset1:offset2]
            loss = self.ce(outputs, targets)

            # backward to update self.model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录训练metric
            # acc, avg_loss = self._metric(outputs, targets, loss)
            # self.args.writer.add_scalar(f'Appr-{self.cid}/self.model/Task-{t}/Train Acc', np.round(acc, 3), self.model_iters + 1)
            # self.args.writer.add_scalar(f'Appr-{self.cid}/self.model/Task-{t}/Train Loss', np.round(avg_loss, 3), self.model_iters + 1)

            # 记录test metric
            # self.eval(model_name='self.model', t=t, train=False)

    def train_epoch_rep(self, t, epoch, oldpackmodel):
        """Compute the rotated feature_net params and update self.model.feature_net by this params. """
        self.model.train()
        self.packmodel.train()
        # Loop batches
        for iter, (images, targets) in enumerate(self.tr_dataloader):
            self.model_iters += 1

            images = images.to(self.device)
            targets = (targets - self.num_classes * t).to(self.device)

            # build an empty grads tensor for grad store
            grads = torch.Tensor(sum(self.grad_dims), 2+t)
            offset1, offset2 = compute_offsets(t, self.num_classes)
            grads = grads.to(self.device)

            K = 2   # 只回溯最近的2个task
            # Gradient Restorer
            if t > 0:
                # compute and store grads about old tasks using both current and last versions models
                preLabels = self.model_old.forward(images, t, pre=True)[:, 0: offset1]
                preoutputs = self.model.forward(images, t, pre=True)[:, 0: offset1]
                self.model.zero_grad()
                self.optimizer.zero_grad()
                pre_loss = MultiClassCrossEntropy(preoutputs, preLabels, t, T=2)
                pre_loss.backward()
                store_grad(self.model.feature_net.parameters, grads, self.grad_dims, 0)
                # if t >= self.select_grad_num:
                #     t = self.select_grad_num -1

                # compute and store the grads of past tasks' models ： 0 -> (t-1)
                # for i in range(t):
                if t > K:
                    old_tids = list(range(t))[-K:]
                else:
                    old_tids = list(range(t))
                    
                for i in old_tids:
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    begin, end = compute_offsets(i, self.num_classes)

                    temppackmodel = deepcopy(oldpackmodel).to(self.device)
                    temppackmodel.train()

                    self.pack.apply_eval_mask(task_idx=i, model=temppackmodel.feature_net)

                    preoutputs = self.model.forward(images, t, pre=True)[:, begin:end]

                    with torch.no_grad():
                        oldLabels = temppackmodel.forward(images, i)[:, begin:end]
                    memoryloss = MultiClassCrossEntropy(preoutputs, oldLabels, i, T=2)      # compute gra
                    memoryloss.backward()

                    store_grad(self.model.feature_net.parameters, grads, self.grad_dims, i+1)
                    del temppackmodel

            # 计算当前任务model的梯度
            outputs = self.model.forward(images, t)[:, offset1:offset2]
            loss = self.ce(outputs, targets)
            self.model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Integrater
            if t > 0:
                store_grad(self.model.feature_net.parameters, grads, self.grad_dims, t+1)
                taskl = [i for i in range(t+2)]
                indx = torch.LongTensor(taskl[:-1]).to(self.device)
                errindx = torch.LongTensor(0).to(self.device)
                dotp = torch.mm(grads[:, 1].unsqueeze(0),
                                grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    project2cone2(grads[:, t+1].unsqueeze(1),
                                  grads.index_select(1, indx), grads.index_select(1, errindx))
                    # copy gradients back
                    overwrite_grad(self.model.feature_net.parameters, grads[:, t+1],
                                   self.grad_dims)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50, norm_type=2)
            self.optimizer.step()

            # 记录训练metric
            acc, avg_loss = self._metric(outputs, targets, loss)
            # self.args.writer.add_scalar(f'Appr-{self.cid}/self.model/Task-{t}/Train Acc', np.round(acc, 3), self.model_iters + 1)
            # self.args.writer.add_scalar(f'Appr-{self.cid}/self.model/Task-{t}/Train Loss', np.round(avg_loss, 3), self.model_iters + 1)

            # 记录test metric
            # self.eval(model_name='self.model', t=t, train=False)

        return

    def _metric(self, outputs, targets, loss):
        _, pred = outputs.max(1)
        hits = (pred == targets).float()
        corr = hits.sum().data.cpu().numpy()
        acc = corr / targets.size(0)
        avg_loss = loss.data.cpu().numpy()
        return acc, avg_loss

    def moretrain(self, t):
        self.packmodel.to(self.device)
        for e in range(self.nepochs):
            self.train_packnet(t)
            self.pack.on_epoch_end(self.packmodel.feature_net, e, t)

    def eval(self, model_name:str, t, train=True, not_log=False):
        total_loss = 0
        total_acc = 0
        total_num = 0

        if 'pack' in model_name:
            _model = self.packmodel
            iter = self.packmodel_iters
        else:
            _model = self.model
            iter = self.model_iters

        _model.eval()

        if train:
            dataloaders = self.tr_dataloader
        else:
            dataloaders = self.ts_dataloader

        # Loop batches
        with torch.no_grad():
            for images, targets in dataloaders:
                images = images.to(self.device)
                targets = (targets - self.num_classes*t).to(self.device)
                # Forward
                offset1, offset2 = compute_offsets(t, self.num_classes)
                output = _model.forward(images,t)[:, offset1:offset2]

                loss = self.ce(output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        loss = total_loss/total_num
        acc = total_acc/total_num

        if not not_log:
            mode = 'Train' if train else 'Test'
            # self.args.writer.add_scalar(f'Appr {self.cid}/{model_name}/Task {t}/{mode} Acc', np.round(acc, 3), iter+1)
            # self.args.writer.add_scalar(f'Appr {self.cid}/{model_name}/Task {t}/{mode} Loss', np.round(loss, 3), iter+1)

        return loss, acc


    def criterion(self, t):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.feature_net.named_parameters(), self.model_old.feature_net.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return self.lamb * loss_reg


def LongLifeTrain(args, appr, task, writer, idx):
    # t = aggNum // args.round

    # print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    # print('cur task:' + str(t))
    # print('*' * 100)
    # print('*' * 100)
    # # Get data
    # task = t

    # Train
    fisher, loss, _ = appr.train(task)
    print('-' * 100)
    return appr.model.state_dict(), fisher, loss, 0
