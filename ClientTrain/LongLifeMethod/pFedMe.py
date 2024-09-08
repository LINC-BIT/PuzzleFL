# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import quadprog


# Auxiliary functions useful for GEM's inner optimization.
from torch.optim import Optimizer


def compute_offsets(task, nc_per_task, is_cifar):
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


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = deepcopy(local_weight_updated)
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update.parameters()):
                p.data = p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']
class Appr():
    def __init__(self,
                 model,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Appr, self).__init__()
        self.margin = args.memory_strength
        self.is_cifar = True
        self.net = model
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.opt = pFedMeOptimizer(self.net.parameters(), lr = args.lr)
        self.gpu = True
        self.lr = args.lr
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.K = args.pFedMeK
        self.local_model = deepcopy(self.net)
        self.lamda = args.pFedMelamda
        self.perlr = args.pFedMelr
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
    def set_model(self,model):
        self.net = model
    def update_parameters(self, new_params):
        for param , new_param in zip(self.net.parameters(), new_params):
            param.data = new_param.data.clone()
        for param in self.net.parameters():
            a = 1
    def observe(self, x, t, y):
        self.opt = pFedMeOptimizer(self.net.parameters(), lr = self.perlr,lamda=self.lamda)
        # update w
        if t != self.old_task:
            self.old_task = t
        self.net.zero_grad()
        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        firstnet = deepcopy(self.net)
        for i in range(self.K):
            self.opt.zero_grad()
            self.update_parameters(firstnet.parameters())
            output = self.net.forward(x, t)[:, offset1: offset2]
            loss = self.ce(output, y - offset1)
            if loss.float() >10:
                print('da')
            loss.backward()
            self.persionalized_model_bar, _ = self.opt.step(self.local_model)
        for new_param, localweight in zip(self.persionalized_model_bar, self.local_model.parameters()):
            localweight.data = localweight.data - self.lamda* self.lr * (localweight.data - new_param.data)

        # update v

    def validTest(self, t, tr_dataloader, sbatch=20):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.net.eval()
        # Loop batches
        with torch.no_grad():
            for images, targets in tr_dataloader:
                images = images.cuda()
                targets = targets.cuda()

                # Forward
                offset1, offset2 = compute_offsets(t, self.nc_per_task,
                                                   self.is_cifar)
                output = self.net.forward(images, t)

                loss = self.ce(output[:, offset1: offset2], targets - offset1)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(targets)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(targets)

        return total_loss / total_num, total_acc / total_num
def life_experience(task,appr,tr_dataloader,epochs,sbatch=10):
    for name,para in appr.net.named_parameters():
        para.requires_grad = True
    appr.local_model = deepcopy(appr.net)
    for e in range(epochs):
        for i,(images,targets) in enumerate(tr_dataloader):
            images = images.cuda()
            targets = targets.cuda()
            appr.net.train()
            appr.observe(images, task, targets)
    appr.update_parameters(appr.local_model.parameters())
    loss,acc = appr.validTest(task,tr_dataloader)
    print('| Train finish, | Train: loss={:.3f}, acc={:5.1f}% | \n'.format( loss, 100 * acc), end='')
    return loss


def LongLifeTrain(args, appr, tr_dataloader, aggNum,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    t = aggNum // args.round
    r = aggNum % args.round
    # for t, ncla in taskcla:

    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    task = t

    # Train
    loss = life_experience(task,appr,tr_dataloader,args.local_ep,args.local_bs)
    print('-' * 100)
    return appr.net.state_dict(),loss,0


def LongLifeTest(args, appr, t, testdatas, aggNum, writer):
    acc = np.zeros((1, t), dtype=np.float32)
    lss = np.zeros((1, t), dtype=np.float32)
    t = aggNum // args.round
    r = aggNum % args.round
    for u in range(t + 1):
        xtest = testdatas[u][0].cuda()
        ytest = (testdatas[u][1]).cuda()
        test_loss, test_acc = appr.validTest(u, xtest, ytest)
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