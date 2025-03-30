
import sys, time, os
import numpy as np
import torch
from copy import deepcopy

from utils import *

sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn


from ClientTrain.AggModel.sixcnn import GsSixCNN as Net

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

def get_model(model):
    return deepcopy(model.state_dict())

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def gs_cal(t, tr_dataloader, criterion, model, sbatch=20):
    # Init
    param_R = {}

    for name, param in model.named_parameters():
        if len(param.size()) <= 1:
            continue
        name = name.split('.')[:-1]
        name = '.'.join(name)
        param = param.view(param.size(0), -1)
        param_R['{}'.format(name)] = torch.zeros((param.size(0))).cuda()

    # Compute
    model.train()


    for images,target in tr_dataloader:

        images = images.cuda()
        offset1, offset2 = compute_offsets(t, 10)
        # Forward and backward
        outputs = model.forward(images,t, True)[:,offset1:offset2]
        cnt = 0

        for idx, j in enumerate(model.act):
            j = torch.mean(j, dim=0)
            if len(j.size()) > 1:
                j = torch.mean(j.view(j.size(0), -1), dim=1).abs()
            model.act[idx] = j

        for name, param in model.named_parameters():
            if len(param.size()) <= 1 or 'last' in name or 'downsample' in name:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            param_R[name] += model.act[cnt].abs().detach() * len(target)
            cnt += 1

    with torch.no_grad():
        for key in param_R.keys():
            param_R[key] = (param_R[key] / len(tr_dataloader.dataset))
    return param_R

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
class Appr(object):

    def __init__(self, model, tr_dataloader, nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5,
                 clipgrad=100, args=None, kd_model=None):
        self.model = model
        self.model_old = model
        self.omega = None
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        self.args=args
        self.freeze = {}
        self.mask = {}
        self.rho = 0.3
        self.eta = 0.9
        self.lamb = 400
        self.initail_mu = 20
        self.mu = 0.1
        self.old_task = -1
        self.kd_epoch = 0
        self.init_kdmodels(kd_model)

        for (name, p) in self.model.named_parameters():
            if len(p.size()) < 2:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            self.mask[name] = torch.zeros(p.shape[0])

        # if len(args.parameter) >= 1:
        #     params = args.parameter.split(',')
        #     print('Setting parameters to', params)
        #     self.lamb = float(params[0])

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def set_model(self,model):
        self.model = model

    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader

    def init_kdmodels(self,kd_model):
        self.kd_models=[]
        for t in range(self.args.task):
            self.kd_models.append(deepcopy(kd_model))

    def train_kd(self,t):
        self.cur_kd.cuda()
        self.cur_kd.train()
        kd_optimizer = torch.optim.Adam(self.cur_kd.parameters(), lr=0.0005)
        l,a = self.eval(t,model = self.cur_kd)
        print('first kd:', a)
        for e in range(self.kd_epoch):
            for images,targets in self.tr_dataloader:
                images = images.cuda()
                targets = (targets - 10 * t).cuda()
                # Forward current model
                offset1, offset2 = compute_offsets(t, 10)
                outputs = self.cur_kd.forward(images, t)[:, offset1:offset2]
                loss = self.ce(outputs, targets)
                kd_optimizer.zero_grad()
                loss.backward()
                kd_optimizer.step()
        l,a = self.eval(t, model = self.cur_kd)
        print('last kd:', a)
        return a

    def train(self, t, input_size, last=False):

        if t!=self.old_task:
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            freeze_model(self.model_old)  # Freeze the weights
            self.old_task=t
            self.cur_kd = self.kd_models[t]
            self.first_train = True
        kd_lambda = self.train_kd(t)
        kd_lambda = 0
        self.cur_kd.eval()


        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        if t > 0:
            self.freeze = {}
            for name, param in self.model.named_parameters():
                if 'bias' in name or 'last' in name:
                    continue
                key = name.split('.')[0]
                if 'conv1' not in name:
                    if 'conv' in name:  # convolution layer
                        temp = torch.ones_like(param)
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp
                    else:  # linear layer
                        temp = torch.ones_like(param)
                        temp = temp.reshape((temp.size(0), self.omega[prekey].size(0), -1))
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp.reshape(param.shape)
                prekey = key

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()

            # CUB 200 xtrain_cropped = crop(x_train)
            self.model.train()
            self.train_epoch(t, kd_lambda)

            train_loss, train_acc = self.eval(t)
            if e % 5 == 4:
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                    e + 1,  train_loss, 100 * train_acc), end='')



        # Update old
        if last:
            self.model.act = None

            temp = gs_cal(t, self.tr_dataloader, self.criterion, self.model)
            for n in temp.keys():
                if t > 0:
                    self.omega[n] = self.eta * self.omega[n] + temp[n]
                else:
                    self.omega = temp
                self.mask[n] = (self.omega[n] > 0).float()


            test_loss, test_acc = self.eval(t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(test_loss, 100 * test_acc))

            dummy = Net(input_size).cuda()

            pre_name = 0

            for (name, dummy_layer), (_, layer) in zip(dummy.named_children(), self.model.named_children()):
                with torch.no_grad():
                    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                        if pre_name != 0:
                            temp = (self.omega[pre_name] > 0).float()
                            if isinstance(layer, nn.Linear) and 'conv' in pre_name:
                                temp = temp.unsqueeze(0).unsqueeze(-1)
                                weight = layer.weight
                                weight = weight.view(weight.size(0), temp.size(1), -1)
                                weight = weight * temp
                                layer.weight.data = weight.view(weight.size(0), -1)
                            elif len(weight.size()) > 2:
                                temp = temp.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                                layer.weight *= temp
                            else:
                                temp = temp.unsqueeze(0)
                                layer.weight *= temp

                        weight = layer.weight.data
                        bias = layer.bias.data

                        if len(weight.size()) > 2:
                            norm = weight.norm(2, dim=(1, 2, 3))
                            mask = (self.omega[name] == 0).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                        else:
                            norm = weight.norm(2, dim=(1))
                            mask = (self.omega[name] == 0).float().unsqueeze(-1)

                        zero_cnt = int((mask.sum()).item())
                        indice = np.random.choice(range(zero_cnt), int(zero_cnt * (1 - self.rho)), replace=False)
                        indice = torch.tensor(indice).long()
                        idx = torch.arange(weight.shape[0])[mask.flatten(0) == 1][indice]
                        mask[idx] = 0

                        layer.weight.data = (1 - mask) * layer.weight.data + mask * dummy_layer.weight.data
                        mask = mask.squeeze()
                        layer.bias.data = (1 - mask) * bias + mask * dummy_layer.bias.data

                        pre_name = name

                    if isinstance(layer, nn.ModuleList):

                        weight = layer[t].weight

                        weight[:, self.omega[pre_name] == 0] = 0
            test_loss, test_acc = self.eval(t)

            # self.model_old = deepcopy(self.model)
            # self.model_old.train()
            # freeze_model(self.model_old)  # Freeze the weights
        return train_loss,train_acc

    def train_epoch(self, t, kd_lambda):
        self.model.train()

        # Loop batches
        for images, targets in self.tr_dataloader:

            images = images.cuda()
            targets = (targets - 10 * t).cuda()

            # Forward current model
            offset1, offset2 = compute_offsets(t, 10)
            outputs = self.model.forward(images,t)[:,offset1:offset2]
            kd_outpus = self.cur_kd.forward(images,t)[:,offset1:offset2]
            loss = self.criterion(t, outputs, targets) + kd_lambda * MultiClassCrossEntropy(outputs,kd_outpus,t,T=2)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Freeze the outgoing weights
            if t > 0:
                for name, param in self.model.named_parameters():
                    if 'bias' in name or 'last' in name or 'conv1' in name:
                        continue
                    key = name.split('.')[0]
                    param.data = param.data * self.freeze[key]

        self.proxy_grad_descent(t)

        return

    def eval(self, t, train=True, model=None, dataloaders=None):
        total_loss = 0
        total_acc = 0
        total_num = 0
        if train:
            dataloaders = self.tr_dataloader
        if model is None:
            model = self.model
        # Loop batches
        model.eval()
        with torch.no_grad():
            for images, targets in dataloaders:
                images = images.cuda()
                targets = (targets - 10 * t).cuda()
                # Forward
                offset1, offset2 = compute_offsets(t, 10)
                output = model.forward(images, t)[:, offset1:offset2]

                loss = self.criterion(t, output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

    def proxy_grad_descent(self, t):
        with torch.no_grad():
            for (name, module), (_, module_old) in zip(self.model.named_children(), self.model_old.named_children()):
                if not isinstance(module, torch.nn.Linear) and not isinstance(module, torch.nn.Conv2d):
                    continue

                mu = self.mu

                key = name
                weight = module.weight
                bias = module.bias
                weight_old = module_old.weight
                bias_old = module_old.bias

                if len(weight.size()) > 2:
                    norm = weight.norm(2, dim=(1, 2, 3))
                else:
                    norm = weight.norm(2, dim=(1))
                norm = (norm ** 2 + bias ** 2).pow(1 / 2)

                aux = F.threshold(norm - mu * self.lr, 0, 0, False)
                alpha = aux / (aux + mu * self.lr)
                coeff = alpha * (1 - (self.mask[key].cuda()))

                if len(weight.size()) > 2:
                    sparse_weight = weight.data * coeff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    sparse_weight = weight.data * coeff.unsqueeze(-1)
                sparse_bias = bias.data * coeff

                penalty_weight = 0
                penalty_bias = 0

                if t > 0:
                    if len(weight.size()) > 2:
                        norm = (weight - weight_old).norm(2, dim=(1, 2, 3))
                    else:
                        norm = (weight - weight_old).norm(2, dim=(1))

                    norm = (norm ** 2 + (bias - bias_old) ** 2).pow(1 / 2)

                    aux = F.threshold(norm - self.omega[key] * self.lamb * self.lr, 0, 0, False)
                    boonmo = self.lr * self.lamb * self.omega[key] + aux
                    alpha = (aux / boonmo)
                    alpha[alpha != alpha] = 1

                    coeff_alpha = alpha * self.mask[key]
                    coeff_beta = (1 - alpha) * self.mask[key]

                    if len(weight.size()) > 2:
                        penalty_weight = coeff_alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * weight.data + \
                                         coeff_beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * weight_old.data
                    else:
                        penalty_weight = coeff_alpha.unsqueeze(-1) * weight.data + coeff_beta.unsqueeze(
                            -1) * weight_old.data
                    penalty_bias = coeff_alpha * bias.data + coeff_beta * bias_old.data

                diff_weight = (sparse_weight + penalty_weight) - weight.data
                diff_bias = sparse_bias + penalty_bias - bias.data

                weight.data = sparse_weight + penalty_weight
                bias.data = sparse_bias + penalty_bias

        return

    def criterion(self, t, output, targets):
        return self.ce(output, targets)

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
    last=False
    if r == args.round - 1:
        last=True
    # for t, ncla in taskcla:

    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    task = t

    # Train
    loss,_ = appr.train(task,[3,32,32],last=last)
    print('-' * 100)
    from random import sample
    samplenum = 4
    if samplenum < t:
        samplenum = t
    kd_state_dicts = sample([kd_model.state_dict() for i, kd_model in enumerate(appr.kd_models) if i < samplenum], 4)
    kd_state_dicts.append(appr.kd_models[t].state_dict())

    return kd_state_dicts, loss, 0

from ClientTrain.dataset.Cifar100 import Cifar100Task
from ClientTrain.AggModel.sixcnn import GsSixCNN
from torch.utils.data import DataLoader
from Agg.AggModel.sixcnn import SixCNN as KDmodel
from ClientTrain.utils.options import args_parser
from ClientTrain.utils.train_utils import get_data
from ClientTrain.models.Update import LocalUpdate,DatasetSplit
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
# task = Cifar100Task('../../data/cifar100',task_num=10)
# trains, tests = task.getTaskDataSet()
dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
net_glob = GsSixCNN([3,32,32],outputsize=100)
kd_model = KDmodel([3,32,32],100)
tl=[]
appr = Appr(net_glob.cuda(), None, lr=0.001, nepochs=50, args=args, kd_model=kd_model)
for idx in dict_users_train.keys():
    np.random.shuffle(dict_users_train[idx])
client_task = [[j for j in range(args.task)] for i in range(args.num_users)]
for task in range(10):
    train_loader= DataLoader(DatasetSplit(dataset_train[client_task[0][task]],dict_users_train[0][:args.m_ft],tran_task=[task,client_task[0][task]]),batch_size=args.local_bs, shuffle=True)




    appr.set_trData(train_loader)
    appr.train(task,[3,32,32],True)
    test_loader = DataLoader(
        DatasetSplit(dataset_test[client_task[0][task]], dict_users_test[0], tran_task=[task, client_task[0][task]]),
        batch_size=args.local_bs, shuffle=True)
    _,acc1 = appr.eval(task,train=False,dataloaders=test_loader)
    _, acc2 = appr.eval(task, train=True)
    print(acc1,acc2)
    tl.append(test_loader)

    for i,ttl in enumerate(tl):
        loss,acc = appr.eval(i,train=False,dataloaders=ttl)
        print('task'+str(i)+': '+str(acc))