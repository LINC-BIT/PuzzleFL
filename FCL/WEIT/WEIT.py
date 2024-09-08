import sys, time, os
from typing import OrderedDict

import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm

# from utils import *
from torch.utils.tensorboard import SummaryWriter
import quadprog
sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn


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
    def __init__(self, model, tr_dataloader, nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None):
        self.model = model
        self.model_old = model
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
        self.optimizer = self._get_optimizer(args.lr)
        self.lamb = args.lamb
        self.e_rep = args.local_rep_ep
        self.old_task=-1
        self.grad_dims = []
        self.pre_weight = {
            'weight':[],
            'aw':[],
            'mask':[]
        }
        return

    def set_sw(self,glob_weights):
        i = 0
        keys = [k for k, _ in self.model.named_parameters()]
        if len(glob_weights)>0:
            all_weights = []
            for name, para in self.model.named_parameters():
                if 'sw' in name:
                    all_weights.append(glob_weights[i])
                    i=i+1
                else:
                    all_weights.append(para)

            feature_dict = zip(keys, all_weights)
            # last_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(last_keys,last_para)})
            state_dict = OrderedDict({k: v for k, v in feature_dict})
            ##### sovle the lacking of running mean and running var keys
            new_state_dict = self.model.state_dict()
            new_state_dict.update(state_dict)
            #####
            self.model.load_state_dict(new_state_dict)
        print()

    def get_sw(self):
        sws = []
        for name, para in self.model.named_parameters():
            if 'sw' in name:
                sws.append(para)
        return sws

    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        if 'vit' in self.args.model or 'pit' in self.args.model:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)
        # self.momentum = 0.9
        # self.weight_decay = 0.0001
        #
        # optimizer =  torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum,
        #                       weight_decay=self.weight_decay)
        return optimizer

    def train(self, t,from_kbs,know):
        if t!=self.old_task:
            self.old_task=t
        
        # lr = self.lr
        for name, para in self.model.named_parameters():
            para.requires_grad = True
            
        self.model.set_knowledge(t,from_kbs)
        self.optimizer = self._get_optimizer(self.args.lr)
        self.model.to(self.args.device)
        
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            self.train_epoch(t)

            train_loss = 0
            train_acc = 0

        if len(self.pre_weight['aw'])<=t:
            self.pre_weight['aw'].append([])
            self.pre_weight['mask'].append([])
            self.pre_weight['weight'].append([])
            for name,para in self.model.named_parameters():
                if 'aw' in name:
                    aw = para.detach()
                    aw.requires_grad = False
                    self.pre_weight['aw'][-1].append(aw)
                elif 'mask' in name:
                    mask = para.detach()
                    mask.requires_grad = False
                    self.pre_weight['mask'][-1].append(mask)
            self.pre_weight['weight'][-1] = self.model.get_weights()
        else:
            self.pre_weight['aw'].pop()
            self.pre_weight['mask'].pop()
            self.pre_weight['weight'].pop()
            self.pre_weight['aw'].append([])
            self.pre_weight['mask'].append([])
            self.pre_weight['weight'].append([])
            for name, para in self.model.named_parameters():
                if 'aw' in name:
                    self.pre_weight['aw'][-1].append(para)
                elif 'mask' in name:
                    self.pre_weight['mask'][-1].append(para)
            self.pre_weight['weight'][-1] = self.model.get_weights()

        return self.get_sw(),train_loss, train_acc

    def train_epoch(self,t):
        self.model.train()
        for images,targets in self.tr_dataloader:
            images = images.to(self.args.device)

            # _num_per_class = self.args.num_classes // self.args.task
            _num_per_class = self.args.num_classes
            targets = (targets - _num_per_class * t).to(self.args.device)
            
            # Forward current model
            offset1, offset2 = compute_offsets(t, _num_per_class)
            self.optimizer.zero_grad()
            self.model.zero_grad()
            # store_grad(self.model.parameters,grads, self.grad_dims,0)
            outputs = self.model.forward(images,t)[:,offset1:offset2]
            loss = self.get_loss(outputs, targets,t)
            ## 根据这个损失计算梯度，变换此梯度

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def l2_loss(self,para):
        return torch.sum(torch.pow(para,2))/2

    def get_loss(self,outputs,targets,t):
        loss = self.ce(outputs,targets)
        i = 0
        weight_decay = 0
        sparseness = 0
        approx_loss = 0
        sw = None
        aw = None
        mask = None
        for name, para in self.model.named_parameters():
            # if t> 0:
            #     print()

            if 'sw' in name:
                sw = para
            elif 'aw' in name:
                aw = para
            elif 'mask' in name:
                mask = para
            # elif 'atten' in name:

            elif 'atten' in name:
                if ('t2t' in self.args.model and name.endswith('atten')) or 'attention' not in name:    # 这里需要跟vit中的attention层关键词进行区分
                    weight_decay += self.args.wd * self.l2_loss(aw)     # except
                    weight_decay += self.args.wd * self.l2_loss(mask)
                    sparseness += self.args.lambda_l1 * torch.sum(torch.abs(aw))
                    sparseness += self.args.lambda_mask * torch.sum(torch.abs(mask))

                    # if torch.isnan(weight_decay).sum() > 0:
                    #     print('weight_decay nan')
                    # if torch.isnan(sparseness).sum() > 0:
                    #     print('sparseness nan')

                    if t == 0:
                        weight_decay += self.args.wd * self.l2_loss(sw)
                    else:
                        for tid in range(t):
                            prev_aw = self.pre_weight['aw'][tid][i]
                            prev_mask = self.pre_weight['mask'][tid][i]
                            m = torch.nn.Sigmoid()
                            g_prev_mask = m(prev_mask)
                            #################################################
                            sw2 = sw.transpose(0,-1)
                            try:
                                sg = sw2 * g_prev_mask
                            except Exception:
                                print()
                                pass
                            sgt = sg.transpose(0,-1)
                            # (sw2 * g_prev_mask).transpose(0, -1)
                            restored = sgt + prev_aw
                            a_l2 = self.l2_loss(restored - self.pre_weight['weight'][tid][i])
                            approx_loss += self.args.lambda_l2 * a_l2
                            #################################################
                        i+=1
        loss+=weight_decay+sparseness+approx_loss
        return loss

    def eval(self, t,train=True):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        dataloaders = self.tr_dataloader

        # Loop batches
        with torch.no_grad():
            for images,targets in dataloaders:
                images = images.to(self.args.device)
                _num_per_class = self.args.num_classes // self.args.task
                targets = (targets - _num_per_class*t).to(self.args.device)
                # Forward
                offset1, offset2 = compute_offsets(t, _num_per_class)
                output = self.model.forward(images,t)[:,offset1:offset2]

                loss = self.ce(output, targets)
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
        return self.ce(output, targets) + self.lamb * loss_reg

    def set_kb(self, kb):
        self.kb = kb
        
    
    def get_kb(self):
        return self.kb

    def set_neibors(self, ids):
        self.neibors = ids
        
    def get_neibors(self):
        return self.neibors

def LongLifeTrain(args, appr, aggNum, from_kbs,idx):
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
    know = False
    if r == args.round - 1:
        know=True
    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    task = t

    # Train
    sws,loss,_ = appr.train(task,from_kbs,know)
    print('-' * 100)
    if know:
        return sws,appr.pre_weight['aw'][-1],loss,0
    else:
        return sws, None, loss, 0

def LongLifeTest(args, appr, t, testdatas, aggNum, writer):
    acc = np.zeros((1, t), dtype=np.float32)
    lss = np.zeros((1, t), dtype=np.float32)
    t = aggNum // args.round
    r = aggNum % args.round
    for u in range(t + 1):
        xtest = testdatas[u][0].to(args.device)
        ytest = (testdatas[u][1] - u * 10).to(args.device)
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
