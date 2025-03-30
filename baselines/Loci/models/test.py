# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/test.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ClientTrain.config import cfg
from ClientTrain.models.ChannelGatemodel.model import ChannelGatedCL
from ClientTrain.utils.ChannelGateutils import perform_task_incremental_test
import time
# from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
from copy import deepcopy
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs,tran_task=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.tran_task=tran_task
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        if self.tran_task is not None:
            label = label - 10 * self.tran_task[1] + 10 * self.tran_task[0]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label
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
def test_img_local(net_g, dataset, args,t, idxs=None,num_classes=10,glob_classify =None, tran_task=None, appr=None, spe=False):
    net_g.to(args.device)
    net_g.eval()
    if glob_classify is not None:
        glob_classify.to(args.device)
        glob_classify.eval()
    test_loss = 0
    correct = 0

    data_loader = DataLoader(DatasetSplit(dataset,idxs,tran_task=tran_task), batch_size=args.local_test_bs,shuffle=False)
    count = 0
    for idx, (data, target) in enumerate(data_loader):
        offset1, offset2 = compute_offsets(t, num_classes)
        data = data.to(args.device)
        target = (target - num_classes * t).to(args.device)

        data, target = data.to(args.device), target.to(args.device)
        # if appr is not None:
        #     appr.pernet.to(args.device)
        #     output1 = appr.pernet(data,t)[:, offset1:offset2]
        #     output2 = net_g(data,t)[:, offset1:offset2]
        #     log_probs = appr.alpha * output1 + (1-appr.alpha)*output2
        # else:
        #     if glob_classify is None:
        #         log_probs = net_g(data,t)[:, offset1:offset2]
        #     else:
        #         features = net_g(data, return_feat=False)
        #         log_probs = glob_classify.forward(features, t)[:, offset1:offset2]
        # sum up batch loss
        if spe:
            if len(appr.pack.masks) <= t:
                continue
            appr.pack.apply_eval_mask(task_idx=t, model=net_g.feature_net)

        log_probs = net_g(data,t)[:,offset1:offset2]
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    count = len(data_loader.dataset)
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return  accuracy, test_loss


def test_img_local_channel(net_g, dataset, args,t, idxs=None,num_classes=10,glob_classify =None, tran_task=None, appr=None, spe=False):
    net_g.to(args.device)
    net_g.eval()
    if glob_classify is not None:
        glob_classify.to(args.device)
        glob_classify.eval()
    test_loss = 0
    correct = 0

    data_loader = DataLoader(DatasetSplit(dataset,idxs,tran_task=tran_task), batch_size=args.local_test_bs,shuffle=False)
    count = 0
    for idx, (data, target) in enumerate(data_loader):
        offset1, offset2 = compute_offsets(t, num_classes)
        data = data.to(args.device)
        target = (target - num_classes * t).to(args.device)

        data, target = data.to(args.device), target.to(args.device)
        # if appr is not None:
        #     appr.pernet.to(args.device)
        #     output1 = appr.pernet(data,t)[:, offset1:offset2]
        #     output2 = net_g(data,t)[:, offset1:offset2]
        #     log_probs = appr.alpha * output1 + (1-appr.alpha)*output2
        # else:
        #     if glob_classify is None:
        #         log_probs = net_g(data,t)[:, offset1:offset2]
        #     else:
        #         features = net_g(data, return_feat=False)
        #         log_probs = glob_classify.forward(features, t)[:, offset1:offset2]
        # sum up batch loss



        head_idx = torch.full_like(target,t).to(args.device)
        out = net_g(data, head_idx, task_supervised_eval=cfg.TASK_SUPERVISED_VALIDATION)

        test_loss += F.cross_entropy(out, target, reduction='sum').item()
        y_pred = out.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    count = len(data_loader.dataset)
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return  accuracy, test_loss


def test_img_local_all(net, args, dataset_test, dict_users_test,t,w_locals=None,return_all=False,write =None,apprs = None,num_classes = 10,glob_classify=None,round=None, client_task=None,spe=False):
    print('test begin'+'*'*100)
    print('task '+str(t)+' finish train')
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        all_task_acc = 0
        all_task_loss = 0
        for u in range(t + 1):
            if apprs is not None:
                appr = apprs[idx]
            else:
                appr = None
            a, b = test_img_local(appr.model, dataset_test[client_task[idx][u]], args,u,idxs=dict_users_test[idx],num_classes=num_classes,glob_classify = glob_classify, tran_task=[u,client_task[idx][u]] ,appr=appr,spe=spe)
            all_task_acc += a
            all_task_loss += b
        all_task_acc /= t+1
        all_task_loss /= t+1


        acc_test_local[idx] = all_task_acc
        loss_test_local[idx] = all_task_loss
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local)/num_idxxs, round)
    if return_all:
        return acc_test_local, loss_test_local

    return sum(acc_test_local)/num_idxxs, sum(loss_test_local)/num_idxxs


def test_img_local_multi_all(multi_model_dict, args, dataset_test, dict_users_test,t,return_all=False,write =None,apprs = None,num_classes = 10,glob_classify=None,round=None, client_task=None,spe=False):
    print('test begin'+'*'*100)
    print('task '+str(t)+' finish train')
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        all_task_acc = 0
        all_task_loss = 0
        for u in range(t + 1):
            if apprs is not None:
                appr = apprs[idx]
            else:
                appr = None
            if multi_model_dict[idx]['cl_method'] == 'Packnet':
                a, b = test_img_local(appr.model, dataset_test[client_task[idx][u]], args,u,idxs=dict_users_test[idx],num_classes=num_classes,glob_classify = glob_classify, tran_task=[u,client_task[idx][u]] ,appr=appr,spe=True)
            elif multi_model_dict[idx]['cl_method'] == 'ChannelGate':
                a, b = test_img_local_channel(appr.model, dataset_test[client_task[idx][u]], args, u,
                                              idxs=dict_users_test[idx],
                                              num_classes=num_classes, glob_classify=glob_classify,
                                              tran_task=[u, client_task[idx][u]], appr=appr, spe=False)
            else:
                a, b = test_img_local(appr.model, dataset_test[client_task[idx][u]], args, u, idxs=dict_users_test[idx],
                                      num_classes=num_classes, glob_classify=glob_classify,
                                      tran_task=[u, client_task[idx][u]], appr=appr, spe=False)
            all_task_acc += a
            all_task_loss += b
        all_task_acc /= t+1
        all_task_loss /= t+1


        acc_test_local[idx] = all_task_acc
        loss_test_local[idx] = all_task_loss
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local)/num_idxxs, round)
    if return_all:
        return acc_test_local, loss_test_local

    return sum(acc_test_local)/num_idxxs, sum(loss_test_local)/num_idxxs


def test_img_local_all_channel(net, args, dataset_test, dict_users_test,t,w_locals=None,return_all=False,write =None,apprs = None,num_classes = 10,glob_classify=None,round=None, client_task=None,spe=False):
    print('test begin' + '*' * 100)
    print('task ' + str(t) + ' finish train')
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        all_task_acc = 0
        all_task_loss = 0
        for u in range(t + 1):
            if apprs is not None:
                appr = apprs[idx]
            else:
                appr = None
            a, b = test_img_local_channel(appr.model, dataset_test[client_task[idx][u]], args, u, idxs=dict_users_test[idx],
                                  num_classes=num_classes, glob_classify=glob_classify,
                                  tran_task=[u, client_task[idx][u]], appr=appr, spe=spe)
            all_task_acc += a
            all_task_loss += b
        all_task_acc /= t + 1
        all_task_loss /= t + 1

        acc_test_local[idx] = all_task_acc
        loss_test_local[idx] = all_task_loss
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local) / num_idxxs, round)
    if return_all:
        return acc_test_local, loss_test_local

    return sum(acc_test_local) / num_idxxs, sum(loss_test_local) / num_idxxs

def test_img_local_all_WEIT(appr, args, dataset_test, dict_users_test, t, w_locals=None, w_glob_keys=None, indd=None,
                       dataset_train=None, dict_users_train=None, return_all=False, write=None,num_classes = 10):
    print('test begin' + '*' * 100)
    print('task ' + str(t) + ' finish train')
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(appr[idx].model)
        net_local.eval()
        all_task_acc = 0
        all_task_loss = 0
        for u in range(t + 1):

            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                a, b = test_img_local(net_local, dataset_test, args, t, idx=dict_users_test[idx], indd=indd,
                                      user_idx=idx)
                # tot += len(dataset_test[dict_users_test[idx]]['x'])
            else:
                a, b = test_img_local(net_local, dataset_test[u], args, u, user_idx=idx, idxs=dict_users_test[idx],num_classes=num_classes)
                all_task_acc += a
                all_task_loss += b
        all_task_acc /= (t + 1)
        all_task_loss /= (t + 1)
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            tot += len(dataset_test[dict_users_test[idx]]['x'])
        else:
            tot += len(dict_users_test[idx])
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = all_task_acc * len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = all_task_acc * len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local[idx] = all_task_acc * len(dict_users_test[idx])
            loss_test_local[idx] = all_task_loss * len(dict_users_test[idx])
        del net_local

    if return_all:
        return acc_test_local, loss_test_local
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local) / tot, t + 1)
    return sum(acc_test_local) / tot, sum(loss_test_local) / tot
