# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy, copy

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog
from torch.utils.data import DataLoader

def euclidean_distance(a, x):
    # 计算A中每一列与x的欧氏距离
    distances = torch.norm(a - x.view(1, -1), dim=1)
    return distances

def find_top_k_max_indices(tensor, k):
    # 找出张量中前k个最大值的索引
    _, indices = torch.topk(tensor, k)
    return indices

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
    try:
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1, 1))
    except ValueError:
        print('无法求解')
        gradient.copy_(torch.Tensor(gradient_np).view(-1, 1))


def _freeze_params(model, kw):
    """
    Zuo: freeze parameter gradients of the given model
    :param model:
    :param kw:  freeze body or head in a model
    """
    assert kw in ['body', 'head', 'all']
    if kw == 'all':
        for name, para in model.named_parameters():
            para.requires_grad = False
    else:
        for name, para in model.named_parameters():
            # if 'feature_net' in name and 'cls_token' not in name and 'pos_embed' not in name and 'patch_embed' not in name:
            if 'feature_net' in name:
                para.requires_grad = kw != 'body'
            else:
                para.requires_grad = kw == 'body'

class Appr():
    def __init__(self,
                 model,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args, cid=None, body_keys=None):
        super(Appr, self).__init__()
        self.args = args
        self.margin = args.memory_strength
        self.cid = cid
        self.is_cifar = True
        self.model = model
        self.body_keys = body_keys
        self.bef_agg_net = None  # 用来解决聚合带来的负迁移问题
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.opt = self._get_optimizer()

        self.args.n_memories = int(self.args.n_memories)
        self.n_memories = self.args.n_memories
        print('Number of memory is : ', self.n_memories)

        self.gpu = True

        self.mem_data_shape = n_inputs if isinstance(n_inputs, tuple) else [n_inputs]

        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        # fedvit
        self.unicls_memory_data = torch.FloatTensor(n_tasks, self.n_memories, *self.mem_data_shape)
        self.unicls_memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        self.unicls_memory_losses = torch.LongTensor(n_tasks, self.n_memories)
        self.unicls_memory_logits = torch.FloatTensor(n_tasks, self.n_memories, self.nc_per_task)
        if self.gpu:
            self.unicls_memory_data = self.unicls_memory_data.to(self.args.device)
            self.unicls_memory_labs = self.unicls_memory_labs.to(self.args.device)
            self.unicls_memory_losses = self.unicls_memory_losses.to(self.args.device)
            # self.unicls_memory_logits = self.unicls_memory_logits.to(self.args.device)

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())

        # for head limited
        self.curr_head = None
        self.old_head = None

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1

        self.tr_dataset = None
        self.tr_dataloader = None
        self.local_ep = args.local_ep
        self.device = args.device

        # rep
        self.e_rep = args.local_rep_ep

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.args.lr
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)
        return optimizer
    
    def set_model(self, model):
        self.model = model

    def set_trData(self, tr_dataset):
        # self.tr_dataset = tr_dataset
        # self.tr_dataloader = DataLoader(tr_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.tr_dataloader = tr_dataset

    def _metric(self, outputs, targets, loss):
        _, pred = outputs.max(1)
        hits = (pred == targets).float()
        corr = hits.sum().data.cpu().numpy()
        acc = corr / targets.size(0)
        avg_loss = loss.data.cpu().numpy()
        return acc, avg_loss

    def train(self, t, tot_round):
        self.model.train()

        # whether new task
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

            # add old model's distillation
            # self.model_old = deepcopy(self.model)
            self.old_head = deepcopy(self.curr_head)

        _local_ep = self.local_ep

        # 本地训练 local_ep 个 epochs
        for e in range(_local_ep):
            # fine-tune head
            if e < self.e_rep:
                _freeze_params(self.model, 'body')    # 冻结 body
                # if 'head_kd' in self.args.comment:
                #     self._train_head_kd(t)  # 使用旧模型的head指导新任务head
                # else:
                self._train_epoch(t, replay=False)  # 默认训练头部的方法

            # train body
            else:
                _freeze_params(self.model, 'head')    # 冻结 head
                self._train_epoch(t, replay=True)   # body 部分采用 replay
            # self._train_epoch(t)
        self._loss_based_by_cls(t)  # save samples as current task

        self.bef_agg_net = deepcopy(self.model)       # for mgnt
        self._record_head(t)        # recored the head params


    def _record_head(self, t):
        w_head = {}
        w_local = self.model.state_dict()
        for k in w_local.keys():
            # if k not in self.args.w_glob_keys:
            if 'last' in k:
                w_head[k] = deepcopy(w_local[k])
        print(f"Recored the current ({t}) task's head params.")
        self.curr_head = w_head

    def _train_epoch(self, t, replay=False):
        """ 优化了显存占用问题 """
        self.opt = self._get_optimizer()            ## TODO: 优化器位置
        for images, targets in self.tr_dataloader:
            # self.opt = self._get_optimizer()            ## TODO: 优化器位置

            images = images.to(self.device)
            targets = targets.to(self.device)
            _mem_data = self.unicls_memory_data
            _mem_labs = self.unicls_memory_labs

            # build a graidnets space
            if len(self.observed_tasks) > 1 and replay:
                grads = torch.Tensor(sum(self.grad_dims), t+1).to(self.args.device)
                # print('grad shape: {}'.format(grads.shape))
            else:
                grads = None

            # signitificant past task
            K = 2 
            if len(self.observed_tasks) > K+1:
                old_tids = self.observed_tasks[-(K+1):-1]
            else:
                old_tids = self.observed_tasks[:-1]
                    
            # compute gradient on previous tasks
            if len(self.observed_tasks) > 1 and replay:
                # for tt in range(len(self.observed_tasks) - 1):
                for tt in old_tids:
                    self.model.zero_grad()
                    # fwd/bwd on the examples in the memory
                    past_task = self.observed_tasks[tt]
                    offset1, offset2 = compute_offsets(past_task, self.nc_per_task, self.is_cifar)

                    output = self.model(_mem_data[past_task], past_task)[:, offset1: offset2]

                    # c = _mem_labs[past_task]
                    ptloss = self.ce(output, _mem_labs[past_task] - offset1)

                    ptloss.backward()
                    store_grad(self.model.parameters, grads, self.grad_dims, past_task)

            # now compute the grad on the current mini_batch
            self.model.zero_grad()
            offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            # print('offset1: {}'.format(offset1))

            outputs = self.model.forward(images, t)[:, offset1: offset2]
            # print('output {}'.format(outputs))

            targets = targets - offset1
            # print(' label {}'.format(targets))

            loss = self.ce(outputs, targets)  # 当前任务loss
            loss.backward()

            # 梯度集成: 当前任务梯度与过去所有任务的梯度进行集成
            if len(self.observed_tasks) > 1 and replay:
                store_grad(self.model.parameters, grads, self.grad_dims, t)
                indx = torch.LongTensor(self.observed_tasks[:-1])
                indx = indx.to(self.args.device)
                a = grads[:, t].unsqueeze(0)
                b = grads.index_select(1, indx)
                dotp = torch.mm(a, b)
                if (dotp < 0).sum() != 0:
                    # project2cone2(grads[:, t].unsqueeze(1), grads.index_select(1, indx), self.margin)
                    project2cone2(grads[:, t].unsqueeze(1), b, self.margin)
                    overwrite_grad(self.model.parameters, grads[:, t], self.grad_dims)

            # 更新模型
            self.opt.step()

            # freeze grads
            del grads
            torch.cuda.empty_cache()

    # def _train_head_kd(self, t):
        # """ step1. 旧任务的head+当前任务的body
        #     ste2. 训练当前head的同时不丢失旧任务head的知识
        # """
        # if t > 0 and self.old_head is not None:
        #     old_model = deepcopy(self.model)
        #     net_glob = old_model.state_dict()
        #     for k in net_glob.keys():
        #         if k in self.old_head.keys():
        #             net_glob[k].data = self.old_head[k].data
        #     old_model.load_state_dict(net_glob)
        #     old_model.train()
        #     _freeze_params(old_model, 'all')
        #     print('Replace the body of old model with global model')

        # for images, targets in self.tr_dataloader:
        #     self.opt = self._get_optimizer()

        #     images = images.to(self.device)
        #     targets = targets.to(self.device)

        #     # now compute the grad on the current mini_batch
        #     self.model.zero_grad()
        #     offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        #     outputs = self.model.forward(images, t)
        #     # curr_logits = outputs[1]
        #     # outputs = outputs[0][:, offset1: offset2]
        #     outputs = outputs[:, offset1: offset2]

        #     # outputs = self.model.forward(images, t)[:, offset1, offset2]   # 20 x 100
        #     targets = targets - offset1
        #     loss = self.ce(outputs, targets)  # 当前任务loss

        #     # compute gradient on previous tasks
        #     if t > 0 and self.old_head is not None:
        #         old_logits = old_model.forward(images, t, dist=True)[0][:, offset1: offset2]
        #         # old_logits = old_model.forward(images, t, dist=True)[1]
        #         # old_logits = old_model.forward(images, t, dist=True)[:, offset1: offset2]
        #         from torch.nn import functional as F
        #         loss_kd = F.kl_div(
        #             F.log_softmax(outputs, dim=1),
        #             F.log_softmax(old_logits, dim=1),
        #             reduction='batchmean',
        #             log_target=True
        #         )
        #     else:
        #         loss_kd = 0.
        #     loss += 10000 * loss_kd

        #     loss.backward()

        #     # 更新模型
        #     self.opt.step()

    def _loss_based_by_cls(self, task):
        print(f'---- saving samples for task {task}----')
        """ 统计当前任务中每个类别的样本数目比例，构建等比例的类别样本memory，每个类种的样本 loss 按从小到大取topk个。"""
        _per_ce = nn.CrossEntropyLoss(reduction='none')
        sample_losses = torch.tensor([]).to(self.device)
        samples = None
        labels = None
        _offset1 = None
        # logits = None # for DK
        self.model.eval()
        with torch.no_grad():
            for b_idx, (images, targets) in enumerate(self.tr_dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                offset1, offset2 = compute_offsets(task, self.nc_per_task, self.is_cifar)
                _offset1 = offset1
                outputs = self.model.forward(images, task)[:, offset1: offset2]
                targets = targets - offset1
                losses = _per_ce(outputs, targets)  # losses of each sample
                sample_losses = torch.concat((sample_losses, losses))

                # 保存一份带有新索引的samples
                if samples is None:
                    samples = images.detach()
                    labels = targets.detach()
                    # logits = outputs.detach()

                else:
                    samples = torch.concat((samples, images))
                    labels = torch.concat((labels, targets))
                    # logits = torch.concat((logits, outputs))

        # 获得每个类别的存储样本数
        num_samp_per_cls = torch.bincount(labels).long()
        # print(f'Labels and their count in this task is : {num_samp_per_cls}')
        tot_num = labels.size(0)

        save_ration = self.args.n_memories / tot_num

        num_samp_per_cls = (num_samp_per_cls * save_ration)
        # print(f'num_samp_per_cls * save_ration: {num_samp_per_cls}')
        # num_samp_per_cls = num_samp_per_cls.ceil()    # ceil 会导致缺少最后一个类别的数据
        num_samp_per_cls = torch.round(num_samp_per_cls)
        # print(f'num_samp_per_cls after round is: {num_samp_per_cls}')

        data_per_cls = {}
        label_per_cls = {}
        loss_per_cls = {}
        logits_per_cls = {}
        _cls_num = num_samp_per_cls.size(0)
        for i in range(_cls_num):
            cls_indices = torch.nonzero((labels == i).float()).squeeze()
            _samples = samples.index_select(dim=0, index=cls_indices)
            _labels = labels.index_select(dim=0, index=cls_indices)
            _sample_loss = sample_losses.index_select(dim=0, index=cls_indices)
            # _logits = logits.index_select(dim=0, index=cls_indices)
            if i not in label_per_cls.keys():
                data_per_cls[i] = _samples
                label_per_cls[i] = _labels
                loss_per_cls[i] = _sample_loss
                # logits_per_cls[i] = _logits

            else:
                data_per_cls[i] = torch.concat((data_per_cls[i], _samples))
                label_per_cls[i] = torch.concat((label_per_cls[i], _labels))
                loss_per_cls[i] = torch.concat((loss_per_cls[i], _sample_loss))
                # logits_per_cls[i] = torch.concat((logits_per_cls[i], _logits))

        # 获取loss值最小的前200的样本索引
        oval_samples = None
        oval_labels = None
        oval_losses = None
        oval_logits = None
        for i in range(_cls_num):
            # if 'max' in self.args.comment:      # 取 top 最大loss
            #     r_sample_losses = loss_per_cls[i]
            # else:
            r_sample_losses = loss_per_cls[i] * -1
            
            if num_samp_per_cls[i] >= r_sample_losses.size(0):
                k = r_sample_losses.size(0)
            else:
                k = int(num_samp_per_cls[i].item())
            _, indices = r_sample_losses.topk(k)
            _data = data_per_cls[i].index_select(dim=0, index=indices)
            _labs = label_per_cls[i].index_select(dim=0, index=indices)
            _loss = loss_per_cls[i].index_select(dim=0, index=indices)

            if oval_labels is None:
                oval_samples = _data
                oval_labels = _labs
                oval_losses = _loss
                # oval_logits = _logits

            else:
                oval_samples = torch.concat((oval_samples, _data))
                oval_labels = torch.concat((oval_labels, _labs))
                oval_losses = torch.concat((oval_losses, _loss))
                # oval_logits = torch.concat((oval_logits, _logits))

            # 如果存储的样本量不够。
            if i == _cls_num-1 and oval_labels.size(0) < self.args.n_memories:
                left = self.args.n_memories - oval_labels.size(0)
                indices = torch.tensor([i for i in range(left)]).to(self.args.device)
                _data = data_per_cls[i].index_select(dim=0, index=indices)
                _labs = label_per_cls[i].index_select(dim=0, index=indices)
                _loss = loss_per_cls[i].index_select(dim=0, index=indices)
                oval_samples = torch.concat((oval_samples, _data))
                oval_labels = torch.concat((oval_labels, _labs))
                oval_losses = torch.concat((oval_losses, _loss))
                # oval_logits = torch.concat((oval_logits, _logits))

        # 保存loss值前10%样本
        oval_labels = oval_labels + _offset1
        self.unicls_memory_data[task, :self.args.n_memories].copy_(oval_samples[:self.args.n_memories].view(self.args.n_memories, *self.mem_data_shape))
        self.unicls_memory_labs[task, :self.args.n_memories].copy_(oval_labels[:self.args.n_memories])
        self.unicls_memory_losses[task, :self.args.n_memories].copy_(oval_losses[:self.args.n_memories])
        # self.unicls_memory_logits[task, :self.args.n_memories].copy_(oval_logits[:self.args.n_memories, :])

   
def LongLifeTrain(args, appr, aggNum, idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))

    t = aggNum // args.round
    # Get data
    task = t

    # Train
    appr.train(task, aggNum)
    print('-' * 100)
    return appr.model.state_dict(), None, 0


if __name__ == '__main__':
    layer = torch.nn.Linear(2, 3)
    g_dims = []
    for p in layer.parameters():
        g_dims.append(p.data.numel())
    print(g_dims)

    # store_grad(layer.parameters, )
    a = torch.FloatTensor((3, 3)).to('cuda:0')

    print(torch.cuda.memory_allocated())

    del a
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated())
