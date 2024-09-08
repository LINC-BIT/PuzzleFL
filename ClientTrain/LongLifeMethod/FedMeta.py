import sys, time, os
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from utils import *
from torch.utils.tensorboard import SummaryWriter
import quadprog
sys.path.append('..')
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
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, tr_dataloader,nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None):
        self.model = model
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.inner_lr = lr
        self.lr_min = lr_min * 1 / 3
        self.q_grads = None
        self.ce = torch.nn.CrossEntropyLoss()
        self.old_task=-1
        self.inner_bat = 10
        return
    def set_model(self,model):
        self.model = model
    def set_fisher(self,fisher):
        self.fisher = fisher
    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader

    def train(self, t):
        if t!=self.old_task:
            self.old_task=t
        # Loop epochs
        train_accs = []
        self.q_grads=None
        for e in range(self.nepochs):
            if e % self.nepochs == self.nepochs -1:
                self.train_epoch(t,last=True)
            else:
                train_accs.append(self.train_epoch(t))
            if e % self.nepochs == self.nepochs -1:
                print('Inner update acc: ',end='')
                print(train_accs)
        # Fisher ops
        return 0, train_accs[-1]

    def train_epoch(self,t,last=False):
        self.model.train()
        self.opt = torch.optim.Adam(self.model.parameters(),self.inner_lr)
        for i,(images,targets) in enumerate(self.tr_dataloader):
            ## support update
            targets = (targets - 10 * t)
            if i == 0:
                bat = len(images)//self.inner_bat
                loss = 0
                for b in range(bat):
                    begin = b*self.inner_bat
                    end = (b+1)*self.inner_bat
                    if end > len(images):
                        end = len(images)
                    img = images[begin:end]
                    tag = targets[begin:end]
                    img = img.cuda()
                    tag = tag.cuda()
                    offset1, offset2 = compute_offsets(t, 10)
                    outputs = self.model.forward(img,t)[:,offset1:offset2]
                    loss += (end-begin)*self.ce(outputs, tag)

                # Backward
                loss /= len(images)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            else:
                bat = len(images) // self.inner_bat
                loss = 0
                total_acc = 0
                for b in range(bat):
                    begin = b*self.inner_bat
                    end = (b+1)*self.inner_bat
                    if end > len(images):
                        end = len(images)
                    img = images[begin:end]
                    tag = targets[begin:end]
                    img = img.cuda()
                    tag = tag.cuda()
                    if last:
                        # Forward current model
                        offset1, offset2 = compute_offsets(t, 10)
                        outputs = self.model.forward(img, t)[:, offset1:offset2]
                        loss += (end-begin)*self.ce(outputs, tag)
                        ## 根据这个损失计算梯度，变换此梯度
                        # Backward
                    else:
                        with torch.no_grad():
                            # Forward current model
                            offset1, offset2 = compute_offsets(t, 10)
                            outputs = self.model.forward(img, t)[:, offset1:offset2]
                            loss += (end-begin)*self.ce(outputs, tag)
                            _, pred = outputs.max(1)
                            hits = (pred == tag).float()
                            total_acc += hits.sum().data.cpu().numpy()
                if last:
                    loss /= len(images)
                    q_grads = torch.autograd.grad(loss, list(self.model.parameters()),
                                                       create_graph=True, retain_graph=True)
                    self.q_grads=[]
                    for i in q_grads:
                        self.q_grads.append(i.cpu().detach())
                    del q_grads
                return total_acc/len(images)


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
                images = images.cuda()
                targets = (targets - 10*t).cuda()
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



def LongLifeTrain(args, appr, aggNum, writer,idx):
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
    return appr.q_grads,loss,0

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

# def main():
#     # cifar100 = Cifar100Task('../data',batch_size=900,num_clients=5,cur_client=4,task_num=10,isFed=True)
#     cifar100 = Cifar100Task('../data/cifar-100-python', batch_size=4500, task_num=10, num_clients=5, cur_client=0,
#                       isFed=True)
#     TaskDatas = cifar100.getDatas()
#     net = network.RepTail([3, 32, 32]).cuda()
#
#
# if __name__ == "__main__":
#     main()