r"""Train routines.

This module contains training infrastructure for Conditional Channel Gated Networks.
"""


from tqdm.auto import tqdm

from ClientTrain.config import cfg
from ClientTrain.utils import ChannelGateexperiment as experiment
from ClientTrain.models.ChannelGatemodel.model import freeze_relevant_kernels, ChannelGatedCL
from ClientTrain.dataset.Cifar100 import Cifar100Task,setTask
from ClientTrain.utils.ChannelGateutils import train_one_task, perform_task_incremental_test




import sys,os

import torch
from copy import deepcopy
from tqdm import tqdm

sys.path.append('..')


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
                 args=None, kd_model=None,client=0):
        self.model = model
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
        self.optimizer = self._get_optimizer()
        self.lamb = args.lamb
        self.e_rep = args.local_rep_ep
        self.old_task=-1
        self.grad_dims = []
        self.kd_epoch=10
        self.client = client
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.first_train = True
        return

    def set_model(self,model):
        self.model = model
    def set_fisher(self,fisher):
        self.fisher = fisher
    def set_trData(self,tr_dataloader,te=None):
        self.tr_dataloader = tr_dataloader
    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.momentum = 0.9
        # self.weight_decay = 0.0001
        #
        # optimizer =  torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum,
        #                       weight_decay=self.weight_decay)
        return optimizer
    def train(self, t):
        if t!=self.old_task:
            self.first_train = True
            self.model.add_task()
            if self.old_task >= 0:
                freeze_relevant_kernels(self.model, self.tr_dataloader,
                                    task_identifier=self.old_task,
                                    save_freqs=True, verbose=False,
                                    save_fname=None,t = self.old_task)
            self.old_task = t
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)

        train_loss,train_acc = train_one_task(self.model, self.tr_dataloader, self.nepochs, val_loader=self.tr_dataloader, t = t)


        return train_loss, train_acc




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
                images = images.cuda()
                targets = (targets - 10*t).cuda()
                # Forward
                offset1, offset2 = compute_offsets(t, 10)
                output = model.forward(images)

                loss = self.criterion(t, output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

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


def train_torch_model(model):
    r"""
    Perform training of the Conditional Channel Gated Network.
    The torch part in the function's name stands here to distinguish it
        from pytorch-lightning based one, which was removed with decision to
        abandon this framework.
    Args:
        model: ChannelGatedCL

    Returns:
        None
    """
    if not os.path.exists(cfg.RESULTS_ROOT):
        os.mkdir(cfg.RESULTS_ROOT)

    # logger = SummaryWriter(f'{str(cfg.LOGGING_ROOT)}/task_0')
    cifar100task = Cifar100Task('/data/lpyx/FedAgg/data/cifar100',task_num=10)
    train_datasets,test_datasets = cifar100task.getTaskDataSet()
    # 0-th task fit
    task_num = 0
    train_loader,test_loader = setTask(task_num, train_datasets, test_datasets)
    train_one_task(model, train_loader,30,val_loader=test_loader)

    save_fname = f'{cfg.RESULTS_ROOT}/{cfg.DATASET_NAME}_task_{task_num}'
    freeze_relevant_kernels(model, test_loader,
                            task_identifier=task_num, verbose=False,
                            save_freqs=True,
                            save_fname=save_fname)

    # model.save_model_state_dict(f'after_task_{0}.ckpt')
    perform_task_incremental_test(model, 1)
    # logger.close()

    for task_num in range(1, cfg.N_TASKS):
        # logger = SummaryWriter(logdir=f'{str(cfg.LOGGING_ROOT)}/task_{task_num}')
        train_loader, test_loader = setTask(task_num, train_datasets, test_datasets)
        model.add_task()

        train_one_task(model, train_loader, 30, val_loader=test_loader)

        save_fname = f'{cfg.RESULTS_ROOT}/{cfg.DATASET_NAME}_task_{task_num}'
        freeze_relevant_kernels(model, test_loader,
                                task_identifier=task_num,
                                save_freqs=True, verbose=False,
                                save_fname=save_fname)
        # model.save_model_state_dict(f'after_task_{task_num}.ckpt')
        perform_task_incremental_test(model, task_num + 1)
        # logger.close()
    # torch.save(model.state_dict(), f'{cfg.CHECKPOINTS_ROOT}/after_task_{task_num}.ckpt')


# # Todo: remove pytorch-lightning handling
# def aggregate_firing_stat_on_data(litmodel, data_loader, verbose=False):
#     r"""
#     Use data from data_loader to retrieve fates firing statistics for
#         visualization or later analysis.
#     Args:
#         litmodel: ChannelGatedCL
#         data_loader: torch.utils.data.DataLoader
#         verbose: bool, specifying usage of tqdm
#
#     Returns:
#         Gates firing statistics: list of tuples (frequencies, number_of_aggregations)
#             for each layer.
#
#     Note:
#         Normalized frequency can be calculated as frequencies / number_of_aggregations
#     """
#     # check, if litmodel is an instance of pytorch-lightning wrapper
#     lightning_model = hasattr(litmodel, 'model')
#     litmodel.to(cfg.DEVICE)
#     litmodel.enable_gates_firing_tracking()
#
#     litmodel.model.eval() if lightning_model else litmodel.eval()
#     if verbose:
#         iterator = tqdm(data_loader)
#     else:
#         iterator = data_loader
#     for x, y, task_idx in iterator:
#         x = x.to(cfg.DEVICE)
#         task_idx = task_idx.to(cfg.DEVICE)
#         _, _ = litmodel(x, task_idx)
#
#     stat = litmodel.model.get_gates_firing_stat().copy() if lightning_model else litmodel.get_gates_firing_stat().copy()
#
#     litmodel.reset_gates_firing_tracking()
#     litmodel.model.train() if lightning_model else litmodel.train()
#     return stat


# if __name__ == '__main__':
#     experiment.init()
#
#     model = ChannelGatedCL(in_ch=cfg.IN_CH, out_dim=cfg.OUT_DIM,
#                            conv_ch=cfg.CONV_CH,
#                            sparsity_patience_epochs=cfg.SPARSITY_PATIENCE_EPOCHS,
#                            lambda_sparse=cfg.LAMBDA_SPARSE,
#                            freeze_fixed_proc=cfg.FREEZE_FIXED_PROC,
#                            freeze_top_proc=cfg.FREEZE_TOP_PROC,
#                            freeze_prob_thr=cfg.FREEZE_PROB_THR).to(cfg.DEVICE)
#
#
#     train_torch_model(model)
#
#     model = ChannelGatedCL(in_ch=cfg.IN_CH, out_dim=cfg.OUT_DIM,
#                            conv_ch=cfg.CONV_CH,
#                            sparsity_patience_epochs=cfg.SPARSITY_PATIENCE_EPOCHS,
#                            lambda_sparse=cfg.LAMBDA_SPARSE,
#                            freeze_fixed_proc=cfg.FREEZE_FIXED_PROC,
#                            freeze_top_proc=cfg.FREEZE_TOP_PROC,
#                            freeze_prob_thr=cfg.FREEZE_PROB_THR).to(cfg.DEVICE)
#
#     task_incremental_acc = perform_task_incremental_test(model, cfg.N_TASKS)
#     # torch.save(task_incremental_acc, cfg.RESULTS_ROOT / 'task_incremental_acc.pt')
#     print('\n---- Task incremental accuracies ----')
#     # print(task_incremental_acc)
