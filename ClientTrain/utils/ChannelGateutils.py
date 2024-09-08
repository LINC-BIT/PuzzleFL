r"""Train routines.

This module contains various functions, which are aimed to structure the
training procedure of the Conditional Channel Gated Networks.
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ClientTrain.config import cfg
from ClientTrain.dataset.Cifar100 import Cifar100Task,setTask



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






def train_one_task(model, train_loader,epochs, kd_model=None, kd_lambda=0.0, val_loader=None, logger=None,t=0):
    optimizer = cfg.OPT(model.parameters())
    scheduler = cfg.SCHEDULER(optimizer) if cfg.SCHEDULER else None
    for epoch_num in tqdm(range(epochs)):
        train_step(model,kd_model, optimizer, scheduler, epoch_num, train_loader, logger, kd_lambda, t=t)
        val_dict = {}



        if epoch_num % 5 ==0:
            if val_loader is not None:
                val_dict = val_step(model, epoch_num, val_loader, logger, t=t)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                epoch_num + 1, val_dict['val_loss'], 100 * val_dict['val_acc']), end='')


        # if epoch_num % cfg.CKPT_FREQ == 0:
        #     torch.save(model.state_dict(), f'{cfg.CHECKPOINTS_ROOT}/task_{0}_ep{epoch_num}.pt')
    val_dict = val_step(model, epoch_num, val_loader, logger,t=t)
    return val_dict['val_loss'], val_dict['val_acc']


def val_step(model, epoch_num, val_loader, logger=None, t=0):
    r"""
    Perform one validation epoch.
    Args:
        model: ChannelGatedCL
        epoch_num: int
        val_loader: torch.utils.data.DataLoader
        logger: tensorboardX logger

    Returns:
        tensorboard_logs dict
    """
    epoch_logs = []

    model.eval()
    for batch in val_loader:
        x, y = batch
        x = x.to(cfg.DEVICE)
        y = (y - 10*t).to(cfg.DEVICE)
        head_idx = torch.full_like(y, t).to(cfg.DEVICE)

        out = model(x, head_idx, task_supervised_eval=cfg.TASK_SUPERVISED_VALIDATION)
        head_loss = F.cross_entropy(out, y)
        val_loss = head_loss

        bs = y.shape[0]
        val_acc = (F.softmax(out, dim=-1).argmax(dim=-1) == y).sum() / float(bs)

        epoch_logs.append({'loss': val_loss.detach().cpu(),
                           'acc': val_acc.detach().cpu()})

    avg_loss = torch.stack([x['loss'] for x in epoch_logs]).mean()
    avg_acc = torch.stack([x['acc'] for x in epoch_logs]).mean()

    tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

    if logger:
        for k,v in tensorboard_logs.items():
            logger.add_scalar(k, v, epoch_num)

    return tensorboard_logs

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
def train_step(model, kd_model, optimizer, scheduler, epoch_num, train_loader, logger=None, kd_lambda=0.0, t = 0):
    r"""
    Perform one training epoch.
    Args:
        model: ChannelGatedCL
        optimizer: optimizer from torch.optim
        scheduler: scheduler from torch.optim
        epoch_num: int
        train_loader: torch.utils.data.DataLoader
        logger: tensorboardX logger

    Returns:
        None
    """
    epoch_logs = []
    model.train()
    if kd_model is not None:
        kd_model.eval()
    for batch in train_loader:
        x, y = batch
        x = x.to(cfg.DEVICE)
        y = (y - 10 * t).to(cfg.DEVICE)
        head_idx = torch.full_like(y, t).to(cfg.DEVICE)

        optimizer.zero_grad()
        out = model(x, head_idx)

        head_loss = F.cross_entropy(out, y)

        if epoch_num <= cfg.SPARSITY_PATIENCE_EPOCHS:
            sparsity_loss = torch.FloatTensor([0]).to(cfg.DEVICE)
        else:
            sparsity_loss = model.calc_sparcity_loss(head_idx).to(cfg.DEVICE)
        kd_loss = 0
        if kd_model is not None:
            offset1, offset2 = compute_offsets(t, 10)
            kd_output = kd_model.forward(x, t)[:, offset1:offset2]
            kd_loss = kd_lambda*MultiClassCrossEntropy(out,kd_output,0,T=2)

        loss = head_loss + sparsity_loss + kd_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_VAL)
        optimizer.step()

        bs = y.shape[0]
        train_acc = (F.softmax(out, dim=-1).argmax(dim=-1) == y).sum() / float(bs)

        epoch_logs.append({'loss': loss.detach().cpu(),
                            'acc': train_acc.detach().cpu(),
                            'head_loss': head_loss.detach().cpu(),
                            'sparse_loss': sparsity_loss.detach().cpu()})

    avg_loss = torch.stack([x['loss'] for x in epoch_logs]).mean()
    avg_acc = torch.stack([x['acc'] for x in epoch_logs]).mean()
    avg_head_loss = torch.stack([x['head_loss'] for x in epoch_logs]).mean()
    avg_sparse_loss = torch.stack([x['sparse_loss'] for x in epoch_logs]).mean()

    # tensorboard_logs = {
    #     'train_loss': avg_loss,
    #     'train_acc': avg_acc,
    #     'head_loss': avg_head_loss,
    #     'sparse_loss': avg_sparse_loss
    # }

    #



# Todo: remove pytorch-lightning handling
def test_step(lit_model, test_dataloaders):
    epoch_logs = []
    lit_model.eval()
    acc_num=0
    all_num=0
    for batch in test_dataloaders:
        x, y, head_idx = batch
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)
        head_idx = head_idx.to(cfg.DEVICE)

        out = lit_model(x, head_idx, task_supervised_eval=cfg.TASK_SUPERVISED_VALIDATION)
        head_loss = F.cross_entropy(out, y)
        val_loss = head_loss

        all_num += y.shape[0]
        acc_num += (F.softmax(out, dim=-1).argmax(dim=-1) == y).sum()


    avg_acc = acc_num / float(all_num)
    print(avg_acc)

    tensorboard_logs = {'test_acc': avg_acc}


    return tensorboard_logs


def perform_task_incremental_test(lit_model, N_tasks):
    r"""
    When all tasks has been trained, use this function to check,
        how quality changed/stayed the same in the process of learning new
        tasks

    The last column represents final quality after all tasks being fitted and
        relevant kernels for each task being frozen.

    Args:
        lit_model: ChannelGatedCL
        N_tasks: int, total number of tasks

    Returns:
        torch.Tensor, with row i representing quality on the i-th task
            and column j specifying the snapshot moment:
            the quality is checked after tasks 0..j had been fitted
            and relevant kernels frozen.
    """
    scores = torch.zeros((N_tasks, N_tasks), dtype=torch.float)

    cifar100task = Cifar100Task('../data/cifar100', task_num=10)
    train_datasets, test_datasets = cifar100task.getTaskDataSet()

    for task_fitted_num in range(0, N_tasks):
        load_after_next_task(lit_model, task_fitted_num)

        for prev_task_num in range(0, task_fitted_num + 1):
            train_loader, test_loader = setTask(prev_task_num, train_datasets, test_datasets)
            test_results = test_step(lit_model, test_dataloaders=test_loader)
            scores[prev_task_num, task_fitted_num] = test_results['test_acc']

    return scores


def load_after_next_task(lit_model, next_task_num):
    r"""
    if the model currently supports k tasks, appends for an upcoming task
        and loads proper checkpoint
    Args:
        lit_model: ChannelGatedCL
        next_task_num: int

    Returns:
        None
    """
    if next_task_num == 0:
        lit_model.load_model_state_dict(f'after_task_{next_task_num}.ckpt')
    else:
        lit_model.add_task()
        lit_model.load_model_state_dict(f'after_task_{next_task_num}.ckpt')


def load_after_many_tasks(lit_model, prev_task_num, next_task_num):
    r"""
    Loads checkpoint, corresponding to next_task_num,
        appending the model with parameters along the way
    Args:
        lit_model: ChannelGatedCL
        prev_task_num: int, currently loaded task
        next_task_num:

    Returns:

    """
    n_additions = next_task_num - prev_task_num
    for _ in range(n_additions):
        lit_model.add_task()

    lit_model.load_model_state_dict(f'after_task_{next_task_num}.ckpt')
