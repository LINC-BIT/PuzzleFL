"""
Re-implementation of PackNet Continual Learning Method
"""

import torch
from torch import nn
import numpy as np
from copy import deepcopy
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
class PackNet():

    def __init__(self, n_tasks, prune_instructions=0.95, prunable_types=(nn.Conv2d, nn.Linear),device =None):

        self.n_tasks = n_tasks
        self.prune_instructions = prune_instructions
        self.prunable_types = prunable_types
        self.device = device
        # Set up an array of quantiles for pruning procedure
        # if n_tasks:
        #     self.config_instructions()

        self.PATH = None
        self.prun_epoch = 5
        self.tune_epoch = 5
        self.current_task = 0
        self.masks = []  # 3-dimensions: task (list), layer (dict), parameter mask (tensor)
        self.mode = None

    def prune(self, t,model, prune_quantile):
        """
        Create task-specific mask and prune least relevant weights
        :param model: the model to be pruned
        :param prune_quantile: The percentage of weights to prune as a decimal
        """
        # Calculate Quantil
        all_prunable = torch.tensor([]).to(self.device)
        for name, param_layer in model.named_parameters():
            if 'bias' not in name:

                # get fixed weights for this layer
                prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)

                for task in self.masks:
                    if name in task:
                        prev_mask |= task[name]

                p = param_layer.masked_select(~prev_mask)

                if p is not None:
                    all_prunable = torch.cat((all_prunable.view(-1), p), -1)
        B = torch.abs(all_prunable.cpu()).detach().numpy()
        cutoff = np.quantile(B, q=prune_quantile)
        mask = {}  # create mask for this task
        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if 'bias' not in name:
                    # get weight mask for this layer
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device) # p
                    for task in self.masks:
                        if name in task:
                            prev_mask |= task[name]

                    curr_mask = torch.abs(param_layer).ge(cutoff)  # q
                    curr_mask = torch.logical_and(curr_mask, ~prev_mask)  # (q & ~p)

                    # Zero non masked weights
                    param_layer *= (curr_mask | prev_mask)

                    mask[name] = curr_mask
        if len(self.masks) <= t :
            self.masks.append(mask)
        else:
            self.masks[t] = mask

    def fine_tune_mask(self, model,t):
        """
        Zero the gradgradient of pruned weights this task as well as previously fixed weights
        Apply this mask before each optimizer step during fine-tuning
        """
        assert len(self.masks) > t
        mask_idx = 0
        for name, param_layer in model.named_parameters():
            if 'bias' not in name and  param_layer.grad is not None:
                param_layer.grad *= self.masks[t][name]
                mask_idx += 1

    def training_mask(self, model):
        """
        Zero the gradient of only fixed weights for previous tasks
        Apply this mask after .backward() and before
        optimizer.step() at every batch of training a new task
        """
        if len(self.masks) == 0:
            return
        for name, param_layer in model.named_parameters():
            if 'bias' not in name:
                # get mask of weights from previous tasks
                prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)

                for task in self.masks:
                    prev_mask |= task[name]

                # zero grad of previous fixed weights
                if param_layer.grad is not None:
                    param_layer.grad *= ~prev_mask

    def fix_biases(self, model):
        """
        Fix the gradient of prunable bias parameters
        """
        for name, param_layer in model.named_parameters():
            if 'bias' in name:
                param_layer.requires_grad = False

    def fix_batch_norm(self, model):
        """
        Fix batch norm gain, bias, running mean and variance
        """
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.affine = False
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False

    def apply_eval_mask(self, model, task_idx):
        """
        Revert to network state for a specific task
        :param model: the model to apply the eval mask to
        :param task_idx: the task id to be evaluated (0 - > n_tasks)
        """
        assert len(self.masks) > task_idx

        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if 'bias' not in name:

                    # get indices of all weights from previous masks
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                    for i in range(0, task_idx + 1):
                        prev_mask |= self.masks[i][name]

                    # zero out all weights that are not in the mask for this task
                    param_layer *= prev_mask

    def mask_remaining_params(self, model):
        """
        Create mask for remaining parameters
        """
        mask = {}
        for name, param_layer in model.named_parameters():
            if 'bias' not in name:

                # Get mask of weights from previous tasks
                prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                for task in self.masks:
                    prev_mask |= task[name]

                # Create mask of remaining parameters
                layer_mask = ~prev_mask
                mask[name] = layer_mask

        self.masks.append(mask)

    def total_epochs(self):
        return self.prun_epoch + self.tune_epoch

    def config_instructions(self):
        """
        Create pruning instructions for this task split
        :return: None
        """
        assert self.n_tasks is not None

        if not isinstance(self.prune_instructions, list):  # if a float is passed in
            assert 0 < self.prune_instructions < 1
            self.prune_instructions = [self.prune_instructions] * (self.n_tasks - 1)
        assert len(self.prune_instructions) == self.n_tasks - 1, "Must give prune instructions for each task"

    def save_final_state(self, model, PATH='model_weights.pth'):
        """
        Save the final weights of the model after training
        :param model: pl_module
        :param PATH: The path to weights file
        """
        self.PATH = PATH
        torch.save(model.state_dict(), PATH)

    def load_final_state(self, model):
        """
        Load the final state of the model
        """
        model.load_state_dict(torch.load(self.PATH))

    def on_init_end(self,pl_module,task):
        self.mode = 'train'
        if task !=0 :
            self.fix_biases(pl_module)  # Fix biases after first task
            self.fix_batch_norm(pl_module)  # Fix batch norm mean, var, and params

    def on_after_backward(self, pl_module,t):

        if self.mode == 'train':
            self.training_mask(pl_module)

        elif self.mode == 'fine_tune':
            self.fine_tune_mask(pl_module,t)

    def on_epoch_end(self, pl_module,epoch,task):

        if epoch == self.prun_epoch - 1:  # Train epochs completed
            self.mode = 'fine_tune'
            if task == self.n_tasks - 1:
                self.mask_remaining_params(pl_module)
            else:
                self.prune(task,
                    model=pl_module,
                    prune_quantile=self.prune_instructions)

        elif epoch == self.total_epochs() - 1:  # Train and fine tune epochs completed
            self.mode = 'train'


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, tr_dataloader, nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5,
                 clipgrad=100,args=None):
        self.device = args.device
        self.pack = PackNet(n_tasks=args.task, device=args.device)
        self.num_classes = args.num_classes
        self.model = model
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_min = lr_min * 1 / 3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.lamb = args.lamb
        self.e_rep = args.local_rep_ep
        self.old_task=-1
        self.args = args
        return
    def set_model(self,model):
        self.model = model

    def set_trData(self,tr_dataloader):
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
        if t != self.old_task:
            self.old_task = t
            self.first_train = True
        self.model.to(self.device)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        self.pack.on_init_end(self.model,t)
        # trian model
        if len(self.pack.masks) > t:
            self.pack.masks.pop()
        for e in range(self.nepochs):

            self.train_packnet(t)
            self.pack.on_epoch_end(self.model.feature_net,e,t)


            train_loss, train_acc = self.eval(t)
            if e % 5 == 4:
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                    e + 1,  train_loss, 100 * train_acc), end='')

        return train_loss, train_acc
    def train_packnet(self,t,kd_lambda = 0.0):
        self.model.train()
        for images, targets in self.tr_dataloader:
            images = images.to(self.device)
            targets = (targets - self.num_classes * t).to(self.device)
            offset1, offset2 = compute_offsets(t, self.num_classes)
            outputs = self.model.forward(images, t)[:, offset1:offset2]
            loss = self.ce(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.pack.on_after_backward(self.model.feature_net,t)
            self.optimizer.step()



    def moretrain(self,t):
        self.model.to(self.device)
        for e in range(self.nepochs):
            self.train_packnet(t)
            self.pack.on_epoch_end(self.model.feature_net, e, t)

    def eval(self, t, train=True, model=None):
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
                images = images.to(self.device)
                targets = (targets - 10 * t).to(self.device)
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

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
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