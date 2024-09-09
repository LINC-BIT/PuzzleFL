"""
Re-implementation of PackNet Continual Learning Method
"""

import torch
from torch import nn
import numpy as np


def get_prune_keys(feature_net: nn.Module):
    prune_keys = []
    for name, _ in feature_net.named_parameters():
        if 'bias' not in name or 'attn.qkv' not in name:
            prune_keys.append(name)
    print('Pruned keys are: ', prune_keys)
    return prune_keys


class PackNetViT():

    def __init__(self, n_tasks, local_ep,local_rep_ep,prune_instructions=.9, prunable_types=(nn.Conv2d, nn.Linear),
                 device =None, net=None):
        """

            Args:
                n_tasks (int): total task number
                local_ep (int): local training epochs
                local_rep_ep (int):  local representation training epochs
                prune_instructions (float): prune rate
                prunable_types (Tuple[nn.Module]): module type to be pruned
                device:
        """
        self.n_tasks = n_tasks
        self.prune_instructions = prune_instructions
        self.prunable_types = prunable_types
        self.device = device
        self.PATH = None
        # self.prun_epoch = local_ep - local_rep_ep
        # self.tune_epoch = local_rep_ep
        self.prun_epoch = int(local_ep/2)
        self.tune_epoch = local_ep - self.prun_epoch
        self.current_task = 0
        self.masks = []  # 3-dimensions: [(task_idx, layer_name: parameter mask (tensor)]
        self.mode = None

        self.prune_keys = net.get_prune_keys()
        
        _prune_keys = []
        for name, _ in net.feature_net.named_parameters():
            if 'bias' not in name:
                _prune_keys.append(name)
                
        print('Using ViT PackNet, and got prune_keys: {}'.format(self.prune_keys))

    def prune(self, t,model, prune_quantile):
        """
        Create task-specific mask and prune least relevant weights
        :param model: the model to be pruned
        :param prune_quantile: The percentage of weights to prune as a decimal
        """
        # Calculate Quantil
        all_prunable = torch.tensor([]).to(self.device)
        for name, param_layer in model.named_parameters():
            if name in self.prune_keys:

                prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                for task in self.masks:
                    if name in task:
                        prev_mask |= task[name]
                p = param_layer.masked_select(~prev_mask)       # only get unmasked
                if p is not None:
                    all_prunable = torch.cat((all_prunable.view(-1), p), -1)
        B = torch.abs(all_prunable.cpu()).detach().numpy()
        # if B.shape[0] == 0:
        #     cutoff = float('inf')
        # else:
        cutoff = np.quantile(B, q=prune_quantile)

        # create mask for this task
        mask = {}
        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if name in self.prune_keys:

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

    def fine_tune_mask(self, model, t):
        """
        训练当前任务参数：只保留剪枝后当前任务参数的梯度，当新任务的mask * param.grad
        """
        assert len(self.masks) > t
        mask_idx = 0
        for name, param_layer in model.named_parameters():
            if name in self.prune_keys and param_layer.grad is not None:
                param_layer.grad *= self.masks[t][name]
                mask_idx += 1

    def training_mask(self, model):
        """
        训练剩余模型参数：将所有旧任务模型参数梯度置零，只更新可用的剩余参数。
        """
        if len(self.masks) == 0:
            return
        for name, param_layer in model.named_parameters():
            if name in self.prune_keys:

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
            if name in self.prune_keys:
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

    def fix_layer_norm(self, model):
        """
        Fix batch norm gain, bias, running mean and variance
        """
        for mod in model.modules():
            if isinstance(mod, nn.LayerNorm):
                mod.affine = False
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False

    def apply_eval_mask(self, model, task_idx):
        """
        激活task_idx之前的所有任务模型参数，其余参数值置为0.
        :param model: the model to apply the eval mask to
        :param task_idx: the task id to be evaluated (0 - > n_tasks)
        """
        assert len(self.masks) > task_idx

        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if name in self.prune_keys:

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
            if name in self.prune_keys:

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
        """
            Args:
                 pl_module (nn.Module): feature net of model
                 task (int): task id
        """
        self.mode = 'train'
        if task !=0 :
            self.fix_biases(pl_module)  # Fix biases after first task
            self.fix_batch_norm(pl_module)  # Fix batch norm mean, var, and params
            self.fix_layer_norm(pl_module)  # added for vit model

    def on_after_backward(self, pl_module, t):

        if self.mode == 'train':
            self.training_mask(pl_module)

        elif self.mode == 'fine_tune':
            self.fine_tune_mask(pl_module,t)

    def on_epoch_end(self, pl_module,epoch,task):
        """ model pruning
            Args:
                 pl_module (nn.Module): model.feature_net
                 epoch (int):
                 task (int): task id
        """
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


if __name__ == '__main__':

    # PackNetViT(1, 1, 1, net=RepTailCvT())
    pass

