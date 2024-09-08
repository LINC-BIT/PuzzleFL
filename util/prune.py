import torch
import numpy as np
from torch import nn

def compute_offsets(task, nc_per_task, is_cifar=True):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar and task != -1:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2
class ModelPrune():

    def __init__(self,prune_instructions=0.95, prunable_types=(nn.Conv2d, nn.Linear),device =None):

        self.prune_instructions = prune_instructions
        self.prunable_types = prunable_types
        self.device = device
        self.PATH = None
        self.prun_epoch = 0
        self.tune_epoch = 10
        self.current_task = 0
        self.mask = {}  # 3-dimensions: task (list), layer (dict), parameter mask (tensor)
        self.ce = nn.CrossEntropyLoss()

    def prune(self, model, prune_quantile):
        # Calculate Quantil
        all_prunable = torch.tensor([]).to(self.device)
        for name, param_layer in model.named_parameters():
            if 'bias' not in name:
                prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                p = param_layer.masked_select(~prev_mask)
                if p is not None:
                    all_prunable = torch.cat((all_prunable.view(-1), p), -1)
        B = torch.abs(all_prunable.cpu()).detach().numpy()
        cutoff = np.quantile(B, q=prune_quantile)
        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if 'bias' not in name:
                    curr_mask = torch.abs(param_layer).ge(cutoff)  # q
                    param_layer *= curr_mask
                    self.mask[name] = curr_mask

    def fine_tune_mask(self, model):
        mask_idx = 0
        for name, param_layer in model.named_parameters():
            if 'bias' not in name and  param_layer.grad is not None:
                param_layer.grad *= self.mask[name]
                mask_idx += 1


    def fix_biases(self, model):

        for name, param_layer in model.named_parameters():
            if 'bias' in name:
                param_layer.requires_grad = False

    def fix_batch_norm(self, model):

        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.affine = False
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False

    def apply_eval_mask(self, model):
        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if 'bias' not in name:

                    # get indices of all weights from previous masks
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                    prev_mask |= self.mask[name]

                    # zero out all weights that are not in the mask for this task
                    param_layer *= prev_mask

    def on_after_backward(self, pl_module):
        self.fine_tune_mask(pl_module)

    def on_epoch_end(self, pl_module):
        self.prune(model=pl_module,
                    prune_quantile=self.prune_instructions)

    def decode_model(self,model, state_dict):
        for name, parameter in model.named_parameters():
            cur_all_data = torch.flatten(parameter.data)
            cur_position = state_dict[name]['position']
            big_weight = state_dict[name]['weight']
            new_weight = torch.zeros(cur_all_data.shape)
            new_weight.scatter_(0, cur_position, big_weight)
            parameter.data = new_weight.view(parameter.data.shape)

    def encode_model(self,prune_model):
        pos_weight = {}
        for name, parameter in prune_model.named_parameters():
            pos_weight[name] = {}
            d = torch.flatten(parameter.data)
            cur_position = torch.nonzero(d)
            cur_position = torch.squeeze(cur_position)
            # print(cur_position.shape)
            big_weight = torch.index_select(d, 0, cur_position)
            # print(big_weight.shape)
            pos_weight[name]['position'] = torch.LongTensor(cur_position.cpu())
            pos_weight[name]['weight'] = big_weight.cpu()
        encode_model_dict = prune_model.state_dict()

        for key in encode_model_dict.keys():
            encode_model_dict[key] = pos_weight[key]
        return encode_model_dict

    def prune_one_model(self,model,dataloader,t = -1):
        model = model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        for e in range(self.prun_epoch):
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = (targets - 10 * t).to(self.device)
                # Forward current model
                offset1, offset2 = compute_offsets(t, 10)
                outputs = model.forward(images, 0)
                # outputs = model.forward(images, t)[:, offset1:offset2]
                loss = self.ce(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.on_epoch_end(model)
        for e in range(self.tune_epoch):
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = (targets - 10 * t).to(self.device)
                # Forward current model
                offset1, offset2 = compute_offsets(t, 10)
                outputs = model.forward(images, 0)
                # outputs = model.forward(images, t)[:, offset1:offset2]
                loss = self.ce(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                self.on_after_backward(model)
                optimizer.step()
        return model.state_dict()
        # return self.encode_model(model)
        
class InfoPrune():
    
    def __init__(self,info,prune_instructions=0.95, prunable_types=(nn.Conv2d, nn.Linear),device =None):

        self.prune_instructions = prune_instructions
        self.prunable_types = prunable_types
        self.device = device
        self.PATH = None
        self.prun_epoch = 0
        self.tune_epoch = 0
        self.current_task = 0
        self.mask = {}  # 3-dimensions: task (list), layer (dict), parameter mask (tensor)
        self.info = info
        self.ce = nn.CrossEntropyLoss()
        
    def prune(self, model, prune_quantile):
        # Calculate Quantil
        all_prunable = torch.tensor([]).to(self.device)
        for name, param_layer in self.info.items():
            if 'bias' not in name:
                prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                p = param_layer.masked_select(~prev_mask)
                if p is not None:
                    all_prunable = torch.cat((all_prunable.view(-1), p), -1)
        B = torch.abs(all_prunable.cpu()).detach().numpy()
        cutoff = np.quantile(B, q=prune_quantile)
        
        with torch.no_grad():
            for name, param_layer in self.info.items():
                if 'bias' not in name:
                    curr_mask = torch.abs(param_layer).ge(cutoff)  # q
                    param_layer *= curr_mask
                    self.mask[name] = curr_mask
        
        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if 'bias' not in name:
                    param_layer *= self.mask[name]
    
        # # Calculate Quantil
        # all_prunable = torch.tensor([]).to(self.device)
        # for name, param_layer in model.named_parameters():
        #     if 'bias' not in name:
        #         prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
        #         p = param_layer.masked_select(~prev_mask)
        #         if p is not None:
        #             all_prunable = torch.cat((all_prunable.view(-1), p), -1)
        # B = torch.abs(all_prunable.cpu()).detach().numpy()
        # cutoff = np.quantile(B, q=prune_quantile)
        # with torch.no_grad():
        #     for name, param_layer in model.named_parameters():
        #         if 'bias' not in name:
        #             curr_mask = torch.abs(param_layer).ge(cutoff)  # q
        #             param_layer *= curr_mask
        #             self.mask[name] = curr_mask

    def fine_tune_mask(self, model):
        mask_idx = 0
        for name, param_layer in model.named_parameters():
            if 'bias' not in name and  param_layer.grad is not None:
                param_layer.grad *= self.mask[name]
                mask_idx += 1


    def fix_biases(self, model):

        for name, param_layer in model.named_parameters():
            if 'bias' in name:
                param_layer.requires_grad = False

    def fix_batch_norm(self, model):

        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.affine = False
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False

    def apply_eval_mask(self, model):
        with torch.no_grad():
            for name, param_layer in model.named_parameters():
                if 'bias' not in name:

                    # get indices of all weights from previous masks
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False).to(self.device)
                    prev_mask |= self.mask[name]

                    # zero out all weights that are not in the mask for this task
                    param_layer *= prev_mask

    def on_after_backward(self, pl_module):
        self.fine_tune_mask(pl_module)

    def on_epoch_end(self, pl_module):
        self.prune(model=pl_module,
                    prune_quantile=self.prune_instructions)

    def decode_model(self,model, state_dict):
        for name, parameter in model.named_parameters():
            cur_all_data = torch.flatten(parameter.data)
            cur_position = state_dict[name]['position']
            big_weight = state_dict[name]['weight']
            new_weight = torch.zeros(cur_all_data.shape)
            new_weight.scatter_(0, cur_position, big_weight)
            parameter.data = new_weight.view(parameter.data.shape)

    def encode_model(self,prune_model):
        pos_weight = {}
        for name, parameter in prune_model.named_parameters():
            pos_weight[name] = {}
            d = torch.flatten(parameter.data)
            cur_position = torch.nonzero(d)
            cur_position = torch.squeeze(cur_position)
            # print(cur_position.shape)
            big_weight = torch.index_select(d, 0, cur_position)
            # print(big_weight.shape)
            pos_weight[name]['position'] = torch.LongTensor(cur_position.cpu())
            pos_weight[name]['weight'] = big_weight.cpu()
        encode_model_dict = prune_model.state_dict()

        for key in encode_model_dict.keys():
            encode_model_dict[key] = pos_weight[key]
        return encode_model_dict

    def prune_one_model(self,model,dataloader,t = -1):
        model = model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        for e in range(self.prun_epoch):
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = (targets - 10 * t).to(self.device)
                # Forward current model
                offset1, offset2 = compute_offsets(t, 10)
                outputs = model.forward(images, 0)
                # outputs = model.forward(images, t)[:, offset1:offset2]
                loss = self.ce(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.on_epoch_end(model)
        for e in range(self.tune_epoch):
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = (targets - 10 * t).to(self.device)
                # Forward current model
                offset1, offset2 = compute_offsets(t, 10)
                outputs = model.forward(images, 0)
                # outputs = model.forward(images, t)[:, offset1:offset2]
                loss = self.ce(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                self.on_after_backward(model)
                optimizer.step()
        return model.state_dict(), model
        # return self.encode_model(model)