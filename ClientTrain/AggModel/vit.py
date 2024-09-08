
from abc import abstractmethod


import torch

from torch import nn

from ClientTrain.AggModel.vit_utils.factory import vit_patch4_32, deit_ti_32
from timm.models.vision_transformer import  deit_small_patch16_224
from timm.models.pit import pit_ti_224

class RepTailViT(nn.Module):
    body = None
    pre_trained_path = None

    def __init__(self, inputsize,outputsize=100, nc_per_task=10, pretrained=False):
        super().__init__()
        vit_body = self.body
        if pretrained:
            print('Using pretrained model: ', self.pre_trained_path)
            vit_body.load_state_dict(torch.load(self.pre_trained_path))
        assert getattr(vit_body, 'get_classifier'), "get_classifier method must be in ViT model"

        if 'Levit' in vit_body.__class__.__name__:
            body_outdim = vit_body.embed_dim[-1]
        else:
            body_outdim = vit_body.get_classifier().in_features
        print('body_outdim is: ', body_outdim)
        vit_body.reset_classifier(num_classes=-1)       # remove head

        # # remove pre_trained head
        # if pretrained:
        #     assert self.pre_trained_path is not None
        #     pre_weights = torch.load(os.path.join(self.pre_trained_path))
        #     pre_weights.pop('head.weight')
        #     pre_weights.pop('head.bias')
        #     vit_body.load_state_dict(pre_weights)

        self.feature_net = vit_body
        self.last = torch.nn.Linear(body_outdim, outputsize)
        self.weight_keys = []
        self.prune_keys = []
        self.nc_per_task = nc_per_task
        self.n_outputs = outputsize
        # set weight_keys
        for name, para in self.named_parameters():
            temp = []
            if 'last' not in name:
                temp.append(name)
                self.weight_keys.append(temp)

    def forward(self,x,t=-1,pre=False,acg_act=False):
        h = self.feature_net(x)
        output = self.last(h)
        if t!=-1:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def get_prune_keys(self):
        if not self.prune_keys:
            self.prune_keys = self._gen_prune_keys()
        return self.prune_keys

    @abstractmethod
    def _gen_prune_keys(self):
        pass


class SixLayerViT(RepTailViT):
    body = vit_patch4_32()
    pre_trained_path = None
    prune_method = 'b&all'

    def _gen_prune_keys(self):
        _prune_keys = []
        if self.prune_method == 'b&qkv-v1':
            for name, _ in self.body.named_parameters():
                if 'bias' not in name and 'attn.qkv' not in name:
                    _prune_keys.append(name)
        elif self.prune_method == 'b&all':
            _prune_keys = [name + '.weight' for name, module in self.body.named_modules()
                           if isinstance(module, torch.nn.modules.linear.Linear)]
        elif self.prune_method == 'b&qkv-v2':
            _prune_keys = [name + '.weight' for name, module in self.body.named_modules()
                           if isinstance(module, torch.nn.modules.linear.Linear) and 'qkv' not in name]
        else:
            raise ValueError(f"prune method %s not support" % self.prune_method)

        print('Pruned keys are: ', _prune_keys)
        return _prune_keys

class RepTailTinyViT(RepTailViT):
    # body = vit_tiny_patch16_224()
    body = deit_ti_32()
    pre_trained_path = None

    def _gen_prune_keys(self):
        _prune_keys = []
        for name, _ in self.body.named_parameters():
            if 'bias' not in name or 'attn.qkv' not in name:
                _prune_keys.append(name)
        print('Pruned keys are: ', _prune_keys)
        return _prune_keys

class RepTailTinyPiT(RepTailViT):
    # 4.9M
    # body = pit_ti_224(pretrained=False)       # for 224
    body = pit_ti_224(pretrained=False, img_size=32)
    # pre_trained_path = os.path.join(HOME, "pre_train/pit_ti_224.pth")

    def _gen_prune_keys(self):
        _prune_keys = []
        for name, _ in self.body.named_parameters():
            if 'bias' not in name:
                _prune_keys.append(name)
        print('Pruned keys are: ', _prune_keys)
        return _prune_keys
    
    
    
if __name__ == '__main__':
    
    import copy

    

    # def deit_ti_32():
    #     # Just a placeholder for the actual implementation
    #     class DummyModel:
    #         def named_parameters(self):
    #             return {'layer1.weight': None, 'layer1.bias': None, 'attn.qkv': None}.items()
    #     return DummyModel()

    # Create an instance of RepTailTinyViT
    model1 = RepTailTinyViT(inputsize=None)

    # Create a deep copy of model1
    model2 = copy.deepcopy(model1)

    # Check if both instances have the same body object
    # print(model1.body is model2.body)  # Should print: False
    print(id(model1.feature_net))
    print(id(model2.feature_net))
