import os
from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn
from timm.models.vision_transformer import vit_tiny_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224
from timm.models.pit import pit_ti_224

from torch.nn import Parameter
from ClientTrain.AggModel.weitnet.layer import DecomposedConv, DecomposedLinear
from ClientTrain.AggModel.vit_utils.factory import deit_ti_32, vit_patch4_32
# from models.vits.cvt_vit_pytorch import CvT
# from models.vits.factory import vit_patch4_32, deit_ti_32
from ClientTrain.AggModel.densenet import Densenet121
from ClientTrain.AggModel.mobilenet import Mobilenetv2
from ClientTrain.AggModel.resnet import Resnet18
from ClientTrain.AggModel.sixcnn import SixCNN


def replace_module(model: nn.Module, submodule_key: str, module: nn.Module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


class WEITViT(nn.Module):
    body = None
    pre_trained_path = None

    def __init__(self, output=100, nc_per_task=10, pretrained=False):
        super().__init__()
        print('Using {}'.format(self.__class__.__name__))
        self.nc_per_task = nc_per_task
        self.n_outputs = output

        vit_body = self.body
        if pretrained:
            print('Using pretrained model: ', self.pre_trained_path)
            vit_body.load_state_dict(torch.load(self.pre_trained_path))

        if 'Levit' in vit_body.__class__.__name__:
            body_outdim = vit_body.embed_dim[-1]
        else:
            body_outdim = vit_body.get_classifier().in_features
        # body_outdim = vit_body.get_classifier().in_features
        print('body_outdim is: ', body_outdim)
        vit_body.reset_classifier(num_classes=-1)  # remove head

        # change to decomposed nn
        for name, layer in vit_body.named_modules():
            if isinstance(layer, nn.Conv2d):
                ks = layer.kernel_size[1] if isinstance(layer.kernel_size, Tuple) else layer.kernel_size
                # add for cvt
                st = layer.stride[1] if isinstance(layer.stride, Tuple) else layer.stride
                groups = layer.groups
                padding = layer.padding

                dcp_layer = DecomposedConv(layer.in_channels, layer.out_channels, kernel_size=ks, stride=st,
                                           groups=groups, padding=padding)
                replace_module(vit_body, name, dcp_layer)
            if isinstance(layer, nn.Linear):
                dcp_layer = DecomposedLinear(layer.in_features, layer.out_features)
                replace_module(vit_body, name, dcp_layer)

        self.feature_net = vit_body
        self.last = nn.Linear(body_outdim, output)

    def set_sw(self, glob_weights):
        idx = -1
        for layer, param in self.feature_net.named_parameters():
            if 'sw' in layer:
                idx += 1
                param.data = Parameter(glob_weights[idx])

    def set_knowledge(self, t, from_kbs: list):
        idx = -1
        for _, layer in self.feature_net.named_modules():
            if isinstance(layer, DecomposedConv) or isinstance(layer, DecomposedLinear):
                idx += 1
                layer.set_atten(t, from_kbs[idx].size(-1))
                layer.set_knlwledge(from_kbs[idx])

    def get_weights(self):
        weights = []

        for name, layer in self.feature_net.named_modules():
            if isinstance(layer, DecomposedConv) or isinstance(layer, DecomposedLinear):
                # print(name)
                w = layer.get_weight().detach()
                w.requires_grad = False
                weights.append(w)

        return weights

    def forward(self, x, t, pre=False, is_con=False):
        h = self.feature_net(x)
        output = self.last(h)
        if is_con:
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


class WEITTinyViT(WEITViT):
    # body = vit_tiny_patch16_224()         # this is only for 224 x 224
    body = deit_ti_32()

class WEITTinyViT224(WEITViT):
    body = vit_tiny_patch16_224()         # this is only for 224 x 224


class WEIT6layerViT(WEITViT):
    body = vit_patch4_32()

class WEITTinyPiT(WEITViT):
    body = pit_ti_224(pretrained=False, img_size=32)






class WEITCNN(nn.Module):
    # body = None
    # pre_trained_path = None

    def __init__(self, output=100, nc_per_task=10, body=None, model_name=None):
        super().__init__()
        print('Using {}'.format(self.__class__.__name__))
        self.nc_per_task = nc_per_task
        self.n_outputs = output
        self.model_name = model_name
        vit_body = body.feature_net

        # change to decomposed nn
        for name, layer in vit_body.named_modules():
            if isinstance(layer, nn.Conv2d):
                ks = layer.kernel_size[1] if isinstance(layer.kernel_size, Tuple) else layer.kernel_size
                # add for cvt
                st = layer.stride[1] if isinstance(layer.stride, Tuple) else layer.stride
                groups = layer.groups
                padding = layer.padding

                dcp_layer = DecomposedConv(layer.in_channels, layer.out_channels, kernel_size=ks, stride=st,
                                           groups=groups, padding=padding)
                replace_module(vit_body, name, dcp_layer)
            if isinstance(layer, nn.Linear):
                dcp_layer = DecomposedLinear(layer.in_features, layer.out_features)
                replace_module(vit_body, name, dcp_layer)

        self.feature_net = vit_body
        self.last = body.last

    def set_sw(self, glob_weights):
        idx = -1
        for layer, param in self.feature_net.named_parameters():
            if 'sw' in layer:
                idx += 1
                param.data = Parameter(glob_weights[idx])

    def set_knowledge(self, t, from_kbs: list):
        idx = -1
        for _, layer in self.feature_net.named_modules():
            if isinstance(layer, DecomposedConv) or isinstance(layer, DecomposedLinear):
                idx += 1
                layer.set_atten(t, from_kbs[idx].size(-1))
                layer.set_knlwledge(from_kbs[idx])

    def get_weights(self):
        weights = []

        for name, layer in self.feature_net.named_modules():
            if isinstance(layer, DecomposedConv) or isinstance(layer, DecomposedLinear):
                # print(name)
                w = layer.get_weight().detach()
                w.requires_grad = False
                weights.append(w)

        return weights

    def forward(self, x, t, pre=False, is_con=False):
        if 'resnet' in self.model_name or 'mobinet' in self.model_name:
            h = self.feature_net(x, avg_act=False)[0]
        else:
            h = self.feature_net(x)[0]
        output = self.last(h)
        if is_con:
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
    
    
class WEITResnet18(WEITCNN):
    def __init__(self, output=100, nc_per_task=10, body=None):
        body = Resnet18(inputsize=None, outputsize=output, nc_per_task=nc_per_task)
        super().__init__(output, nc_per_task, body=body, model_name="resnet")
    
    
class WEIT6CNN(WEITCNN):
    def __init__(self, output=100, nc_per_task=10, body=None):
        body = SixCNN([3,32,32], outputsize=output)
        super().__init__(output, nc_per_task, body=body, model_name="cnn")
 
class WEITMobiNet(WEITCNN):
    def __init__(self, output=100, nc_per_task=10, body=None):
        body = Mobilenetv2(inputsize=None, outputsize=output, nc_per_task=nc_per_task)
        super().__init__(output, nc_per_task, body=body, model_name="mobinet")


class WEITDense(WEITCNN):
    def __init__(self, output=100, nc_per_task=10, body=None):
        body = Densenet121(inputsize=None, outputsize=output, nc_per_task=nc_per_task)
        super().__init__(output, nc_per_task, body=body,model_name="densenet")
 
    
    
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICE'] = '0'  # compatible to cuda()
    # device = torch.device('cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu')
    model = WEITTinyPiT(output=200, nc_per_task=10)
    
    
    for n, p in model.named_parameters():
        print(n)