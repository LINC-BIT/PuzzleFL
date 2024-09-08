from ClientTrain.AggModel.resnet import Resnet18,WideResnet
from ClientTrain.AggModel.mobilenet import Mobilenetv2
from ClientTrain.AggModel.densenet import Densenet121
from ClientTrain.AggModel.vit import SixLayerViT
from ClientTrain.AggModel.vit import RepTailTinyPiT
from ClientTrain.AggModel.sixcnn import SixCNN
from ClientTrain.models.ChannelGatemodel.model import ChannelGatedCL
from copy import deepcopy
from ClientTrain.config import cfg
def get_model_dict():
    pit = RepTailTinyPiT([3, 32, 32], outputsize=200)
    multi_model_dict = [
        {
            'name':'SixLayerViT',
            'model':SixLayerViT([3, 32, 32], outputsize=200),
            'lr':0.0005,
            'cl_method':'EWC',
            'local_epoch':4
        },
        {
            'name': 'Mobilenetv2',
            'model': Mobilenetv2([3, 32, 32], outputsize=200),
            'lr': 0.001,
            'cl_method': 'EWC',
            'local_epoch': 4
        },
        {
            'name': None,
            'model':  deepcopy(pit),
            'lr': 0.001,
            'cl_method': 'MAS',
            'local_epoch': 4
        },
        {
            'name': 'PiT',
            'model': deepcopy(pit),
            'lr': 0.001,
            'cl_method': 'MAS',
            'local_epoch':4
        },
        {
            'name': 'Resnet18',
            'model': Resnet18([3, 32, 32], outputsize=200),
            'lr': 0.0005,
            'cl_method': 'GEM',
            'local_epoch': 4
        },
        {
            'name': None,
            'model': Resnet18([3, 32, 32], outputsize=200),
            'lr': 0.0005,
            'cl_method': 'GEM',
            'local_epoch': 4
        },
        {
            'name': 'WideResnet',
            'model': WideResnet([3, 32, 32], outputsize=200),
            'lr': 0.001,
            'cl_method': 'GEM',
            'local_epoch': 4
        },
        {
            'name': 'Densenet121',
            'model': Densenet121([3, 32, 32], outputsize=200),
            'lr': 0.0005,
            'cl_method': 'GEM',
            'local_epoch': 4
        },
        {
            'name':'SixCNN',
            'model': SixCNN([3, 32, 32], outputsize=200),
            'lr': 0.0005,
            'cl_method': 'EWC',
            'local_epoch': 4
        },
        {
            'name': None,
            'model': SixCNN([3, 32, 32], outputsize=200),
            'lr': 0.0005,
            'cl_method': 'MAS',
            'local_epoch': 4
        },
        # {
        #     'name': 'ChannelGatedCL',
        #     'model': ChannelGatedCL(in_ch=cfg.IN_CH, out_dim=cfg.OUT_DIM,
        #                             conv_ch=cfg.CONV_CH,
        #                             sparsity_patience_epochs=cfg.SPARSITY_PATIENCE_EPOCHS,
        #                             lambda_sparse=cfg.LAMBDA_SPARSE,
        #                             freeze_fixed_proc=cfg.FREEZE_FIXED_PROC,
        #                             freeze_top_proc=cfg.FREEZE_TOP_PROC,
        #                             freeze_prob_thr=cfg.FREEZE_PROB_THR).to(cfg.DEVICE),
        #     'lr': 0.0005,
        #     'cl_method': 'ChannelGate',
        #     'local_epoch': 10
        # },
        # {
        #     'name': None,
        #     'model': ChannelGatedCL(in_ch=cfg.IN_CH, out_dim=cfg.OUT_DIM,
        #                    conv_ch=cfg.CONV_CH,
        #                    sparsity_patience_epochs=cfg.SPARSITY_PATIENCE_EPOCHS,
        #                    lambda_sparse=cfg.LAMBDA_SPARSE,
        #                    freeze_fixed_proc=cfg.FREEZE_FIXED_PROC,
        #                    freeze_top_proc=cfg.FREEZE_TOP_PROC,
        #                    freeze_prob_thr=cfg.FREEZE_PROB_THR).to(cfg.DEVICE),
        #     'lr': 0.0005,
        #     'cl_method': 'ChannelGate',
        #     'local_epoch': 10
        # },
    ]
    # model_dict['0'] = SixLayerViT([3, 32, 32], outputsize=100)  # Adam 0.0005 2
    # model_dict['1'] = deepcopy(model_dict['0'])  # Adam 0.0005 2
    # model_dict['2'] = RepTailTinyPiT([3, 32, 32], outputsize=100)  # Adam 0.001 2
    # model_dict['3'] = deepcopy(model_dict['0'])  # Adam 0.001 2
    # model_dict['4'] = Resnet18([3, 32, 32], outputsize=100)  # Adam 0.0005/0.001 2
    # model_dict['5'] = Resnet18([3, 32, 32], outputsize=100)  # Adam 0.0005/0.001 2
    # model_dict['6'] = WideResnet([3, 32, 32], outputsize=100)  # Adam 0.001 1
    # model_dict['7'] = Mobilenetv2([3, 32, 32], outputsize=100)  # Adam 0.001 2
    # # model_dict['8'] = Mobilenetv2([3, 32, 32], outputsize=100)  # Adam 0.001 2
    # model_dict['9'] = Densenet121([3, 32, 32], outputsize=100)  # Adam 0.0005 1
    # lr_dict = [0.0005,0.0005,0.001,0.001,0.0005,0.0005, 0.001,0.001,0.001,0.0005]
    # clmethod_dict = ['EWC','EWC','MAS','MAS','GEM','GEM','FedKNOW']
    return multi_model_dict