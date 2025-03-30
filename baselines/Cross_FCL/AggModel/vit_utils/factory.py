import torch
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import _create_vision_transformer
from ClientTrain.AggModel.vit_utils.vit import VisionTransformer



def vit_patch4_32():
    model = VisionTransformer(
        num_classes=10,
        img_size=32,
        patch_size=4,
        embed_dim=192,
        depth=6,
        num_heads=12
    )
    return model

def vit_8layer():
    model = VisionTransformer(
        num_classes=10,
        img_size=32,
        patch_size=4,
        embed_dim=144,
        depth=10,
        num_heads=3
    )
    return model


def deit_ti_32():
    cfg = {
        'url': '',
        'num_classes': 10, 'input_size': (3, 32, 32), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
    }
    model_kwargs = dict(patch_size=4, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=False, default_cfg=cfg, **model_kwargs)
    return model


# def cait_xxs():
#     cfg = {
#         'url': '',
#         'num_classes': 100, 'input_size': (3, 32, 32), 'pool_size': None,
#         'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'patch_embed.proj', 'classifier': 'head',
#     }
#     # model_args = dict(patch_size=4, embed_dim=192, depth=24, num_heads=4, init_scale=1e-5)
#     model_args = dict(patch_size=4, embed_dim=144, depth=24, num_heads=4, init_scale=1e-5)
#     # model = _create_cait('cait_xxs24_224', pretrained=pretrained, **model_args)
#     model = cait._create_cait('cait_xxs', pretrained=False, default_cfgs=cfg, **model_args)
#     return model


if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    model = vit_patch4_32()
    # model = deit_ti_32()
    # model = vit_8layer()

    y = model(x)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000)
    print(y)

