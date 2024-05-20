# idea: select model you use in training and the trainer (the warper for training process)

import logging
import sys

sys.path.append('../../')

import torch
import torchvision.models as models
from torchvision.models.resnet import resnet34, resnet50
from typing import Optional
from torchvision.transforms import Resize

from utils.trainer_cls import ModelTrainerCLS
from models.BaseModel import BaseModel

try:
    from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b3
except:
    logging.warning("efficientnet_b0,b3 fails to import, plz update your torch and torchvision")
try:
    from torchvision.models import mobilenet_v3_large
except:
    logging.warning("mobilenet_v3_large fails to import, plz update your torch and torchvision")

try:
    from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
except:
    logging.warning("vit fails to import, plz update your torch and torchvision")


def partially_load_state_dict(model, state_dict):
    # from https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113, thanks to chenyuntc Yun Chen
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        try:
            param = param.data
            own_state[name].copy_(param)
        except:
            print(f"unmatch: {name}")
            continue


# trainer is cls
def generate_cls_model(
        model_name: str,
        num_classes: int = 10,
        image_size: int = 32,
        dataset_name: str = None,
        **kwargs,
):
    '''
    # idea: aggregation block for selection of classifcation models
    :param model_name:
    :param num_classes:
    :return:
    '''

    logging.debug("image_size ONLY apply for vit!!!\nIf you use vit make sure you set the image size!")

    if model_name == 'vgg_ulp_customized':
        from models.vgg_ulp_customized import VGG_ULP_customized
        init_num_filters = 64
        inter_fc_dim = 384
        nofclasses = 10  # CIFAR10
        net = VGG_ULP_customized(init_num_filters=init_num_filters, inter_fc_dim=inter_fc_dim,nofclasses=nofclasses,
                         nofchannels=3,use_stn=False)
    elif model_name == 'resnet18':
        from torchvision.models.resnet import resnet18
        net = resnet18(num_classes=num_classes, **kwargs)
    elif model_name == 'preactresnet18':
        logging.debug('Make sure you want PreActResNet18, which is NOT resnet18.')
        from models.preact_resnet import PreActResNet18
        if kwargs.get("pretrained", False):
            logging.warning("PreActResNet18 pretrained on cifar10, NOT ImageNet!")
            net_from_cifar10 = PreActResNet18()  # num_classes = num_classes)
            net_from_cifar10.load_state_dict(
                torch.load("../resource/trojannn/clean_preactresnet18.pt", map_location="cpu"
                           )['model_state_dict']
            )
            net = PreActResNet18(num_classes=num_classes)
            partially_load_state_dict(net, net_from_cifar10.state_dict())
        else:
            net = PreActResNet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        net = resnet34(num_classes=num_classes, **kwargs)
    elif model_name == 'resnet50':
        net = resnet50(num_classes=num_classes, **kwargs)
    elif model_name == 'alexnet':
        net = models.alexnet(num_classes=num_classes, **kwargs)
    elif model_name == "vgg11":
        net = models.vgg11(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg16':
        net = models.vgg16(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg19':
        net = models.vgg19(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg19_bn':
        if kwargs.get("pretrained", False):
            net_from_imagenet = models.vgg19_bn(pretrained=True)  # num_classes = num_classes)
            net = models.vgg19_bn(num_classes=num_classes, **{k: v for k, v in kwargs.items() if k != "pretrained"})
            partially_load_state_dict(net, net_from_imagenet.state_dict())
        else:
            net = models.vgg19_bn(num_classes=num_classes, **kwargs)
    elif model_name == 'squeezenet1_0':
        net = models.squeezenet1_0(num_classes=num_classes, **kwargs)
    elif model_name == 'densenet161':
        net = models.densenet161(num_classes=num_classes, **kwargs)
    elif model_name == 'inception_v3':
        net = models.inception_v3(num_classes=num_classes, **kwargs)
    elif model_name == 'googlenet':
        net = models.googlenet(num_classes=num_classes, **kwargs)
    elif model_name == 'shufflenet_v2_x1_0':
        net = models.shufflenet_v2_x1_0(num_classes=num_classes, **kwargs)
    elif model_name == 'mobilenet_v2':
        net = models.mobilenet_v2(num_classes=num_classes, **kwargs)
    elif model_name == 'mobilenet_v3_large':
        net = models.mobilenet_v3_large(num_classes=num_classes, **kwargs)
    elif model_name == 'resnext50_32x4d':
        net = models.resnext50_32x4d(num_classes=num_classes, **kwargs)
    elif model_name == 'wide_resnet50_2':
        net = models.wide_resnet50_2(num_classes=num_classes, **kwargs)
    elif model_name == 'mnasnet1_0':
        net = models.mnasnet1_0(num_classes=num_classes, **kwargs)
    elif model_name == 'efficientnet_b0':
        net = efficientnet_b0(num_classes=num_classes, **kwargs)
    elif model_name == 'efficientnet_b3':
        net = efficientnet_b3(num_classes=num_classes, **kwargs)
    elif model_name.startswith("vit"):
        logging.debug("All vit model use the default pretrain and resize to match the input shape!")
        if model_name == 'vit_b_16':
            net = vit_b_16(
                pretrained=True,
                **{k: v for k, v in kwargs.items() if k != "pretrained"}
            )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features=num_classes, bias=True)
            net = torch.nn.Sequential(
                Resize((224, 224)),
                net,
            )
        elif model_name == 'vit_b_32':
            net = vit_b_32(
                pretrained=True,
                **{k: v for k, v in kwargs.items() if k != "pretrained"}
            )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features=num_classes, bias=True)
            net = torch.nn.Sequential(
                Resize((224, 224)),
                net,
            )
        elif model_name == 'vit_l_16':
            net = vit_l_16(
                pretrained=True,
                **{k: v for k, v in kwargs.items() if k != "pretrained"}
            )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features=num_classes, bias=True)
            net = torch.nn.Sequential(
                Resize((224, 224)),
                net,
            )
        elif model_name == 'vit_l_32':
            net = vit_l_32(
                pretrained=True,
                **{k: v for k, v in kwargs.items() if k != "pretrained"}
            )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features=num_classes, bias=True)
            net = torch.nn.Sequential(
                Resize((224, 224)),
                net,
            )
    elif model_name == 'densenet121':
        net = models.densenet121(num_classes=num_classes, **kwargs)
    elif model_name == 'resnext29':
        from models.resnext import ResNeXt29_2x64d
        net = ResNeXt29_2x64d(num_classes=num_classes)
    elif model_name == 'senet18':
        from models.senet import SENet18
        net = SENet18(num_classes=num_classes)
    elif model_name == "convnext_tiny":
        logging.debug("All convnext model use the default pretrain!")
        from torchvision.models import convnext_tiny
        net_from_imagenet = convnext_tiny(pretrained=True, )  # num_classes = num_classes)
        net = convnext_tiny(num_classes=num_classes, **{k: v for k, v in kwargs.items() if k != "pretrained"})
        partially_load_state_dict(net, net_from_imagenet.state_dict())
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    if dataset_name == "cifar10":
        mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    elif dataset_name == 'cifar100':
        '''get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151'''
        mean, std = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
    elif dataset_name in ["mnist", "fmnist"]:
        mean, std = [0.5], [0.5]
    elif dataset_name == 'tiny':
        mean, std = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        mean, std = [0, 0, 0], [1, 1, 1]
    elif dataset_name in ["imagenet", "pubfig"]:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise Exception("Invalid Dataset")

    BaseModel(backbone=net, normalize=True, meta_data=model_name, mean=mean, std=std)
    return net


def generate_cls_trainer(
        model,
        attack_name: Optional[str] = None,
        amp: bool = False,
):
    '''
    # idea: The warpper of model, which use to receive training settings.
        You can add more options for more complicated backdoor attacks.

    :param model:
    :param attack_name:
    :return:
    '''

    trainer = ModelTrainerCLS(
        model=model,
        amp=amp,
    )

    return trainer
