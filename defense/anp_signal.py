import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense

from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
from collections import OrderedDict
import copy

import utils.defense_utils.anp.anp_model as anp_model

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import BackdoorModelTrainer, Metric_Aggregator, ModelTrainerCLS, ModelTrainerCLS_v2, PureCleanModelTrainer, general_plot_for_epoch
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model, partially_load_state_dict
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from BAD.data.loaders import get_ood_loader
from attack.load_and_test_model import set_badnet_bd_args, set_blended_bd_args, add_bd_yaml_to_args, add_yaml_to_args, process_args



### anp function
def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.include_noise()
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.include_noise()



def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.exclude_noise()
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, anp_model.NoisyBatchNorm2d) or isinstance(module, anp_model.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)
        if isinstance(module, anp_model.NoiseLayerNorm2d) or isinstance(module, anp_model.NoiseLayerNorm):
            module.reset(rand_init=rand_init, eps=args.anp_eps)

def anp_model_noise_train(args, model, criterion, noise_opt, data_loader):
    model.train()
    nb_samples = 0
    for i, (images, labels, *additional_info) in enumerate(data_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(model, rand_init=True)

            for j in range(args.anp_steps):

                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # TODO: check below line, maybe I should comment it
        exclude_noise(model)

        # clip_mask(model)


def test(args, model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels, *additional_info) in enumerate(data_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def get_anp_network(
    model_name: str,
    num_classes: int = 10,
    **kwargs,
):
    
    if model_name == 'preactresnet18':
        from utils.defense_utils.anp.anp_model.preact_anp import PreActResNet18
        net = PreActResNet18(num_classes = num_classes, **kwargs)
    elif model_name == 'resnet18':
        from torchvision.models.resnet import resnet18
        net = resnet18(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg19_bn':
        net = anp_model.vgg_anp.vgg19_bn(num_classes = num_classes,  **kwargs)
    elif model_name == 'densenet161':
        net = anp_model.den_anp.densenet161(num_classes= num_classes, **kwargs)
    elif model_name == 'mobilenet_v3_large':
        net = anp_model.mobilenet_anp.mobilenet_v3_large(num_classes= num_classes, **kwargs)
    elif model_name == 'efficientnet_b3':
        net = anp_model.eff_anp.efficientnet_b3(num_classes= num_classes, **kwargs)
    elif model_name == 'convnext_tiny':
        # net_from_imagenet = convnext_tiny(pretrained=True) #num_classes = num_classes)
        try :
            net = anp_model.conv_anp.convnext_tiny(num_classes= num_classes, **{k:v for k,v in kwargs.items() if k != "pretrained"})
        except :
            net = anp_model.conv_new_anp.convnext_tiny(num_classes= num_classes, **{k:v for k,v in kwargs.items() if k != "pretrained"})
        # partially_load_state_dict(net, net_from_imagenet.state_dict())
        # net = anp_model.convnext_anp.convnext_tiny(num_classes= num_classes, **kwargs)
    elif model_name == 'vit_b_16':
        try :
            from torchvision.transforms import Resize
            net = anp_model.vit_anp.vit_b_16(
                    pretrained = False,
                    # **{k: v for k, v in kwargs.items() if k != "pretrained"}
                )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                    Resize((224, 224)),
                    net,
                )
        except :
            from torchvision.transforms import Resize
            net = anp_model.vit_new_anp.vit_b_16(
                    pretrained = False,
                    # **{k: v for k, v in kwargs.items() if k != "pretrained"}
                )
            net.heads.head = torch.nn.Linear(net.heads.head.in_features, out_features = num_classes, bias=True)
            net = torch.nn.Sequential(
                    Resize((224, 224)),
                    net,
                )
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)





class anp_signal(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args, args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', default = False, type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        # parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        # parser.add_argument('--yaml_path', type=str, default="./config/defense/anp/config.yaml", help='the path of yaml')

        #set the parameter for the anp defense
        parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--print_every', type=int, help='print results every few iterations')
        parser.add_argument('--nb_iter', type=int, help='the number of iterations for training')

        parser.add_argument('--anp_eps', type=float)
        parser.add_argument('--anp_steps', type=int)
        parser.add_argument('--anp_alpha', type=float)

        parser.add_argument('--pruning_by', type=str, choices=['number', 'threshold'])
        parser.add_argument('--pruning_max', type=float, help='the maximum number/threshold for pruning')
        parser.add_argument('--pruning_step', type=float, help='the step size for evaluating the pruning')

        parser.add_argument('--pruning_number', type=float, help='the default number/threshold for pruning')

        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--prune_all_layers', type=bool, default=False)



    def set_result(self, args, result_file_path):
        self.result = load_attack_result(args, result_file_path, args.attack, args.dataset_path)
        
    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        args.log = ""

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
   
    def set_devices(self):
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

    def check_zero_weights(self, model):
        total_weights = 0
        zero_weights = 0
        for param in model.parameters():
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()
        print(
            f"Total weights: {total_weights}, Zero weights: {zero_weights} ({100 * zero_weights / total_weights:.2f}%)")

    def prune_filters_based_on_noisy_model(self, model1, model2, input, prune_ratio=0.3, prune_all_layers=False):
        model1 = copy.deepcopy(model1)
        model2 = copy.deepcopy(model2)

        if not prune_ratio > 0.0:
            return model1, model2

        # Dictionary to store activations keyed by layer names
        activations_store = {}
        activation_diffs = {}

        def get_layer_path(module, prefix=''):
            """ Recursively get the path for a layer within the model's hierarchy. """
            layer_path = {module: prefix}
            for name, child in module.named_children():
                child_path = get_layer_path(child, prefix=f"{prefix}/{name}" if prefix else name)
                layer_path.update(child_path)
            return layer_path

        # Get layer paths
        layer_paths1 = get_layer_path(model1)
        layer_paths2 = get_layer_path(model2)

        def forward_hook1(module, inp, out):
            layer_name = layer_paths1[module]
            activations_store[layer_name] = out.detach()
            print(f"Activations1 recorded for layer: {layer_name}")

        def forward_hook2(module, inp, out):
            layer_name = layer_paths2[module]
            if layer_name in activations_store:
                activation_diffs[layer_name] = torch.abs(activations_store[layer_name] - out)
                print(f"Difference calculated for layer: {layer_name}")
            else:
                print(f"No previous activations found for layer: {layer_name}")

        # Register hooks for both models based on the boolean input
        hooks1 = []
        hooks2 = []
        if prune_all_layers:
            for module in layer_paths1:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hooks1.append(module.register_forward_hook(forward_hook1))
            for module in layer_paths2:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    hooks2.append(module.register_forward_hook(forward_hook2))
        else:
            # Find the second-to-last convolutional or linear layer
            eligible_layers1 = [m for m in model1.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            eligible_layers2 = [m for m in model2.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            if len(eligible_layers1) > 1 and len(eligible_layers2) > 1:
                module1 = eligible_layers1[-2]
                module2 = eligible_layers2[-2]
                hooks1.append(module1.register_forward_hook(forward_hook1))
                hooks2.append(module2.register_forward_hook(forward_hook2))

        model1.eval()
        model2.eval()
        with torch.no_grad():
            _ = model1(input)
            _ = model2(input)

        # Remove hooks
        for hook in hooks1:
            hook.remove()
        for hook in hooks2:
            hook.remove()

        total_filters = 0
        pruned_filters = 0
        # Apply pruning based on calculated differences
        for layer_name, diffs in activation_diffs.items():
            if diffs.dim() == 4:  # Conv2d layers
                importance_scores = diffs.mean(dim=[0, 2, 3])
            elif diffs.dim() == 2:  # Linear layers
                importance_scores = diffs.mean(dim=0)
            else:
                continue

            num_filters = importance_scores.size(0)
            total_filters += num_filters
            threshold = torch.quantile(importance_scores, 1 - prune_ratio)
            prune_mask = importance_scores > threshold
            pruned_filters += prune_mask.sum().item()

            # Zero out weights based on the pruning mask
            module1 = next(m for m in model1.modules() if layer_paths1[m] == layer_name)
            module2 = next(m for m in model2.modules() if layer_paths2[m] == layer_name)
            module1.weight.data[prune_mask, ...] = 0
            module2.weight.data[prune_mask, ...] = 0
            if module1.bias is not None:
                module1.bias.data[prune_mask] = 0
            if module2.bias is not None:
                module2.bias.data[prune_mask] = 0

            print(f"Pruning applied to layers: {layer_name} | Pruned filters in this layer: {prune_mask.sum().item()}")

        print(f"Total filters: {total_filters}, Pruned filters: {pruned_filters}")
        return model1, model2

    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        args = self.args
        result = self.result
        # a. train the mask of old model
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length) 
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_clean = self.result['clean_train']
        data_set_clean.wrapped_dataset = data_set_without_tran
        data_set_clean.wrap_img_transform = train_tran
        # data_set_clean.wrapped_dataset.getitem_all = False
        random_sampler = RandomSampler(data_source=data_set_clean, replacement=True,
                                    num_samples=args.print_every * args.batch_size)
        clean_val_loader = DataLoader(data_set_clean, batch_size=args.batch_size,
                                    shuffle=False, sampler=random_sampler, num_workers=0)
        
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        # data_bd_testset.wrapped_dataset.getitem_all = False
        poison_test_loader = DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        clean_test_loader = DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = clean_test_loader
        test_dataloader_dict["bd_test_dataloader"] = poison_test_loader
        state_dict = self.result['model']
        noisy_model = get_anp_network(args.model, num_classes=args.num_classes, norm_layer=anp_model.NoisyBatchNorm2d)
        load_state_dict(noisy_model, orig_state_dict=state_dict)
        noisy_model = noisy_model.to(args.device)
        criterion = torch.nn.CrossEntropyLoss().to(args.device)

        parameters = list(noisy_model.named_parameters())
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

        logging.info('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
        for i in range(nb_repeat):
            start = time.time()
            anp_model_noise_train(args, model=noisy_model, criterion=criterion, data_loader=clean_val_loader, noise_opt=noise_optimizer)
            cl_test_loss, cl_test_acc = test(args, model=noisy_model, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss, po_test_acc = test(args, model=noisy_model, criterion=criterion, data_loader=poison_test_loader)
            end = time.time()
            logging.info('{} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                (i + 1) * args.print_every, end - start, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc))

        # b. prune the model
        original_model = generate_cls_model(args.model,args.num_classes)
        original_model.load_state_dict(result['model'])
        original_model.to(args.device)

        test_ood_loader = get_ood_loader("cifar10", 'rot', in_source='train', sample_num=200,
                                         batch_size=512)  # , out_filter_labels=[0, 1])

        original_model.eval()

        # Assuming `attack` function and `test_ood_loader` are defined elsewhere

        for inputs, targets in test_ood_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            break  # Assuming we use only one batch for the example

        pruned_model, _ = self.prune_filters_based_on_noisy_model(original_model, noisy_model, inputs, prune_ratio=args.anp_signal_prune_ratio, prune_all_layers=args.prune_all_layers)

        # model = resnet18(pretrained=True)
        print("model")
        self.check_zero_weights(original_model)

        print("pruned_model")
        self.check_zero_weights(pruned_model)

        agg = self.evaluate_model(original_model, "original_model", clean_val_loader, criterion, test_dataloader_dict)
        agg = self.evaluate_model(pruned_model, "pruned_model", clean_val_loader, criterion, test_dataloader_dict)

        agg.to_dataframe().to_csv(f"{args.save_path}anp_df_summary.csv")
        result = {}
        result['model'] = original_model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=original_model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def evaluate_model(self, model, model_name, clean_val_loader, criterion, test_dataloader_dict):
        self.set_trainer(model)
        self.trainer.set_with_dataloader(
            ### the train_dataload has nothing to do with the backdoor defense
            train_dataloader=clean_val_loader,
            test_dataloader_dict=test_dataloader_dict,

            criterion=criterion,
            optimizer=None,
            scheduler=None,
            device=self.args.device,
            amp=self.args.amp,

            frequency_save=self.args.frequency_save,
            save_folder_path=self.args.save_path,
            save_prefix='anp',

            prefetch=self.args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",
            non_blocking=self.args.non_blocking,
        )
        agg = Metric_Aggregator()
        clean_test_loss_avg_over_batch, \
        bd_test_loss_avg_over_batch, \
        test_acc, \
        test_asr, \
        test_ra = self.trainer.test_current_model(
            test_dataloader_dict, self.args.device,
        )
        agg({
            "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
            "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra,
        })
        print(f"{model_name}:")
        results_dict = {
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra,
        }
        print(f"prune_ratio: {args.anp_signal_prune_ratio}")
        print(results_dict)
        return agg

    def defense(self,result_file):
        self.set_result(args, args.result_file)
        self.set_logger()
        result = self.mitigation()
        return result

def set_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--attack', type=str)
    # parser.add_argument('--dataset', type=str)
    # parser.add_argument('--dataset_path', type=str)
    # parser.add_argument('--model', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--yaml_path', type=str)
    # parser.add_argument('--bd_yaml_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--anp_signal_prune_ratio', type=float)
    # parser.add_argument('--pratio', type=float)
    return parser
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = set_args(parser)
    anp_signal.add_arguments(parser)
    args = parser.parse_args()
    if args.attack == "badnet":
        set_badnet_bd_args(parser)
    elif args.attack == "blended":
        set_blended_bd_args(parser)
    args = parser.parse_args()
    add_bd_yaml_to_args(args)
    add_yaml_to_args(args)
    args.yaml_path = f"../config/attack/prototype/{args.dataset}.yaml"
    add_yaml_to_args(args)
    args = process_args(args)
    print(f"args.__dict__: {args.__dict__}")
    anp_method = anp_signal(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'

    result = anp_method.defense(args.result_file)