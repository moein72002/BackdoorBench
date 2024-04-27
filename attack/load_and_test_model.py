import argparse
import os,sys

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import logging
import time
import yaml

from attack.badnet import BadNet
from attack.blended import Blended

from utils.trainer_cls import given_dataloader_test
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_transform, get_num_classes, get_input_shape
from utils.save_load_attack import load_attack_result

import copy
from utils.BAD.data_utils.loaders import get_ood_loader

import torch
import torch.nn as nn

def check_zero_weights(model):
    total_weights = 0
    zero_weights = 0
    for param in model.parameters():
        total_weights += param.numel()
        zero_weights += (param == 0).sum().item()
    print(f"Total weights: {total_weights}, Zero weights: {zero_weights} ({100 * zero_weights / total_weights:.2f}%)")

def prune_filters(model, input1, input2, prune_ratio=0.3, prune_all_layers=False):
    activation_diffs = {}

    def forward_hook(module, inp, out):
        if hasattr(module, 'activations'):
            # Calculate and store absolute differences between activations
            activation_diffs[module] = torch.abs(module.activations - out)
        # Store current output for next invocation or comparison
        module.activations = out.detach()

    hooks = []
    # Include both Conv2d and Linear layers for potential pruning
    layers = [module for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
    # Determine which layers to register hooks based on the prune_all_layers flag
    if prune_all_layers:
        target_layers = layers  # All layers will be considered for pruning
    else:
        target_layers = layers[-2:-1]  # Only the second-to-last layer is targeted

    # Register hooks on the selected layers
    for module in target_layers:
        hook = module.register_forward_hook(forward_hook)
        hooks.append(hook)

    # Perform model inference to trigger the hooks and calculate activation differences
    model.eval()
    with torch.no_grad():
        _ = model(input1)
        _ = model(input2)

    # Remove hooks after use to clean up
    for hook in hooks:
        hook.remove()

    total_filters = 0
    pruned_filters = 0
    # Process each layer that had a hook registered and had activation differences computed
    for module in target_layers:
        if module in activation_diffs:
            diffs = activation_diffs[module]
            if isinstance(module, nn.Conv2d):
                num_filters = module.out_channels
                importance_scores = diffs.mean(dim=[0, 2, 3])  # Mean over batch, height, and width for Conv2d
            elif isinstance(module, nn.Linear):
                num_filters = module.out_features
                importance_scores = diffs.mean(dim=0)  # Mean over batch for Linear

            total_filters += num_filters
            threshold = torch.quantile(importance_scores, 1 - prune_ratio)
            prune_mask = importance_scores > threshold
            pruned_filters += prune_mask.sum().item()

            # Zeroing out the weights and optionally biases of the selected filters
            module.weight.data[prune_mask, ...] = 0
            if module.bias is not None:
                module.bias.data[prune_mask] = 0

    print(f"Total filters: {total_filters}, Pruned filters: {pruned_filters}")
    return model

def evaluate_model_with_prune_ratio_list(args, result_dict):
    model = generate_cls_model(args.model, args.num_classes)
    model.load_state_dict(result_dict["model"])
    model.to(args.device)

    test_ood_loader = get_ood_loader("cifar10", 'rot', in_source='train', sample_num=200,
                                     batch_size=512)  # , out_filter_labels=[0, 1])

    attack_norm = ['linf', 'l2'][0]

    # Loading attack
    if attack_norm == 'linf':
        from utils.BAD.attacks.ood.pgdlinf import PGD as Attack
    elif attack_norm == 'l2':
        from utils.BAD.attacks.ood.pgdl2 import PGD as Attack
    else:
        raise NotImplementedError("This norm for attacks is not supported")

    attack_eps = 1.7 / 255
    attack_steps = 10
    attack_alpha = 2.5 * attack_eps / attack_steps

    attack_params = {
        'model': args.model,
        'target_class': None,
        'eps': attack_eps,
        'steps': attack_steps,
        'alpha': attack_alpha,
    }

    attack = Attack(**attack_params)

    args.model.eval()

    # Assuming `attack` function and `test_ood_loader` are defined elsewhere

    for inputs, targets in test_ood_loader:
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        print("starting attack")
        attacked_inputs = attack(inputs, targets)
        break  # Assuming we use only one batch for the example

    pruning_results = {}

    # pruned_model = prune_filters(copy.deepcopy(model), inputs, attacked_inputs, prune_ratio=0.01)

    for prune_ratio in args.prune_ratio_list:
        print()
        pruned_model = prune_filters(copy.deepcopy(args.model), inputs, attacked_inputs, prune_ratio=prune_ratio, prune_all_layers=True)

        # model = resnet18(pretrained=True)
        print("model")
        check_zero_weights(args.model)

        print("pruned_model")
        check_zero_weights(pruned_model)

        print(f"prune_ratio: {prune_ratio}")
        test_acc, test_asr = eval_model(args, result_dict, pruned_model)

        pruning_results[prune_ratio] = {"test_acc": test_acc, "test_asr": test_asr}

    print(f"pruning_results: {pruning_results}")


def set_result(args, result_file_path):
    result = load_attack_result(args, result_file_path, args.attack, args.dataset_path)

    return result

def eval_model(args, result_dict, model):
    test_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
    data_bd_testset = result_dict['bd_test']
    data_bd_testset.wrap_img_transform = test_tran
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size,
                                                 num_workers=args.num_workers, drop_last=False, shuffle=True,
                                                 pin_memory=args.pin_memory)

    data_clean_testset = result_dict['clean_test']
    data_clean_testset.wrap_img_transform = test_tran
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, drop_last=False, shuffle=True,
                                                    pin_memory=args.pin_memory)


    clean_test_loss_avg_over_batch, \
    bd_test_loss_avg_over_batch, \
    ra_test_loss_avg_over_batch, \
    test_acc, \
    test_asr, \
    test_ra = eval_step(
        model,
        data_clean_loader,
        data_bd_loader,
        args,
    )

    results_dict = {
        "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
        "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
        "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
        "test_acc": test_acc,
        "test_asr": test_asr,
        "test_ra": test_ra,
    }

    print(results_dict)

    return test_acc, test_asr

def set_logger(args):
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()

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

def eval_step(
        netC,
        clean_test_dataloader,
        bd_test_dataloader,
        args,
):

    clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
        netC,
        clean_test_dataloader,
        criterion=torch.nn.CrossEntropyLoss(),
        non_blocking=args.non_blocking,
        device=args.device,
        verbose=0,
    )
    clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
    test_acc = clean_metrics['test_acc']
    bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
        netC,
        bd_test_dataloader,
        criterion=torch.nn.CrossEntropyLoss(),
        non_blocking=args.non_blocking,
        device=args.device,
        verbose=0,
    )
    bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
    test_asr = bd_metrics['test_acc']

    bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = True  # change to return the original label instead
    ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
        netC,
        bd_test_dataloader,
        criterion=torch.nn.CrossEntropyLoss(),
        non_blocking=args.non_blocking,
        device=args.device,
        verbose=0,
    )
    ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
    test_ra = ra_metrics['test_acc']
    bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = False  # switch back

    return clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra


def add_bd_yaml_to_args(args):
    with open(args.bd_yaml_path, 'r') as f:
        mix_defaults = yaml.safe_load(f)
    mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = mix_defaults

def add_yaml_to_args(args):
    with open(args.yaml_path, 'r') as f:
        clean_defaults = yaml.safe_load(f)
    clean_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = clean_defaults

def process_args(args):
    args.terminal_info = sys.argv
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    return args

def get_attack_by_name(attack_name):
    if attack_name == "badnet":
        attack = BadNet()
    elif attack_name == "blended":
        attack = Blended()
    else:
        attack = None

    return attack

def set_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--attack', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--yaml_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--prune_ratio_list_arg', type=float, nargs="+")
    parser.add_argument('--pratio', type=float)
    # parser.add_argument('--prune_ratio', type=float)
    return parser

def add_common_attack_args(parser):
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    return parser

def set_badnet_bd_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_common_attack_args(parser)

    parser.add_argument("--patch_mask_path", type=str, default="../resource/badnet/trigger_image.png")
    parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/badnet/default.yaml',
                        help='path for yaml file provide additional default attributes')
    return parser

def set_blended_bd_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_common_attack_args(parser)
    parser.add_argument("--attack_trigger_img_path", type=str, )
    parser.add_argument("--attack_train_blended_alpha", type=float, )
    parser.add_argument("--attack_test_blended_alpha", type=float, )
    parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/blended/default.yaml',
                        help='path for yaml file provide additional default attributes')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = set_args(parser)
    args = parser.parse_args()
    if args.attack == "badnet":
        set_badnet_bd_args(parser)
    elif args.attack == "blended":
        set_blended_bd_args(parser)
    args = parser.parse_args()
    add_bd_yaml_to_args(args)
    add_yaml_to_args(args)
    args = process_args(args)
    print(args.__dict__)
    result_dict = set_result(args, args.result_file)

    # set_logger(args)
    # eval_model(args, result_dict)