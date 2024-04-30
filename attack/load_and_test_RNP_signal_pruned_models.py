import argparse
import os,sys
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import logging
import time
import yaml

from utils.trainer_cls import given_dataloader_test
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_transform, get_num_classes, get_input_shape
from utils.save_load_attack import load_attack_result

import copy

import torch
import torch.nn as nn

from attack.load_and_test_model import set_badnet_bd_args, set_blended_bd_args, set_sig_bd_args, set_wanet_bd_args, \
    set_bpp_bd_args, add_bd_yaml_to_args, add_yaml_to_args, process_args, visualize_results, eval_model

def check_zero_weights(model):
    total_weights = 0
    zero_weights = 0
    for param in model.parameters():
        total_weights += param.numel()
        zero_weights += (param == 0).sum().item()
    print(f"Total weights: {total_weights}, Zero weights: {zero_weights} ({100 * zero_weights / total_weights:.2f}%)")


def visualize_results(prune_results_dict, fig_name):
    # Extracting data for plotting
    noise_levels = list(prune_results_dict.keys())
    test_acc = [prune_results_dict[n]['test_acc'] for n in noise_levels]
    test_asr = [prune_results_dict[n]['test_asr'] for n in noise_levels]

    # Creating the plot
    plt.figure(figsize=(10, 5))
    plt.plot(noise_levels, test_acc, label='Test Accuracy', marker='o')
    plt.plot(noise_levels, test_asr, label='Test Attack Success Rate', marker='x')
    plt.xlabel('Noise Level')
    plt.ylabel('Percentage')
    plt.title(fig_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_name)
    print(f"Chart saved in {fig_name}")


def evaluate_model_with_prune_ratio_list(args, result_dict):
    model = generate_cls_model(args.model, args.num_classes)
    model.load_state_dict(result_dict["model"])
    model.to(args.device)



    model.eval()


    pruning_results = {}

    # pruned_model = prune_filters(copy.deepcopy(model), inputs, attacked_inputs, prune_ratio=0.01)

    for prune_ratio in args.prune_ratio_list_arg:
        print()
        pruned_model = generate_cls_model(args.model, args.num_classes)
        file_path = f"/kaggle/input/rnp-{args.attack}-{args.model}-{args.dataset}/RNP/weights/RNP_unlearn_signal_pruned_model_{args.attack}_{args.model}_{args.dataset}_target{args.attack_target}_prune_ratio_{prune_ratio}.pt"
        print(f"file_path: {file_path}")
        pruned_model.load_state_dict(torch.load(file_path)["model"])
        pruned_model.to(args.device)

        # model = resnet18(pretrained=True)
        print("model")
        check_zero_weights(model)

        print("pruned_model")
        check_zero_weights(pruned_model)

        print(f"prune_ratio: {prune_ratio}")
        test_acc, test_asr = eval_model(args, result_dict, pruned_model)

        pruning_results[prune_ratio] = {"test_acc": test_acc, "test_asr": test_asr}

    print(f"pruning_results: {pruning_results}")

    fig_name = f"{args.model}_{args.dataset}_{args.attack}_target{args.attack_target}.pdf"
    visualize_results(pruning_results, fig_name)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = set_args(parser)
    args = parser.parse_args()
    if args.attack == "badnet":
        set_badnet_bd_args(parser)
    elif args.attack == "blended":
        set_blended_bd_args(parser)
    elif args.attack == "wanet":
        set_wanet_bd_args(parser)
    elif args.attack == "sig":
        set_sig_bd_args(parser)
    elif args.attack == "bpp":
        set_bpp_bd_args(parser)
    args = parser.parse_args()
    add_bd_yaml_to_args(args)
    add_yaml_to_args(args)
    args = process_args(args)
    print(args.__dict__)
    result_dict = set_result(args, args.result_file)
    evaluate_model_with_prune_ratio_list(args, result_dict)
    # set_logger(args)
    # eval_model(args, result_dict)