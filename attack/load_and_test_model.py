import argparse
import os,sys
import torch

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

def set_result(args, result_file_path):
    result = load_attack_result(result_file_path, args.attack)

    return result

def eval_model(args, result_dict):
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

    model = generate_cls_model(args.model, args.num_classes)

    model.load_state_dict(result_dict["model"])

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    args = parser.parse_args()
    attack = get_attack_by_name(args.attack)
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    add_bd_yaml_to_args(args)
    add_yaml_to_args(args)
    args = process_args(args)
    result_dict = set_result(args, args.result_file)
    eval_model(args, result_dict)
    set_logger(args)
    eval_model(args, result_dict)