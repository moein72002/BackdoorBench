'''
This script aims to save and load the attack result as a bridge between attack and defense files.

Model, clean data, backdoor data and all infomation needed to reconstruct will be saved.

Note that in default, only the poisoned part of backdoor dataset will be saved to save space.

Jun 12th update:
    change save_load to adapt to alternative save method.
    But notice that this method assume the bd_train after reconstruct MUST have the SAME length with clean_train.

'''
import copy
import logging, time

from typing import Optional
import torch, os
from torchvision.datasets import DatasetFolder, ImageFolder

from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
import numpy as np
from copy import deepcopy
from pprint import pformat
from typing import Union

import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.aggregate_block.dataset_and_transform_generate import get_dataset_denormalization

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate

def summary_dict(input_dict):
    '''
    Input a dict, this func will do summary for it.
    deepcopy to make sure no influence for summary
    :return:
    '''
    input_dict = deepcopy(input_dict)
    summary_dict_return = dict()
    for k,v in input_dict.items():
        if isinstance(v, dict):
            summary_dict_return[k] = summary_dict(v)
        elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            summary_dict_return[k] = {
                'shape':v.shape,
                'min':v.min(),
                'max':v.max(),
            }
        elif isinstance(v, list):
            summary_dict_return[k] = {
                'len':v.__len__(),
                'first ten':v[:10],
                'last ten':v[-10:],
            }
        else:
            summary_dict_return[k] = v
    return  summary_dict_return

def sample_pil_imgs(pil_image_list, save_folder, num = 5,):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    select_index = np.random.choice(
        len(pil_image_list),
        num,
    ).tolist() + np.arange(num).tolist() + np.arange(len(pil_image_list) - num, len(pil_image_list)).tolist()

    for ii in select_index :
        if 0 <= ii < len(pil_image_list):
            pil_image_list[ii].save(f"{save_folder}/{ii}.png")

def save_attack_result(
    model_name : str,
    num_classes : int,
    model : dict, # the state_dict
    img_size : Union[list, tuple],
    dataset_name : str,
    save_path : str,
    poison_rate : float,
    model_number: int,
    target_class: int,
    test_acc_list: list,
    test_asr_list: list,
    test_ra_list: list
):
    '''

    main idea is to loop through the backdoor train and test dataset, and match with the clean dataset
    by remove replicated parts, this function can save the space.

    WARNING: keep all dataset with shuffle = False, same order of data samples is the basic of this function !!!!

    :param model_name : str,
    :param num_classes : int,
    :param model : dict, # the state_dict
    :param data_path : str,
    :param img_size : list, like [32,32,3]
    :param clean_data : str, clean dataset name
    :param bd_train : torch.utils.data.Dataset, # dataset without transform !!
    :param bd_test : torch.utils.data.Dataset, # dataset without transform
    :param save_path : str,
    '''

    save_dict = {
            'model_name': model_name,
            'num_classes' : num_classes,
            'model': model,
            'img_size' : img_size,
            'dataset_name': dataset_name,
            'poison_rate': poison_rate,
            'model_number': model_number,
            'target_class': target_class,
            'test_acc_list': test_acc_list,
            'test_asr_list': test_asr_list,
            'test_ra_list': test_ra_list,
        }

    logging.info(f"saving...")
    # logging.debug(f"location : {save_path}/attack_result.pt") #, content summary :{pformat(summary_dict(save_dict))}")

    torch.save(
        save_dict,
        f'{save_path}/{model_name}_{dataset_name}_model{model_number}.pt',
    )

    logging.info("Saved, folder path: {}".format(save_path))

def save_defense_result(
    model_name : str,
    num_classes : int,
    model : dict, # the state_dict
    save_path : str,
):
    '''

    main idea is to loop through the backdoor train and test dataset, and match with the clean dataset
    by remove replicated parts, this function can save the space.

    WARNING: keep all dataset with shuffle = False, same order of data samples is the basic of this function !!!!

    :param model_name : str,
    :param num_classes : int,
    :param model : dict, # the state_dict
    :param save_path : str,
    '''

    save_dict = {
            'model_name': model_name,
            'num_classes' : num_classes,
            'model': model,
        }

    logging.info(f"saving...")
    logging.debug(f"location : {save_path}/defense_result.pt") #, content summary :{pformat(summary_dict(save_dict))}")

    torch.save(
        save_dict,
        f'{save_path}/defense_result.pt',
    )


class Args:
    pass

def load_attack_result(
    args,
    save_path : str,
    attack : str,
    dataset_path : str
):
    '''
    This function first replicate the basic steps of generate models and clean train and test datasets
    then use the index given in files to replace the samples should be poisoned to re-create the backdoor train and test dataset

    save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path!!!
    save_path : the path of "attack_result.pt"
    '''
    load_file = torch.load(save_path)

    if all(key in load_file for key in ['model_name',
        'num_classes',
        'model',
        'img_size',
        'dataset_name',
        'poison_rate',
        'model_number',
        'target_class',
        ]):

        logging.info('key match for attack_result, processing...')

        # model = generate_cls_model(load_file['model_name'], load_file['num_classes'])
        # model.load_state_dict(load_file['model'])

        # attack_setting = Args()

        args.model = load_file['model_name']

        args.dataset = load_file['dataset_name']
        args.pratio = load_file['poison_rate']
        args.attack_target = load_file['target_class']
        args.attack = attack

        # convert the relative/abs path in attack result to abs path for defense
        # clean_setting.dataset_path = load_file['data_path']
        logging.warning("save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path")
        args.dataset_path = dataset_path

        args.img_size = load_file['img_size']

        if attack in ['badnet', 'blended']:
            clean_train_dataset_with_transform, \
            clean_test_dataset_with_transform, \
            bd_train_dataset_with_transform, \
            bd_test_dataset_with_transform = badnet_stage1_non_training_data_prepare(args)
        elif attack == "sig":
            clean_train_dataset_with_transform, \
            clean_test_dataset_with_transform, \
            bd_train_dataset_with_transform, \
            bd_test_dataset_with_transform = sig_stage1_non_training_data_prepare(args)
        elif attack == "wanet":
            clean_train_dataset_with_transform, \
            clean_test_dataset_with_transform, \
            bd_train_dataset_with_transform, \
            bd_test_dataset_with_transform = wanet_stage1_non_training_data_prepare(args)


        new_dict = copy.deepcopy(load_file['model'])
        for k, v in load_file['model'].items():
            if k.startswith('module.'):
                del new_dict[k]
                new_dict[k[7:]] = v

        load_file['model'] = new_dict
        load_dict = {
                'model_name': load_file['model_name'],
                'model': load_file['model'],
                'clean_train': clean_train_dataset_with_transform,
                'clean_test' : clean_test_dataset_with_transform,
                'bd_train': bd_train_dataset_with_transform,
                'bd_test': bd_test_dataset_with_transform,
            }

        print(f"loading...")

        return load_dict

    else:
        logging.info(f"loading...")
        logging.debug(f"location : {save_path}, content summary :{pformat(summary_dict(load_file))}")
        return load_file

def benign_prepare(args):
    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transform, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform = dataset_and_transform_generate(args)

    logging.debug("dataset_and_transform_generate done")

    clean_train_dataset_with_transform = dataset_wrapper_with_transform(
        train_dataset_without_transform,
        train_img_transform,
        train_label_transform
    )

    clean_train_dataset_targets = get_labels(train_dataset_without_transform)

    clean_test_dataset_with_transform = dataset_wrapper_with_transform(
        test_dataset_without_transform,
        test_img_transform,
        test_label_transform,
    )

    clean_test_dataset_targets = get_labels(test_dataset_without_transform)

    return train_dataset_without_transform, \
           train_img_transform, \
           train_label_transform, \
           test_dataset_without_transform, \
           test_img_transform, \
           test_label_transform, \
           clean_train_dataset_with_transform, \
           clean_train_dataset_targets, \
           clean_test_dataset_with_transform, \
           clean_test_dataset_targets

def get_labels(given_dataset):
    if isinstance(given_dataset, DatasetFolder) or isinstance(given_dataset, ImageFolder):
        logging.debug("get .targets")
        return given_dataset.targets
    else:
        logging.debug("Not DatasetFolder or ImageFolder, so iter through")
        return [label for img, label, *other_info in given_dataset]

def badnet_stage1_non_training_data_prepare(args):
    logging.info(f"badnet stage1 start")

    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transform, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform, \
    clean_train_dataset_with_transform, \
    clean_train_dataset_targets, \
    clean_test_dataset_with_transform, \
    clean_test_dataset_targets \
        = benign_prepare(args)

    train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
    ### get the backdoor transform on label
    bd_label_transform = bd_attack_label_trans_generate(args)

    ### 4. set the backdoor attack data and backdoor test data
    train_poison_index = generate_poison_index_from_label_transform(
        clean_train_dataset_targets,
        label_transform=bd_label_transform,
        train=True,
        pratio=args.pratio if 'pratio' in args.__dict__ else None,
        p_num=args.p_num if 'p_num' in args.__dict__ else None,
    )

    logging.debug(f"poison train idx is saved")
    torch.save(train_poison_index,
               args.save_path + '/train_poison_index_list.pickle',
               )

    ### generate train dataset for backdoor attack
    bd_train_dataset = prepro_cls_DatasetBD_v2(
        deepcopy(train_dataset_without_transform),
        poison_indicator=train_poison_index,
        bd_image_pre_transform=train_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        save_folder_path=f"{args.save_path}/bd_train_dataset",
    )

    bd_train_dataset_with_transform = dataset_wrapper_with_transform(
        bd_train_dataset,
        train_img_transform,
        train_label_transform,
    )

    ### decide which img to poison in ASR Test
    test_poison_index = generate_poison_index_from_label_transform(
        clean_test_dataset_targets,
        label_transform=bd_label_transform,
        train=False,
    )

    ### generate test dataset for ASR
    bd_test_dataset = prepro_cls_DatasetBD_v2(
        deepcopy(test_dataset_without_transform),
        poison_indicator=test_poison_index,
        bd_image_pre_transform=test_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        save_folder_path=f"{args.save_path}/bd_test_dataset",
    )

    bd_test_dataset.subset(
        np.where(test_poison_index == 1)[0]
    )

    bd_test_dataset_with_transform = dataset_wrapper_with_transform(
        bd_test_dataset,
        test_img_transform,
        test_label_transform,
    )

    return clean_train_dataset_with_transform, \
                          clean_test_dataset_with_transform, \
                          bd_train_dataset_with_transform, \
                          bd_test_dataset_with_transform

def sig_stage1_non_training_data_prepare(args):
    logging.info(f"stage1 start")

    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transform, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform, \
    clean_train_dataset_with_transform, \
    clean_train_dataset_targets, \
    clean_test_dataset_with_transform, \
    clean_test_dataset_targets \
        = benign_prepare(args)

    train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
    ### get the backdoor transform on label
    bd_label_transform = bd_attack_label_trans_generate(args)

    ### 4. set the backdoor attack data and backdoor test data
    train_poison_index = generate_poison_index_from_label_transform(
        clean_train_dataset_targets,
        label_transform=bd_label_transform,
        train=True,
        pratio=args.pratio if 'pratio' in args.__dict__ else None,
        p_num=args.p_num if 'p_num' in args.__dict__ else None,
        clean_label=True,
    )

    logging.debug(f"poison train idx is saved")
    torch.save(train_poison_index,
               args.save_path + '/train_poison_index_list.pickle',
               )

    ### generate train dataset for backdoor attack
    bd_train_dataset = prepro_cls_DatasetBD_v2(
        deepcopy(train_dataset_without_transform),
        poison_indicator=train_poison_index,
        bd_image_pre_transform=train_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        save_folder_path=f"{args.save_path}/bd_train_dataset",
    )

    bd_train_dataset_with_transform = dataset_wrapper_with_transform(
        bd_train_dataset,
        train_img_transform,
        train_label_transform,
    )

    ### decide which img to poison in ASR Test
    test_poison_index = generate_poison_index_from_label_transform(
        clean_test_dataset_targets,
        label_transform=bd_label_transform,
        train=False,
    )

    ### generate test dataset for ASR
    bd_test_dataset = prepro_cls_DatasetBD_v2(
        deepcopy(test_dataset_without_transform),
        poison_indicator=test_poison_index,
        bd_image_pre_transform=test_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        save_folder_path=f"{args.save_path}/bd_test_dataset",
    )

    bd_test_dataset.subset(
        np.where(test_poison_index == 1)[0]
    )

    bd_test_dataset_with_transform = dataset_wrapper_with_transform(
        bd_test_dataset,
        test_img_transform,
        test_label_transform,
    )

    return clean_train_dataset_with_transform, \
              clean_test_dataset_with_transform, \
              bd_train_dataset_with_transform, \
              bd_test_dataset_with_transform

def wanet_stage1_non_training_data_prepare(args):
    logging.info("wanet stage1 start")

    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transform, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform, \
    clean_train_dataset_with_transform, \
    clean_train_dataset_targets, \
    clean_test_dataset_with_transform, \
    clean_test_dataset_targets \
        = benign_prepare(args)

    logging.info("Be careful, here must replace the regular train tranform with test transform.")
    # you can find in the original code that get_transform function has pretensor_transform=False always.
    clean_train_dataset_with_transform.wrap_img_transform = test_img_transform


    logging.info(f"wanet stage2 start")

    # set the backdoor warping
    ins = torch.rand(1, 2, args.k, args.k) * 2 - 1  # generate (1,2,4,4) shape [-1,1] gaussian
    ins = ins / torch.mean(
        torch.abs(ins))  # scale up, increase var, so that mean of positive part and negative be +1 and -1
    noise_grid = (
        F.upsample(ins, size=args.input_height, mode="bicubic",
                   align_corners=True)  # here upsample and make the dimension match
            .permute(0, 2, 3, 1)
            .to(args.device, non_blocking=args.non_blocking)
    )
    array1d = torch.linspace(-1, 1, steps=args.input_height)
    x, y = torch.meshgrid(array1d,
                          array1d)  # form two mesh grid correspoding to x, y of each position in height * width matrix
    identity_grid = torch.stack((y, x), 2)[None, ...].to(
        args.device,
        non_blocking=args.non_blocking)  # stack x,y like two layer, then add one more dimension at first place. (have torch.Size([1, 32, 32, 2]))

    # filter out transformation that not reversible
    transforms_reversible = transforms.Compose(
        list(
            filter(
                lambda x: isinstance(x, (transforms.Normalize, transforms.Resize, transforms.ToTensor)),
                (clean_test_dataset_with_transform.wrap_img_transform.transforms)
            )
        )
    )
    # get denormalizer
    for trans_t in (clean_test_dataset_with_transform.wrap_img_transform.transforms):
        if isinstance(trans_t, transforms.Normalize):
            denormalizer = get_dataset_denormalization(trans_t)
            logging.info(f"{denormalizer}")

    reversible_test_dataset = (clean_test_dataset_with_transform)
    reversible_test_dataset.wrap_img_transform = transforms_reversible

    reversible_test_dataloader = torch.utils.data.DataLoader(reversible_test_dataset, batch_size=args.batch_size,
                                                             pin_memory=args.pin_memory,
                                                             num_workers=args.num_workers, shuffle=False)
    bd_test_dataset = prepro_cls_DatasetBD_v2(
        clean_test_dataset_with_transform.wrapped_dataset, save_folder_path=f"{args.save_path}/bd_test_dataset"
    )
    for batch_idx, (inputs, targets) in enumerate(reversible_test_dataloader):
        with torch.no_grad():
            inputs, targets = inputs.to(args.device, non_blocking=args.non_blocking), targets.to(args.device,
                                                                                                 non_blocking=args.non_blocking)
            bs = inputs.shape[0]

            # Evaluate Backdoor
            grid_temps = (identity_grid + args.s * noise_grid / args.input_height) * args.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, args.input_height, args.input_height, 2).to(args.device,
                                                                             non_blocking=args.non_blocking) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / args.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = denormalizer(F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True))

            if args.attack_label_trans == "all2one":
                position_changed = (
                        args.attack_target != targets)  # since if label does not change, then cannot tell if the poison is effective or not.
                targets_bd = (torch.ones_like(targets) * args.attack_target)[position_changed]
                inputs_bd = inputs_bd[position_changed]
            if args.attack_label_trans == "all2all":
                position_changed = torch.ones_like(targets) # here assume all2all is the bd label = (true label + 1) % num_classes
                targets_bd = torch.remainder(targets + 1, args.num_classes)
                inputs_bd = inputs_bd

            targets = targets.detach().clone().cpu()
            y_poison_batch = targets_bd.detach().clone().cpu().tolist()
            for idx_in_batch, t_img in enumerate(inputs_bd.detach().clone().cpu()):
                bd_test_dataset.set_one_bd_sample(
                    selected_index=int(
                        batch_idx * int(args.batch_size) + torch.where(position_changed.detach().clone().cpu())[0][
                            idx_in_batch]),
                    # manually calculate the original index, since we do not shuffle the dataloader
                    img=(t_img),
                    bd_label=int(y_poison_batch[idx_in_batch]),
                    label=int(targets[torch.where(position_changed.detach().clone().cpu())[0][idx_in_batch]]),
                )

    bd_test_dataset_with_transform = dataset_wrapper_with_transform(
        bd_test_dataset,
        clean_test_dataset_with_transform.wrap_img_transform,
    )

    return clean_train_dataset_with_transform, \
           clean_test_dataset_with_transform, \
           None, \
           bd_test_dataset_with_transform
