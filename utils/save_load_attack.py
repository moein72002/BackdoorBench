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
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
import numpy as np
from copy import deepcopy
from pprint import pformat
from typing import Union

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate, \
    clean_dataset_and_transform_generate_ood, \
    exposure_dataset_and_transform_generate_ood, exposure_dataset_and_transform_generate, \
    exposure_dataset_and_transform_generate_for_cls


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

def count_unique_labels_of_preprocessed_dataset(dataset, dataset_name):
    label_counts = {}

    # Enumerate through the train_dataset
    for i, (x, label, original_index, poison_indicator, original_target) in enumerate(dataset):
        # Count the occurrences of each label
        label_counts[original_target] = label_counts.get(original_target, 0) + 1

    # Print the count of unique labels
    print(f"\nCount of Unique Labels of {dataset_name}:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
def save_attack_result(
    model_name : str,
    num_classes : int,
    model : dict, # the state_dict
    data_path : str,
    img_size : Union[list, tuple],
    clean_data : str,
    bd_test : prepro_cls_DatasetBD_v2, # MUST be dataset without transform
    save_path : str,
    exposure_blend_rate : int,
    bd_test_for_cls : prepro_cls_DatasetBD_v2,
    bd_out_test_ood : prepro_cls_DatasetBD_v2,
    bd_all_test_ood : prepro_cls_DatasetBD_v2,
    bd_train : Optional[prepro_cls_DatasetBD_v2] = None, # MUST be dataset without transform
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

    # count_unique_labels_of_preprocessed_dataset(bd_test_ood, "bd_test_ood")

    save_dict = {
            'model_name': model_name,
            'num_classes' : num_classes,
            'model': model,
            'data_path': data_path,
            'img_size' : img_size,
            'clean_data': clean_data,
            'bd_train': bd_train.retrieve_state() if bd_train is not None else None,
            'bd_test': bd_test.retrieve_state(),
            'bd_test_for_cls': bd_test_for_cls.retrieve_state(),
            'bd_out_test_ood': bd_out_test_ood.retrieve_state(),
            'bd_all_test_ood': bd_all_test_ood.retrieve_state(),
            'exposure_blend_rate': exposure_blend_rate
        }

    logging.info(f"saving...")
    # logging.debug(f"location : {save_path}/attack_result.pt") #, content summary :{pformat(summary_dict(save_dict))}")

    torch.save(
        save_dict,
        f'{save_path}/attack_result.pt',
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

def load_clean_trained_model(
    save_path : str
):
    load_file = torch.load(save_path)
    if 'model' in load_file:
        new_dict = copy.deepcopy(load_file['model'])
        for k, v in load_file['model'].items():
            if k.startswith('module.'):
                del new_dict[k]
                new_dict[k[7:]] = v

        load_file['model'] = new_dict
        load_dict = {
            'model': load_file['model']
        }

        print(f"loading...")
        return load_dict
    else:
        print("clean_trained_model does not exist")

def load_attack_result(
    save_path : str,
    just_test_exposure_ood = 'false',
    test_blend_rate = 0.1
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
        'data_path',
        'img_size',
        'clean_data',
        'bd_train',
        'bd_test',
        'bd_test_ood'
        ]):

        logging.info('key match for attack_result, processing...')

        # model = generate_cls_model(load_file['model_name'], load_file['num_classes'])
        # model.load_state_dict(load_file['model'])

        clean_setting = Args()

        clean_setting.dataset = load_file['clean_data']

        # convert the relative/abs path in attack result to abs path for defense
        clean_setting.dataset_path = load_file['data_path']
        logging.warning("save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path")
        clean_setting.dataset_path = save_path[:save_path.index('record')] + clean_setting.dataset_path[clean_setting.dataset_path.index('data'):]

        clean_setting.img_size = load_file['img_size']

        exposure_blend_rate = load_file['exposure_blend_rate']
        if just_test_exposure_ood == 'true':
            exposure_blend_rate = test_blend_rate
        clean_setting.exposure_blend_rate = exposure_blend_rate

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform = dataset_and_transform_generate(clean_setting)

        train_dataset_without_transform = exposure_dataset_and_transform_generate(clean_setting)

        clean_test_dataset_without_transform_ood, \
        test_img_transform_ood, \
        test_label_transform_ood = clean_dataset_and_transform_generate_ood(clean_setting) # TODO: check this line

        exposure_test_dataset_without_transform_for_cls, \
        _, \
        _ = exposure_dataset_and_transform_generate_for_cls(clean_setting)

        exposure_out_test_dataset_without_transform_ood, \
        _, \
        _ = exposure_dataset_and_transform_generate_ood(clean_setting, poison_all_test_ood=False)

        exposure_all_test_dataset_without_transform_ood, \
        _, \
        _ = exposure_dataset_and_transform_generate_ood(clean_setting, poison_all_test_ood=True)

        clean_train_dataset_with_transform = dataset_wrapper_with_transform(
            train_dataset_without_transform,
            train_img_transform,
            train_label_transform,
        )

        clean_test_dataset_with_transform = dataset_wrapper_with_transform(
            test_dataset_without_transform,
            test_img_transform,
            test_label_transform,
        )

        clean_test_dataset_with_transform_ood = dataset_wrapper_with_transform(
            clean_test_dataset_without_transform_ood,
            test_img_transform_ood,
            test_label_transform_ood,
        )

        if load_file['bd_train'] is not None:
            bd_train_dataset = prepro_cls_DatasetBD_v2(train_dataset_without_transform)
            bd_train_dataset.set_state(
                load_file['bd_train']
            )
            bd_train_dataset_with_transform = dataset_wrapper_with_transform(
                bd_train_dataset,
                train_img_transform,
                train_label_transform,
            )
        else:
            logging.info("No bd_train info found.")
            bd_train_dataset_with_transform = None


        bd_test_dataset = prepro_cls_DatasetBD_v2(test_dataset_without_transform)
        bd_test_dataset.set_state(
            load_file['bd_test']
        )
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        bd_test_dataset_for_cls = prepro_cls_DatasetBD_v2(exposure_test_dataset_without_transform_for_cls)
        bd_out_test_dataset_ood = prepro_cls_DatasetBD_v2(exposure_out_test_dataset_without_transform_ood)
        bd_all_test_dataset_ood = prepro_cls_DatasetBD_v2(exposure_all_test_dataset_without_transform_ood)

        # if just_test_exposure_ood == 'false':
        #     bd_test_dataset_for_cls.set_state(
        #         load_file['bd_test_for_cls']
        #     )
        #     bd_out_test_dataset_ood.set_state(
        #         load_file['bd_out_test_ood']
        #     )
        #     bd_all_test_dataset_ood.set_state(
        #         load_file['bd_all_test_ood']
        #     )

        bd_test_dataset_with_transform_for_cls = dataset_wrapper_with_transform(
            bd_test_dataset_for_cls,
            test_img_transform,
            test_label_transform,
        )

        bd_out_test_dataset_with_transform_ood = dataset_wrapper_with_transform(
            bd_out_test_dataset_ood,
            test_img_transform_ood,
            test_label_transform_ood,
        )

        bd_all_test_dataset_with_transform_ood = dataset_wrapper_with_transform(
            bd_all_test_dataset_ood,
            test_img_transform_ood,
            test_label_transform_ood,
        )

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
                'clean_test_ood': clean_test_dataset_with_transform_ood,
                'bd_test_for_cls': bd_test_dataset_with_transform_for_cls,
                'bd_out_test_ood': bd_out_test_dataset_with_transform_ood,
                'bd_all_test_ood': bd_all_test_dataset_with_transform_ood,
                'exposure_blend_rate': exposure_blend_rate
            }

        if 'clean_train' in load_dict:
            print("'clean_train' in load_dict")
        else:
            print("'clean_train' is not in load_dict")

        print(f"loading...")

        return load_dict

    else:
        logging.info(f"loading...")
        logging.debug(f"location : {save_path}, content summary :{pformat(summary_dict(load_file))}")
        return load_file

# 'bd_test_for_cls', 'bd_out_test_ood', 'bd_all_test_ood' are added
def load_new_attack_result(
    save_path : str,
    just_test_exposure_ood = 'false',
    test_blend_rate = 0.1
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
        'data_path',
        'img_size',
        'clean_data',
        'bd_train',
        'bd_test',
        'bd_test_for_cls'
        'bd_out_test_ood'
        'bd_all_test_ood'
        ]):

        logging.info('key match for attack_result, processing...')

        # model = generate_cls_model(load_file['model_name'], load_file['num_classes'])
        # model.load_state_dict(load_file['model'])

        clean_setting = Args()

        clean_setting.dataset = load_file['clean_data']

        # convert the relative/abs path in attack result to abs path for defense
        clean_setting.dataset_path = load_file['data_path']
        logging.warning("save_path MUST have 'record' in its abspath, and data_path in attack result MUST have 'data' in its path")
        clean_setting.dataset_path = save_path[:save_path.index('record')] + clean_setting.dataset_path[clean_setting.dataset_path.index('data'):]

        clean_setting.img_size = load_file['img_size']

        exposure_blend_rate = load_file['exposure_blend_rate']
        if just_test_exposure_ood == 'true':
            exposure_blend_rate = test_blend_rate
        clean_setting.exposure_blend_rate = exposure_blend_rate

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform = dataset_and_transform_generate(clean_setting)

        train_dataset_without_transform = exposure_dataset_and_transform_generate(clean_setting)

        clean_test_dataset_without_transform_ood, \
        test_img_transform_ood, \
        test_label_transform_ood = clean_dataset_and_transform_generate_ood(clean_setting) # TODO: check this line

        exposure_test_dataset_without_transform_for_cls, \
        _, \
        _ = exposure_dataset_and_transform_generate_for_cls(clean_setting)

        exposure_out_test_dataset_without_transform_ood, \
        _, \
        _ = exposure_dataset_and_transform_generate_ood(clean_setting, poison_all_test_ood=False)

        exposure_all_test_dataset_without_transform_ood, \
        _, \
        _ = exposure_dataset_and_transform_generate_ood(clean_setting, poison_all_test_ood=True)

        clean_train_dataset_with_transform = dataset_wrapper_with_transform(
            train_dataset_without_transform,
            train_img_transform,
            train_label_transform,
        )

        clean_test_dataset_with_transform = dataset_wrapper_with_transform(
            test_dataset_without_transform,
            test_img_transform,
            test_label_transform,
        )

        clean_test_dataset_with_transform_ood = dataset_wrapper_with_transform(
            clean_test_dataset_without_transform_ood,
            test_img_transform_ood,
            test_label_transform_ood,
        )

        if load_file['bd_train'] is not None:
            bd_train_dataset = prepro_cls_DatasetBD_v2(train_dataset_without_transform)
            bd_train_dataset.set_state(
                load_file['bd_train']
            )
            bd_train_dataset_with_transform = dataset_wrapper_with_transform(
                bd_train_dataset,
                train_img_transform,
                train_label_transform,
            )
        else:
            logging.info("No bd_train info found.")
            bd_train_dataset_with_transform = None


        bd_test_dataset = prepro_cls_DatasetBD_v2(test_dataset_without_transform)
        bd_test_dataset.set_state(
            load_file['bd_test']
        )
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        bd_test_dataset_for_cls = prepro_cls_DatasetBD_v2(exposure_test_dataset_without_transform_for_cls)
        bd_out_test_dataset_ood = prepro_cls_DatasetBD_v2(exposure_out_test_dataset_without_transform_ood)
        bd_all_test_dataset_ood = prepro_cls_DatasetBD_v2(exposure_all_test_dataset_without_transform_ood)

        if just_test_exposure_ood == 'false':
            bd_test_dataset_for_cls.set_state(
                load_file['bd_test_for_cls']
            )
            bd_out_test_dataset_ood.set_state(
                load_file['bd_out_test_ood']
            )
            bd_all_test_dataset_ood.set_state(
                load_file['bd_all_test_ood']
            )

        bd_test_dataset_with_transform_for_cls = dataset_wrapper_with_transform(
            bd_test_dataset_for_cls,
            test_img_transform,
            test_label_transform,
        )

        bd_out_test_dataset_with_transform_ood = dataset_wrapper_with_transform(
            bd_out_test_dataset_ood,
            test_img_transform_ood,
            test_label_transform_ood,
        )

        bd_all_test_dataset_with_transform_ood = dataset_wrapper_with_transform(
            bd_all_test_dataset_ood,
            test_img_transform_ood,
            test_label_transform_ood,
        )

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
                'clean_test_ood': clean_test_dataset_with_transform_ood,
                'bd_test_for_cls': bd_test_dataset_with_transform_for_cls,
                'bd_out_test_ood': bd_out_test_dataset_with_transform_ood,
                'bd_all_test_ood': bd_all_test_dataset_with_transform_ood,
                'exposure_blend_rate': exposure_blend_rate
            }

        if 'clean_train' in load_dict:
            print("'clean_train' in load_dict")
        else:
            print("'clean_train' is not in load_dict")

        print(f"loading...")

        return load_dict

    else:
        logging.info(f"loading...")
        logging.debug(f"location : {save_path}, content summary :{pformat(summary_dict(load_file))}")
        return load_file