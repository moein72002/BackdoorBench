'''
this script is for badnet attack

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. attack or use the model to do finetune with 5% clean data
7. save the attack result for defense

@article{gu2017badnets,
  title={Badnets: Identifying vulnerabilities in the machine learning model supply chain},
  author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
  journal={arXiv preprint arXiv:1708.06733},
  year={2017}
}
'''

import os
import sys
import yaml

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
import numpy as np
import torch
import logging
import pickle

from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.aggregate_block.dataset_and_transform_generate import SIMPLE_DATASET_FOR_VISUALIZATION, get_transform
from utils.save_load_attack import save_attack_result
from attack.prototype import NormalCase
from utils.trainer_cls import BackdoorModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.save_load_attack import load_clean_trained_model
import torchvision
import torchvision.transforms as transforms


def add_common_attack_args(parser):
    parser.add_argument('--attack', type=str, )
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    return parser


class BadNet(NormalCase):

    def __init__(self):
        super(BadNet).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)

        parser.add_argument("--patch_mask_path", type=str)
        parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/badnet/default.yaml',
                            help='path for yaml file provide additional default attributes')
        parser.add_argument("--top_k", type=int, default=0)     # top_k effect is on when it is more than zero
        return parser

    def add_bd_yaml_to_args(self, args):
        with open(args.bd_yaml_path, 'r') as f:
            mix_defaults = yaml.safe_load(f)
        mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = mix_defaults

    # def count_unique_labels_of_dataset(self, dataset, dataset_name):
    #     label_counts = {}
    #
    #     # Enumerate through the train_dataset
    #     for i, (data, label, original_index, poison_indicator, original_target) in enumerate(dataset):
    #         # Count the occurrences of each label
    #         label_counts[label] = label_counts.get(label, 0) + 1
    #         print(data, label, original_index, poison_indicator, original_target)
    #
    #     # Print the count of unique labels
    #     print(f"\nCount of Unique Labels of {dataset_name}:")
    #     for label, count in label_counts.items():
    #         print(f"{label}: {count}")

    def check_and_visualize_saved_dataset(self, file_path):
        file_path = "../clean_trained_model/top_k_selected_images.pkl"
        with open(file_path, 'rb') as file:
            top_k_saved_images = pickle.load(file)

        top_k_dataset = SIMPLE_DATASET_FOR_VISUALIZATION(top_k_saved_images, target_label=self.args.attack_target)

        self.visualize_random_samples_from_clean_dataset(top_k_dataset, "top_k_dataset")


    def stage0_save_top_k_from_target_label_train(self):

        args = self.args

        # Load the pre-trained model
        model = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )

        clean_trained_model_dict = load_clean_trained_model('../clean_trained_model/record/badnet_0_1/attack_result.pt')

        model.load_state_dict(clean_trained_model_dict['model'])

        device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

        model.to(args.device)
        model.eval()

        if not args.dataset.startswith('test'):
            test_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=False)
        else:
            # test folder datset, use the mnist transform for convenience
            test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=None)

        # Filter out the records with label 0
        label_0_indices = [i for i, (image, label) in enumerate(trainset) if label == 0]

        # Select the top k records with label 0 based on model's prediction
        top_k_label_0_indices = sorted(label_0_indices,
                                         key=lambda x: model(test_img_transform(trainset[x][0]).unsqueeze(0).to(device))[
                                             0, 0].item(), reverse=True)[:args.top_k]

        # Get the corresponding images and labels
        top_k_selected_images = [trainset[i][0] for i in top_k_label_0_indices]

        # File path
        file_path = "../clean_trained_model/top_k_selected_images.pkl"

        # Save list using pickle
        with open(file_path, 'wb') as file:
            pickle.dump(top_k_selected_images, file)

        self.check_and_visualize_saved_dataset(file_path)



    def stage1_non_training_data_prepare(self):
        logging.info(f"stage1 start")

        assert 'args' in self.__dict__
        args = self.args

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets, \
        exposure_test_dataset_without_transform_for_cls, \
        exposure_out_test_dataset_without_transform_ood, \
        exposure_all_test_dataset_without_transform_ood, \
        test_img_transform_ood, \
        test_label_transform_ood, \
        clean_test_dataset_with_transform_ood, \
        clean_test_dataset_targets_ood \
            = self.benign_prepare()

        train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
        ### get the backdoor transform on label
        bd_label_transform = bd_attack_label_trans_generate(args)
        bd_label_transform_for_cls = bd_attack_label_trans_generate(args)
        bd_label_transform_ood = bd_attack_label_trans_generate(args, is_ood_dataset=True)

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

        # bd_train_dataset will be clean for exposure_test
        train_poison_index = np.zeros(len(train_dataset_without_transform))

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

        self.count_unique_labels_of_dataset(test_dataset_without_transform, "test_dataset_without_transform")
        self.count_unique_labels_of_preprocessed_dataset(bd_test_dataset, "bd_test_dataset")

        bd_test_dataset.subset(
            np.where(test_poison_index == 1)[0]
        )

        test_poison_index_for_cls = np.zeros(len(exposure_test_dataset_without_transform_for_cls))

        self.count_unique_labels_of_dataset(exposure_test_dataset_without_transform_for_cls,
                                            "exposure_test_dataset_without_transform_for_cls")
        self.visualize_random_samples_from_clean_dataset(exposure_test_dataset_without_transform_for_cls,
                                                         "exposure_test_dataset_without_transform_for_cls")

        bd_test_dataset_for_cls = prepro_cls_DatasetBD_v2(
            deepcopy(exposure_test_dataset_without_transform_for_cls),
            poison_indicator=test_poison_index_for_cls,
            bd_image_pre_transform=test_bd_img_transform,  # TODO: check here
            bd_label_pre_transform=bd_label_transform_for_cls,
            save_folder_path=f"{args.save_path}/bd_test_dataset_ood",
        )

        self.count_unique_labels_of_preprocessed_dataset(bd_test_dataset_for_cls, "bd_test_dataset_for_cls")

        test_poison_index_ood = np.zeros(len(exposure_out_test_dataset_without_transform_ood))

        self.count_unique_labels_of_dataset(clean_test_dataset_with_transform_ood.wrapped_dataset,
                                            "clean_test_dataset_with_transform_ood.wrapped_dataset")
        self.visualize_random_samples_from_clean_dataset(clean_test_dataset_with_transform_ood.wrapped_dataset,
                                                         "clean_test_dataset_with_transform_ood.wrapped_dataset")
        self.count_unique_labels_of_dataset(exposure_out_test_dataset_without_transform_ood, "exposure_out_test_dataset_without_transform_ood")
        self.visualize_random_samples_from_clean_dataset(exposure_out_test_dataset_without_transform_ood, "exposure_out_test_dataset_without_transform_ood")

        self.count_unique_labels_of_dataset(exposure_all_test_dataset_without_transform_ood, "exposure_all_test_dataset_without_transform_ood")
        self.visualize_random_samples_from_clean_dataset(exposure_all_test_dataset_without_transform_ood, "exposure_all_test_dataset_without_transform_ood")

        bd_out_test_dataset_ood = prepro_cls_DatasetBD_v2(
            deepcopy(exposure_out_test_dataset_without_transform_ood),
            poison_indicator=test_poison_index_ood,
            bd_image_pre_transform=test_bd_img_transform, # TODO: check here
            bd_label_pre_transform=bd_label_transform_ood,
            save_folder_path=f"{args.save_path}/bd_test_dataset_ood",
        )

        self.count_unique_labels_of_preprocessed_dataset(bd_out_test_dataset_ood, "bd_out_test_dataset_ood")

        bd_all_test_dataset_ood = prepro_cls_DatasetBD_v2(
            deepcopy(exposure_all_test_dataset_without_transform_ood),
            poison_indicator=test_poison_index_ood,
            bd_image_pre_transform=test_bd_img_transform,  # TODO: check here
            bd_label_pre_transform=bd_label_transform_ood,
            save_folder_path=f"{args.save_path}/bd_test_dataset_ood",
        )

        self.count_unique_labels_of_preprocessed_dataset(bd_all_test_dataset_ood, "bd_all_test_dataset_ood")

        # TODO: check here
        # bd_test_dataset_ood.subset(
        #     np.where(test_poison_index_ood == 1)[0]
        # )

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
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

        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              bd_train_dataset_with_transform, \
                              bd_test_dataset_with_transform, \
                              clean_test_dataset_with_transform_ood, \
                              bd_test_dataset_with_transform_for_cls, \
                              bd_out_test_dataset_with_transform_ood, \
                              bd_all_test_dataset_with_transform_ood

    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform, \
        clean_test_dataset_with_transform_ood, \
        bd_test_dataset_with_transform_for_cls, \
        bd_out_test_dataset_with_transform_ood, \
        bd_all_test_dataset_with_transform_ood, \
            = self.stage1_results

        self.visualize_random_samples_from_bd_dataset(bd_train_dataset_with_transform.wrapped_dataset, "bd_train_dataset_with_transform.wrapped_dataset")

        self.net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )


        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

        if "," in args.device:
            self.net = torch.nn.DataParallel(
                self.net,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )

        trainer = BackdoorModelTrainer(
            self.net,
        )

        criterion = argparser_criterion(args)

        optimizer, scheduler = argparser_opt_scheduler(self.net, args)

        from torch.utils.data.dataloader import DataLoader
        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(bd_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=True,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(clean_test_dataset_with_transform_ood, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(bd_test_dataset_with_transform_for_cls, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(bd_out_test_dataset_with_transform_ood, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(bd_all_test_dataset_with_transform_ood, batch_size=args.batch_size, shuffle=False,
                       drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='attack',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",  # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
            test_every_epoch=args.test_every_epoch
        )

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=trainer.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=bd_train_dataset_with_transform,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path,
            bd_test_for_cls=bd_test_dataset_with_transform_for_cls,
            bd_out_test_ood=bd_out_test_dataset_with_transform_ood,
            bd_all_test_ood=bd_all_test_dataset_with_transform_ood,
            exposure_blend_rate=args.exposure_blend_rate
        )


if __name__ == '__main__':
    attack = BadNet()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    logging.debug("Be careful that we need to give the bd yaml higher priority. So, we put the add bd yaml first.")
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage0_save_top_k_from_target_label_train()
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
