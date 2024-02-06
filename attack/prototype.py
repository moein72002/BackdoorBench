'''
this script is for basic normal training
'''

import os
import sys
import random
from torchvision import transforms
import matplotlib.pyplot as plt

import yaml

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import pformat
import torch
import time
import logging

from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, \
    exposure_dataset_and_transform_generate, clean_dataset_and_transform_generate_ood, \
    exposure_dataset_and_transform_generate_ood
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.bd_dataset_v2 import dataset_wrapper_with_transform, get_labels
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.log_assist import get_git_info
from utils.trainer_cls import ModelTrainerCLS_v2


class NormalCase:

    def __init__(self):
        pass

    def set_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("-n", "--num_workers", type=int, help="dataloader num_workers")
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'],
                            help="dataloader pin_memory")
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
        parser.add_argument('--device', type=str)
        parser.add_argument('--lr_scheduler', type=str,
                            help='which lr_scheduler use for optimizer')
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--dataset', type=str,
                            help='which dataset to use'
                            )
        parser.add_argument('--dataset_path', type=str)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--client_optimizer', type=str)
        parser.add_argument('--random_seed', type=int,
                            help='random_seed')
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')
        parser.add_argument('--model', type=str,
                            help='choose which kind of model')
        parser.add_argument('--save_folder_name', type=str,
                            help='(Optional) should be time str + given unique identification str')
        parser.add_argument('--git_hash', type=str,
                            help='git hash number, in order to find which version of code is used')
        parser.add_argument("--yaml_path", type=str, default="../config/attack/prototype/cifar10.yaml")
        parser.add_argument("--exposure_blend_rate", type=float, default=0.5)
        parser.add_argument("--poison_all_test_ood", type=str, default="false")
        parser.add_argument("--attack_exposure_train", type=str, default="true")
        parser.add_argument("--test_every_epoch", type=str, default="false")
        return parser

    def add_yaml_to_args(self, args):
        with open(args.yaml_path, 'r') as f:
            clean_defaults = yaml.safe_load(f)
        clean_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = clean_defaults

    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        return args

    def prepare(self, args):
        ### save path
        if 'save_folder_name' not in args:
            save_path = generate_save_folder(
                run_info=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + (
                    args.attack if 'attack' in args.__dict__ else "prototype"),
                given_load_file_path=args.load_path if 'load_path' in args else None,
                all_record_folder_path='../record',
            )
        else:
            save_path = '../record/' + args.save_folder_name
            os.mkdir(save_path)
        args.save_path = save_path

        torch.save(args.__dict__, save_path + '/info.pickle')

        ### set the logger
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        # file Handler
        fileHandler = logging.FileHandler(
            save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)
        # consoleHandler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)
        # overall logger level should <= min(handler) otherwise no log will be recorded.
        logger.setLevel(0)

        # disable other debug, since too many debug
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        logging.info(pformat(args.__dict__))

        logging.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

        # record the git infomation for debug (if available.)
        try:
            logging.debug(pformat(get_git_info()))
        except:
            logging.debug('Getting git info fails.')

        ### set the random seed
        fix_random(int(args.random_seed))

        self.args = args

    def count_unique_labels_of_dataset(self, dataset, dataset_name):
        label_counts = {}

        # Enumerate through the train_dataset
        for i, (data, label) in enumerate(dataset):
            # Count the occurrences of each label
            label_counts[label] = label_counts.get(label, 0) + 1

        # Print the count of unique labels
        print(f"\nCount of Unique Labels of {dataset_name}:")
        for label, count in label_counts.items():
            print(f"{label}: {count}")

    def count_unique_labels_of_preprocessed_dataset(self, dataset, dataset_name):
        label_counts = {}

        # Enumerate through the train_dataset
        for i, (x, label, original_index, poison_indicator, original_target) in enumerate(dataset):
            # Count the occurrences of each label
            label_counts[original_target] = label_counts.get(original_target, 0) + 1

        # Print the count of unique labels
        print(f"\nCount of Unique Labels of {dataset_name}:")
        for label, count in label_counts.items():
            print(f"{label}: {count}")
    def benign_prepare(self):

        assert 'args' in self.__dict__

        args = self.args

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform = dataset_and_transform_generate(args)

        if args.attack_exposure_train == 'true':
            train_dataset_without_transform = exposure_dataset_and_transform_generate(args)

        clean_test_dataset_without_transform_ood, \
        test_img_transform_ood, \
        test_label_transform_ood = clean_dataset_and_transform_generate_ood(args)

        exposure_test_dataset_without_transform_ood, \
        _, \
        _ = exposure_dataset_and_transform_generate_ood(args)

        # self.count_unique_labels_of_dataset(test_dataset_without_transform_ood, "test_dataset_without_transform_ood")

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

        clean_test_dataset_with_transform_ood = dataset_wrapper_with_transform(
            clean_test_dataset_without_transform_ood,
            test_img_transform_ood,
            test_label_transform_ood,
        )

        clean_test_dataset_targets_ood = get_labels(clean_test_dataset_without_transform_ood)

        return train_dataset_without_transform, \
               train_img_transform, \
               train_label_transform, \
               test_dataset_without_transform, \
               test_img_transform, \
               test_label_transform, \
               clean_train_dataset_with_transform, \
               clean_train_dataset_targets, \
               clean_test_dataset_with_transform, \
               clean_test_dataset_targets, \
               exposure_test_dataset_without_transform_ood, \
               test_img_transform_ood, \
               test_label_transform_ood, \
               clean_test_dataset_with_transform_ood, \
               clean_test_dataset_targets_ood

    def stage1_non_training_data_prepare(self):

        # You should rewrite this for specific attack method

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
        test_dataset_without_transform_ood, \
        test_img_transform_ood, \
        test_label_transform_ood, \
        clean_test_dataset_with_transform_ood, \
        clean_test_dataset_targets_ood, \
            = self.benign_prepare()

        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              None, \
                              None

    def stage2_training(self):

        # You should rewrite this for specific attack method

        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset, \
        bd_test_dataset = self.stage1_results

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

        trainer = ModelTrainerCLS_v2(
            self.net,
        )

        criterion = argparser_criterion(args)

        optimizer, scheduler = argparser_opt_scheduler(self.net, args)

        trainer.set_with_dataset(
            batch_size=args.batch_size,
            train_dataset=clean_train_dataset_with_transform,
            test_dataset_dict={
                "ACC": clean_test_dataset_with_transform,
            },
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='attack',
            amp=args.amp,
            prefetch=args.prefetch,
            num_workers=args.num_workers,
            prefetch_transform_attr_name="wrap_img_transform",  # since we use the preprocess_bd_dataset
            pin_memory=args.pin_memory,
            non_blocking=args.non_blocking,
        )

        for epoch_idx in range(args.epochs):
            trainer.train_one_epoch()
            trainer.agg(
                trainer.test_all_inner_dataloader()["ACC"]
            )
            trainer.agg_save_dataframe()

        torch.save(self.net.cpu().state_dict(), f"{args.save_path}/clean_model.pth")

    def show_images(self, images, labels, dataset_name):
        num_images = len(images)
        rows = int(num_images / 5) + 1

        fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3))

        for i, ax in enumerate(axes.flatten()):
            if i < num_images:
                ax.imshow(images[i].permute(1, 2, 0))  # permute to (H, W, C) for displaying RGB images
                ax.set_title(f"Label: {labels[i]}")
            ax.axis("off")

        plt.savefig(f'{dataset_name}_visualization.png')

    def visualize_random_samples_from_clean_dataset(self, dataset, dataset_name):
        print(f"Start visualization of clean dataset: {dataset_name}")
        # Choose 20 random indices from the dataset
        random_indices = random.sample(range(len(dataset)), 20)

        # Retrieve corresponding samples
        random_samples = [dataset[i] for i in random_indices]

        # Separate images and labels
        images, labels = zip(*random_samples)

        # Convert PIL images to PyTorch tensors
        transform = transforms.ToTensor()
        images = [transform(image) for image in images]

        # Convert labels to PyTorch tensor
        labels = torch.tensor(labels)

        # Show the 20 random samples
        self.show_images(images, labels, dataset_name)

    def visualize_random_samples_from_bd_dataset(self, dataset, dataset_name):
        print(f"Start visualization of bd dataset: {dataset_name}")
        # Choose 20 random indices from the dataset
        random_indices = random.sample(range(len(dataset)), 20)

        # Retrieve corresponding samples
        random_samples = [dataset[i] for i in random_indices]

        # Separate images and labels
        images, labels, original_index, poison_indicator, original_targets = zip(*random_samples)

        # Convert PIL images to PyTorch tensors
        transform = transforms.ToTensor()
        images = [transform(image) for image in images]

        # Convert labels to PyTorch tensor
        original_targets = torch.tensor(original_targets)

        # Show the 20 random samples
        self.show_images(images, original_targets, dataset_name)


if __name__ == '__main__':
    normal_train_process = NormalCase()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = normal_train_process.set_args(parser)
    args = parser.parse_args()
    normal_train_process.add_yaml_to_args(args)
    args = normal_train_process.process_args(args)
    normal_train_process.prepare(args)
    normal_train_process.stage1_non_training_data_prepare()
    normal_train_process.stage2_training()
