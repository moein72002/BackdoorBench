import torch
from torch.utils.data import DataLoader
from utils.visualize_dataset import visualize_random_samples_from_bd_dataset, visualize_random_samples_from_clean_dataset
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform



class defense(object):


    def __init__(self,):
        # TODO:yaml config log(测试两个防御方法同时使用会不会冲突)
        print(1)

    def add_arguments(parser):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法需要重写该方法以实现给参数的功能
        print('You need to rewrite this method for passing parameters')
    
    def set_result(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法需要重写该方法以读取攻击的结果
        print('You need to rewrite this method to load the attack result')
        
    def set_trainer(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法可以重写该方法以实现整合训练模块的功能
        print('If you want to use standard trainer module, please rewrite this method')
    
    def set_logger(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法可以重写该方法以实现存储log的功能
        print('If you want to use standard logger, please rewrite this method')

    def denoising(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

    def mitigation(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

    def inhibition(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')
    
    def defense(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')
    
    def detect(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

    def get_test_data_loaders_dict(self, args, test_tran):
        data_bd_testset = self.result['bd_test']
        visualize_random_samples_from_bd_dataset(data_bd_testset.wrapped_dataset, "data_bd_testset.wrapped_dataset")
        data_bd_testset.wrap_img_transform = test_tran
        # data_bd_testset.wrapped_dataset.getitem_all = False
        poison_test_loader = DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,
                                        drop_last=False, shuffle=True, pin_memory=True)
        test_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        data_clean_testset = self.result['clean_test']
        visualize_random_samples_from_clean_dataset(data_clean_testset.wrapped_dataset,
                                                    "data_clean_testset.wrapped_dataset")
        data_clean_testset.wrap_img_transform = test_tran
        clean_test_loader = DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       drop_last=False, shuffle=True, pin_memory=True)
        data_bd_out_testset_ood = self.result['bd_out_test_ood']
        visualize_random_samples_from_bd_dataset(data_bd_out_testset_ood.wrapped_dataset,
                                                 "data_bd_out_testset_ood.wrapped_dataset")
        data_bd_out_testset_ood.wrap_img_transform = test_tran
        bd_out_test_loader_ood = torch.utils.data.DataLoader(data_bd_out_testset_ood, batch_size=self.args.batch_size,
                                                             num_workers=self.args.num_workers, drop_last=False,
                                                             shuffle=True,
                                                             pin_memory=args.pin_memory)
        data_bd_all_testset_ood = self.result['bd_all_test_ood']
        visualize_random_samples_from_bd_dataset(data_bd_all_testset_ood.wrapped_dataset,
                                                 "data_bd_all_testset_ood.wrapped_dataset")
        data_bd_all_testset_ood.wrap_img_transform = test_tran
        bd_all_test_loader_ood = torch.utils.data.DataLoader(data_bd_all_testset_ood, batch_size=self.args.batch_size,
                                                             num_workers=self.args.num_workers, drop_last=False,
                                                             shuffle=True,
                                                             pin_memory=args.pin_memory)
        data_clean_testset_ood = self.result['clean_test_ood']
        visualize_random_samples_from_clean_dataset(data_clean_testset_ood.wrapped_dataset,
                                                    "data_clean_testset_ood.wrapped_dataset")
        data_clean_testset_ood.wrap_img_transform = test_tran
        clean_test_loader_ood = torch.utils.data.DataLoader(data_clean_testset_ood, batch_size=self.args.batch_size,
                                                            num_workers=self.args.num_workers, drop_last=False,
                                                            shuffle=True, pin_memory=args.pin_memory)
        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = clean_test_loader
        test_dataloader_dict["bd_test_dataloader"] = poison_test_loader
        test_dataloader_dict["clean_test_dataloader_ood"] = clean_test_loader_ood
        test_dataloader_dict["bd_out_test_dataloader_ood"] = bd_out_test_loader_ood
        test_dataloader_dict["bd_all_test_dataloader_ood"] = bd_all_test_loader_ood
        if 'bd_test_for_cls' in self.result.__dict__:
            data_bd_testset_for_cls = self.result['bd_test_for_cls']
            visualize_random_samples_from_bd_dataset(data_bd_testset_for_cls.wrapped_dataset,
                                                     "data_bd_testset_for_cls.wrapped_dataset")
            data_bd_testset_for_cls.wrap_img_transform = test_tran
            bd_test_loader_for_cls = torch.utils.data.DataLoader(data_bd_testset_for_cls,
                                                                 batch_size=self.args.batch_size,
                                                                 num_workers=self.args.num_workers, drop_last=False,
                                                                 shuffle=True,
                                                                 pin_memory=args.pin_memory)
            test_dataloader_dict["bd_test_dataloader_for_cls"] = bd_test_loader_for_cls
        return test_dataloader_dict


