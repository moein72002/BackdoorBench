import os,sys
import numpy as np
import torch
from utils.trainer_cls import get_score_knn_auc


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

    def eval_step_knn_auc(
            self,
            netC,
            train_loader,
            clean_test_dataloader_ood,
            bd_out_test_dataloader_ood,
            bd_all_test_dataloader_ood,
            args,
    ):
        device = self.args.device
        knn_clean_test_auc = get_score_knn_auc(netC, device, train_loader, clean_test_dataloader_ood, bd_test_loader=False)
        knn_bd_out_test_auc = get_score_knn_auc(netC, device, train_loader, bd_out_test_dataloader_ood, bd_test_loader=True)
        knn_bd_all_test_auc = get_score_knn_auc(netC, device, train_loader, bd_all_test_dataloader_ood, bd_test_loader=True)

        return knn_clean_test_auc, \
               knn_bd_out_test_auc, \
               knn_bd_all_test_auc

