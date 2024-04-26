from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm



class MDSPostprocessor():
    def __init__(self, num_classes=10):
        
        self.num_classes = num_classes
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_train, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(id_loader_train,
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch[0].cuda(), batch[1]
                    features, logits = net.get_embeds_and_logit_from_forward(data)
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            # compute class-conditional statistics
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data
                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples -
                                     self.class_mean[c].view(1, -1))

            self.class_mean = torch.stack(
                self.class_mean)  # shape [#classes, feature dim]

            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            group_lasso.fit(
                torch.cat(centered_data).cpu().numpy().astype(np.float32))
            # inverse of covariance
            self.precision = torch.from_numpy(group_lasso.precision_).float()
            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        features, logits = net.get_embeds_and_logit_from_forward(data)
        pred = logits.argmax(1)

        class_scores = torch.zeros((logits.shape[0], self.num_classes))
        for c in range(self.num_classes):
            tensor = features.cpu() - self.class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, self.precision), tensor.t()).diag()

        conf = torch.max(class_scores, dim=1)[0]
        return conf, class_scores