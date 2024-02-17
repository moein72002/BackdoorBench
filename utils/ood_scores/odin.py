import copy

import sys, logging
sys.path.append('../')
import random
import torch.nn as nn

from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import torch

def get_odin_auc(original_model, test_dataloader, non_blocking: bool = False, device="cpu", verbose=0,
                 clean_dataset=True):
    # TODO: epsilon and temperature in below are set to default value in odin original paper
    noiseMagnitude1 = 0.0014
    temper = 1000
    CUDA_DEVICE = 0
    criterion = nn.CrossEntropyLoss()

    model = copy.deepcopy(original_model)
    model.to(device, non_blocking=non_blocking)
    model.eval()
    # model.eval()

    if verbose == 1:
        batch_label_list = []
        batch_normality_scores_list = []

    if clean_dataset:
        for batch_idx, (x, label) in tqdm(enumerate(test_dataloader)):
            # x = x.to(device, non_blocking=non_blocking)
            inputs = Variable(x.cuda(CUDA_DEVICE), requires_grad=True)

            outputs = model(inputs)

            # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
            nnOutputs = outputs.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = nnOutputs[0]
            nnOutputs = nnOutputs - np.max(nnOutputs)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

            # Using temperature scaling
            outputs = outputs / temper

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of cross entropy loss w.r.t. input
            maxIndexTemp = np.argmax(nnOutputs)
            labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
            loss = criterion(outputs, labels)
            loss.backward()

            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
            gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
            gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
            outputs = model(Variable(tempInputs))
            outputs = outputs / temper
            # Calculating the confidence after adding perturbations
            nnOutputs = outputs.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = nnOutputs[0]  # TODO: HERE I don't want to set batch_size to 1 like odin code
            nnOutputs = nnOutputs - np.max(nnOutputs)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
            # g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
            # normality_scores = torch.max(nnOutputs, dim=1)
            normality_scores = np.array([np.max(nnOutputs)])

            # print(f"original_targets[:5]: {original_targets[:5]}")
            label = label.to(device, non_blocking=non_blocking)
            # print(f"pred.size(): {pred.size()}")
            # print(f"normality_scores.size(): {normality_scores.size()}")

            if verbose == 1:
                batch_label_list.append(label.detach().clone().cpu())
                batch_normality_scores_list.append(normality_scores)
    else:
        for batch_idx, (x, labels, original_index, poison_indicator, original_targets) in enumerate(
                test_dataloader):
            # x = x.to(device, non_blocking=non_blocking)

            inputs = Variable(x.cuda(0), requires_grad=True)

            outputs = model(inputs)

            # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
            nnOutputs = outputs.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = nnOutputs[0]
            nnOutputs = nnOutputs - np.max(nnOutputs)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

            # Using temperature scaling
            outputs = outputs / temper

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of cross entropy loss w.r.t. input
            maxIndexTemp = np.argmax(nnOutputs)
            labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
            loss = criterion(outputs, labels)
            loss.backward()

            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
            gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
            gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
            outputs = model(Variable(tempInputs))
            outputs = outputs / temper
            # Calculating the confidence after adding perturbations
            nnOutputs = outputs.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = nnOutputs[0] # TODO: HERE I don't want to set batch_size to 1 like odin code
            nnOutputs = nnOutputs - np.max(nnOutputs)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
            # g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
            # normality_scores = torch.max(nnOutputs, dim=1)
            normality_scores = np.array([np.max(nnOutputs)])

            # print(f"original_targets[:5]: {original_targets[:5]}")
            original_targets = original_targets.to(device, non_blocking=non_blocking)
            # print(f"pred.size(): {pred.size()}")
            # print(f"normality_scores.size(): {normality_scores.size()}")

            if verbose == 1:
                batch_label_list.append(original_targets.detach().clone().cpu())
                # batch_normality_scores_list.append(normality_scores.detach().clone().cpu())
                batch_normality_scores_list.append(normality_scores)

    auc = roc_auc_score(torch.cat(batch_label_list).detach().cpu().numpy(),
                        np.concatenate(batch_normality_scores_list, axis=0))

    print(f"odin_auc: {auc}")

    if verbose == 0:
        return None
    elif verbose == 1:
        return auc

def eval_step_odin_auc(
        netC,
        clean_test_dataloader_ood_odin,
        bd_out_test_dataloader_ood_odin,
        bd_all_test_dataloader_ood_odin,
        args,
):
    odin_clean_test_auc = get_odin_auc(netC, clean_test_dataloader_ood_odin,
                                       non_blocking=args.non_blocking,
                                       device=args.device,
                                       verbose=1, clean_dataset=True)  # TODO
    odin_bd_out_test_auc = get_odin_auc(netC, bd_out_test_dataloader_ood_odin,
                                        non_blocking=args.non_blocking,
                                        device=args.device, verbose=1,
                                        clean_dataset=False)  # TODO
    odin_bd_all_test_auc = get_odin_auc(netC, bd_all_test_dataloader_ood_odin,
                                        non_blocking=args.non_blocking,
                                        device=args.device, verbose=1,
                                        clean_dataset=False)  # TODO

    odin_auc_result_dict = {
        "odin_clean_test_auc": odin_clean_test_auc,
        "odin_bd_out_test_auc": odin_bd_out_test_auc,
        "odin_bd_all_test_auc": odin_bd_all_test_auc
    }

    return odin_auc_result_dict