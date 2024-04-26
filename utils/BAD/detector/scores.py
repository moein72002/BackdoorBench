import torch
import torchvision
import numpy as np

import torch.nn.functional as F

from numpy.linalg import norm
from tqdm import tqdm
from BAD.eval.eval import evaluate
from BAD.utils import update_attack_params, get_features_mean_dict, find_min_eps
from BAD.utils import get_ood_outputs
from scipy import linalg



def get_epsilon_score(eps_evaluator, eps_config, log=False, proportional=False):
    return find_min_eps(eps_evaluator, eps_config['thresh'], eps_lb=eps_config['lb'], 
                        eps_ub=eps_config['ub'], max_error=eps_config['max_error'], proportional=proportional, log=log)


def get_adv_features(model, loader, target, mean_embeddings, attack, progress=False, ):
    features = []
    labels = []
    
    model.eval()
    model.to(device)

    progress_bar = loader
    if progress:
        progress_bar = tqdm(loader, unit="batch")
        
    for data, label in progress_bar:
        labels += label.tolist()
        data, label = data.to(device), label.to(device)
        if attack is not None:
            data = attack(data, label)
        feature = model.get_features(data)
        c_f = feature.detach().cpu().numpy()
        features.append(c_f)
    features = np.concatenate(features)
    
    labels = np.array(labels)
    out_features = features[1 - labels]
    in_features = features[labels]

    return out_features, in_features

# score in [l2, cosine]
def max_diff(model, testloader, attack_class=None, attack_params=None,
             score='l2', use_in=True, progress=False, num_classes=10, normalize_features=False):
    max_l2 = 0
    
    initial_features = get_features_mean_dict(testloader, feature_extractor=lambda data, targets: model.get_features(data, normalize_features))
    in_features = initial_features[1]
    out_features = initial_features[0]
    
    mean_in_initial_features = np.mean(in_features, axis=0)
    mean_out_initial_features = np.mean(out_features, axis=0)
    initial_diff = (mean_out_initial_features - mean_in_initial_features)
    
    def get_adv_feature_extractor(attack):
        return lambda data, targets : model.get_features(attack(data, targets), normalize_features)
    
    if attack_params.get('target_class') is not None:
        best_target = None
        tq = range(10)
        if progress:
            tq = tqdm(range(10))
        for i in tq:
            attack_params['target_class'] = i
            attack = attack_class(**attack_params)
            adv_features = get_features_mean_dict(testloader, get_adv_feature_extractor(attack))
            in_adv_features = adv_features[1]
            out_adv_features = adv_features[0]
            mean_in_adv_features = np.mean(in_adv_features, axis=0)
            mean_out_adv_features = np.mean(out_adv_features, axis=0)
            if use_in:
                adv_diff = (mean_out_adv_features - mean_in_adv_features)
                #cosine = np.dot(diff_a, diff_b)/(norm(diff_a)*norm(diff_b))
                l2 = norm(adv_diff - initial_diff)     
                if l2 > max_l2:
                    max_l2 = l2
            else:
                diff = mean_out_adv_features - mean_out_initial_features
                l2 = norm(diff)
                if l2 > max_l2:
                    max_l2 = l2
                    best_target = i
        return best_target, max_l2
    else:
        attack = attack_class(**attack_params)
        adv_features = get_features_mean_dict(testloader, get_adv_feature_extractor(attack))
        in_adv_features = adv_features[1]
        out_adv_features = adv_features[0]
        mean_in_adv_features = np.mean(in_adv_features, axis=0)
        mean_out_adv_features = np.mean(out_adv_features, axis=0)
        if use_in:
            adv_diff = (mean_out_adv_features - mean_in_adv_features)
            #score = np.dot(diff_a, diff_b)/(norm(diff_a)*norm(diff_b))
            score = norm(adv_diff - initial_diff)
        else:
            diff = mean_out_adv_features - mean_out_initial_features
            score = norm(diff)
        return score

    

def get_kld(model,testloader):
    ood_clean= get_ood_outputs(model, testloader, device, attack_features=False, target_class = None)
    ood_after = get_ood_outputs(model, testloader, device, attack_features=True, target_class = None)
    kl_divergence = F.kl_div(ood_after.log(), ood_clean)
    kld = kl_divergence.numpy()        
    return kld


def get_fid(features_adv, features_clean):
    mean1 = np.mean(features_adv, axis=0)
    cov1 = np.cov(features_adv, rowvar=False)

    mean2 = np.mean(features_clean, axis=0)
    cov2 = np.cov(features_clean, rowvar=False)

    mean_diff = mean1 - mean2
    mean_diff_squared = np.dot(mean_diff, mean_diff)

    cov_product = np.dot(cov1, cov2)
    cov_sqrt = linalg.sqrtm(cov_product)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = mean_diff_squared + np.trace(cov1 + cov2 - 2 * cov_sqrt)
    return fid
    

