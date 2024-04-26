import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np

from numpy.linalg import norm
from tqdm import tqdm
from BAD.eval.eval import evaluate
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def ood_sanity_check(testloader, adv_check=True, clean_check=True, target=None):
    print("Sanity check started")
        
    # Sanity Check for Clean OOD Detection
    if clean_check:
        print("Performing sanity check for clean ood detection performance")
        clean_roc = get_ood(clean_model, testloader, target=target, attack_eps=0, progress=DEBUG)
        bd_roc = get_ood(bad_model, testloader, target=target, attack_eps=0, progress=DEBUG)
        print("Clean auroc with ood detection:", clean_roc)
        print("BD auroc on ood detection:", bd_roc)
    
    # Sanity Check for Adversarial Attack
    if adv_check:
        print("Performing sanity check for adversarial attack")
        clean_roc = get_ood(clean_model, testloader, target=target, attack_eps=32/255, progress=DEBUG)
        bd_roc = get_ood(bad_model, testloader, target=target, attack_eps=32/255, progress=DEBUG)
        print("Clean auroc with large epsilon:", clean_roc)
        print("BD auroc with large epsilon:", bd_roc)
        
    print("End of Sanity Check")

def find_min_eps(evaluator, thresh, eps_lb=0, eps_ub=1, max_error=1e-3, proportional=False, log=False):
    initial_perf = evaluator(None)
    if proportional:
        thresh *= initial_perf
    
    l = eps_lb
    r = eps_ub
    
    while r-l > max_error:
        if log:
            print(l, r)
        mid = (r+l)/2
        auc = evaluator(mid)
        if auc < thresh:
            r = mid
        else:
            l = mid
    return l


def update_attack_params(attack_dict, eps=None, steps=None):
    if eps is not None:
        attack_dict['eps'] = eps
    if steps is not None:
        attack_dict['steps'] = steps
    attack_dict['alpha'] = 2.5 * attack_dict['eps'] / attack_dict['steps']
    return attack_dict

def find_best_gap(m1, m2, evaluator, config, thresh=0.4, log=False):
    
    print("Working on config:", config['title'])
    
    best_result = {
        1: 0,
        2: 0,
        'gap': -100,
    }
    
    attack_class = config['attack']
    
    eps_lb = config.get('eps_lb')
    if eps_lb is None:
        eps_lb = 0
    
    eps_ub = config.get('eps_ub')
    if eps_ub is None:
        eps_ub = find_eps_upperbound(lambda eps: 
            evaluator(m1, attack=attack_class(m1, **(get_attack_params(eps) | config['attack_params']))), thresh, log=log)
        
    eps_steps = config.get('eps_steps')
    if eps_steps is None:
        eps_steps = 10
    
    
    epsilons = torch.linspace(eps_lb, eps_ub, eps_steps * int(255 * eps_ub)).tolist()
    gaps = []

    for eps in epsilons:
        if log:
            print("Working on epsilon", eps * 255)
        
        attack_params = get_attack_params(eps) | config['attack_params']
        
        attack1 = attack_class(m1, **attack_params)
        attack2 = attack_class(m2, **attack_params)

        score1 = evaluator(m1, attack1)
        score2 = evaluator(m2, attack2)
        
        gap = score1 - score2
        
        gaps.append(gap)
        
        if log:
            print(f'Score 1: {score1}')
            print(f'Score 2: {score2}')

        if gap > best_result['gap']:
            best_result['gap'] = gap
            best_result[1] = score1
            best_result[2] = score2
            
            print(f"{config['title']} --- Best gap until eps = {eps * 255} is {best_result['gap']}")    
    
    plot_process([e*255 for e in epsilons], gaps, config['title'])
    return best_result

def get_mean_features(model, dataloader, target_label):
    in_features = None
    for data, labels in dataloader:
        data = data.to(device)
        labels = labels.to(device)
        data_features = model.get_features(data).detach().cpu()
        new_features = torch.index_select(data_features, 0,
                                          torch.tensor([i for i, x in enumerate(labels) if x]))
        if in_features is not None:
            in_features = torch.cat((in_features, new_features))
        else:
            in_features = new_features
    return torch.mean(in_features, dim=0)

def get_features_mean_dict(loader, feature_extractor):
    embeddings_dict = {}
    counts_dict = {}
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        features = feature_extractor(data, target).detach().cpu().numpy()
        for i in range(len(target)):
            label = target[i].item()
            if label not in embeddings_dict:
                embeddings_dict[label] = features[i]
                counts_dict[label] = 1
            else:
                embeddings_dict[label] += features[i]
                counts_dict[label] += 1
    
    mean_embeddings_dict = {}
    for label in embeddings_dict:
        mean_embeddings_dict[label] = (embeddings_dict[label] / counts_dict[label])
    
    return mean_embeddings_dict

def get_ood_outputs(model, loader, DEVICE, progress=False, attack_features=True, target_class = None):
    outputs = []

    labels = []
    
    model.eval()
    model.to(device)

    
    progress_bar = loader
    if progress:
        progress_bar = tqdm(loader, unit="batch")
        
    for data, label in progress_bar:
        data, label = data.to(DEVICE), label.to(DEVICE)
        if attack_features:
            attack = PGD(model, target_class=target_class, eps=attack_eps, alpha=attack_alpha, steps=attack_steps)
            data = attack(data, label)
        output = model(data)
        output = output[label==10]
        output = torch.softmax(output, dim=1)
        outputs.append(output.detach().cpu())
    o = torch.cat(outputs, dim=0)

    return o
