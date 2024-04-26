import faiss
import torch
from BAD.attacks.ood.pgdknn import PGD_KNN_ADVANCED

normalizer =  lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
def get_features(model, loader, after_attack, attack, normalize=False):
    model.eval()
    features = []
    all_labels = []
    
    for data, label in loader:
        data, label = data.to(device), label.to(device) 
        if after_attack:
            data, label = attack(data, label)
        
        feature = torch.squeeze(model.get_features(data))
        c_f = feature.detach().cpu().numpy()
        if normalize:
            c_f = normalizer(c_f)
        features.append(c_f)
        all_labels += torch.where(label == 10, torch.tensor(0), torch.tensor(1)).tolist()
        
    return np.concatenate(features), all_labels

def get_knn_score(model, trainloader, testloader):
    
    train_features, _ = get_features(model, trainloader, False, None, normalize=True)
    attack =  PGD_KNN_ADVANCED(model, train_features, eps=attack_eps, steps=attack_steps, alpha=attack_alpha)
    test_features, all_labels = get_features(model, testloader, True, attack, normalize=True)
    index = faiss.IndexFlatL2(train_features.shape[1])
    index.add(train_features)
    D, _ = index.search(test_features, 2)
    return all_labels, -np.sum(D, axis=1)