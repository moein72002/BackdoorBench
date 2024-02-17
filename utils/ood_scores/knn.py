import numpy as np
import torch
from tqdm import tqdm
import faiss
from sklearn.metrics import roc_auc_score


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_score_knn_auc(model, device, train_loader, test_loader, bd_test_loader=False):
    model.to(device)
    model.eval()

    train_feature_space = []
    with torch.no_grad():
        for idx, (imgs, target, original_index, poison_indicator, original_targets) in enumerate(train_loader, start=1):
            # print(f"idx: {idx}")
            # print(f"len(imgs): {len(imgs)}")
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        if bd_test_loader:
            for idx, (imgs, _, _, _, original_targets) in tqdm(enumerate(test_loader), desc='Test set feature extracting'):
                imgs = imgs.to(device)
                features = model(imgs)
                test_feature_space.append(features)
                test_labels.append(original_targets)
        else:
            for idx, (imgs, labels) in tqdm(enumerate(test_loader), desc='Test set feature extracting'):
                imgs = imgs.to(device)
                features = model(imgs)
                test_feature_space.append(features)
                test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, -1 * distances) # I multiplied distances(scores) by -1 because here in dist label is 1

    print(f"knn_auc: {auc}")

    return auc

def eval_step_knn_auc(
        netC,
        train_loader,
        clean_test_dataloader_ood,
        bd_out_test_dataloader_ood,
        bd_all_test_dataloader_ood,
        args,
):
    device = args.device
    knn_clean_test_auc = get_score_knn_auc(netC, device, train_loader, clean_test_dataloader_ood, bd_test_loader=False)
    knn_bd_out_test_auc = get_score_knn_auc(netC, device, train_loader, bd_out_test_dataloader_ood, bd_test_loader=True)
    knn_bd_all_test_auc = get_score_knn_auc(netC, device, train_loader, bd_all_test_dataloader_ood, bd_test_loader=True)

    knn_auc_result_dict = {
        "knn_clean_test_auc": knn_clean_test_auc,
        "knn_bd_out_test_auc": knn_bd_out_test_auc,
        "knn_bd_all_test_auc": knn_bd_all_test_auc
    }

    return knn_auc_result_dict