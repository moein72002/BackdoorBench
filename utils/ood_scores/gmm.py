import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn import mixture

def get_score_gmm_auc(model, device, gmm, test_loader, bd_test_loader=False):
    model.to(device)
    model.eval()

    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        if bd_test_loader:
            for idx, (imgs, _, _, _, original_targets) in tqdm(enumerate(test_loader),
                                                               desc='Test set feature extracting'):
                imgs = imgs.to(device)
                features, _ = model.get_embeds_and_logit_from_forward(imgs)
                test_feature_space.append(features)
                test_labels.append(original_targets)
        else:
            for idx, (imgs, labels) in tqdm(enumerate(test_loader), desc='Test set feature extracting'):
                imgs = imgs.to(device)
                features, _ = model.get_embeds_and_logit_from_forward(imgs)
                test_feature_space.append(features)
                test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    test_samples_likelihood = gmm.score_samples(test_feature_space)

    auc = roc_auc_score(test_labels, test_samples_likelihood)
    print(f"gmm_auc: {auc}")
    return auc


def eval_step_gmm_auc(
        model,
        train_loader,
        clean_test_dataloader_ood,
        bd_out_test_dataloader_ood,
        bd_all_test_dataloader_ood,
        args,
        n_components=1
):
    device = args.device

    model.to(device)
    model.eval()
    train_feature_space = []
    with torch.no_grad():
        for idx, (imgs, _, _, _, _) in enumerate(train_loader, start=1):
            imgs = imgs.to(device)
            features, _ = model.get_embeds_and_logit_from_forward(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    gmm = mixture.GaussianMixture(n_components=n_components,
                                  max_iter=500,
                                  verbose=1,
                                  n_init=1)
    print("fitting gmm model started")
    gmm.fit(train_feature_space)
    print("fitting gmm model finished")

    gmm_clean_test_auc = get_score_gmm_auc(model, device, gmm, clean_test_dataloader_ood, bd_test_loader=False)
    gmm_bd_out_test_auc = get_score_gmm_auc(model, device, gmm, bd_out_test_dataloader_ood, bd_test_loader=True)
    gmm_bd_all_test_auc = get_score_gmm_auc(model, device, gmm, bd_all_test_dataloader_ood, bd_test_loader=True)

    gmm_auc_result_dict = {
        f"gmm{n_components}_clean_test_auc": gmm_clean_test_auc,
        f"gmm{n_components}_bd_out_test_auc": gmm_bd_out_test_auc,
        f"gmm{n_components}_bd_all_test_auc": gmm_bd_all_test_auc
    }

    print(gmm_auc_result_dict)

    return gmm_auc_result_dict