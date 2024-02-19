import torch
from sklearn.metrics import roc_auc_score

def test_ood_given_dataloader(model, test_dataloader, non_blocking : bool = False, device = "cpu", verbose = 0, clean_dataset = True):

    model.to(device, non_blocking=non_blocking)
    model.eval()

    if verbose == 1:
        batch_label_list = []
        batch_normality_scores_list = []

    with torch.no_grad():
        if clean_dataset:
            for batch_idx, (x, label) in enumerate(
                    test_dataloader):
                x = x.to(device, non_blocking=non_blocking)
                # print(f"original_targets[:5]: {original_targets[:5]}")
                label = label.to(device, non_blocking=non_blocking)
                pred = model(x)

                # TODO: check below
                normality_scores = torch.max(pred.detach().cpu(), dim=1).values
                # print(f"pred.size(): {pred.size()}")
                # print(f"normality_scores.size(): {normality_scores.size()}")

                if verbose == 1:
                    batch_label_list.append(label.detach().clone().cpu())
                    batch_normality_scores_list.append(normality_scores.detach().clone().cpu())
        else:
            for batch_idx, (x, labels, original_index, poison_indicator, original_targets) in enumerate(test_dataloader):
                x = x.to(device, non_blocking=non_blocking)
                # print(f"original_targets[:5]: {original_targets[:5]}")
                original_targets = original_targets.to(device, non_blocking=non_blocking)
                pred = model(x)

                #TODO: check below
                normality_scores = torch.max(pred.detach().cpu(), dim=1).values
                # print(f"pred.size(): {pred.size()}")
                # print(f"normality_scores.size(): {normality_scores.size()}")

                if verbose == 1:
                    batch_label_list.append(original_targets.detach().clone().cpu())
                    batch_normality_scores_list.append(normality_scores.detach().clone().cpu())

    auc = roc_auc_score(torch.cat(batch_label_list).detach().cpu().numpy(), torch.cat(batch_normality_scores_list).detach().cpu().numpy())

    print(f"auc: {auc}")

    if verbose == 0:
        return None
    elif verbose == 1:
        return auc

def eval_step_msp_auc(
        netC,
        clean_test_dataloader_ood,
        bd_out_test_dataloader_ood,
        bd_all_test_dataloader_ood,
        args,
        result_name_prefix=""
):
    msp_clean_test_auc = test_ood_given_dataloader(netC, clean_test_dataloader_ood, non_blocking=args.non_blocking,
                                               device=args.device,
                                               verbose=1, clean_dataset=True)
    msp_bd_out_test_auc = test_ood_given_dataloader(netC, bd_out_test_dataloader_ood, non_blocking=args.non_blocking,
                                                device=args.device, verbose=1,
                                                clean_dataset=False)
    msp_bd_all_test_auc = test_ood_given_dataloader(netC, bd_all_test_dataloader_ood, non_blocking=args.non_blocking,
                                                device=args.device, verbose=1,
                                                clean_dataset=False)

    msp_auc_result_dict = {
        f"{result_name_prefix}msp_clean_test_auc": msp_clean_test_auc,
        f"{result_name_prefix}msp_bd_out_test_auc": msp_bd_out_test_auc,
        f"{result_name_prefix}msp_bd_all_test_auc": msp_bd_all_test_auc
    }

    return msp_auc_result_dict

