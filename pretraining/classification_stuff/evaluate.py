import torch
import torch.nn as nn
import sklearn.metrics as metrics
from tqdm import tqdm
from pretraining.classification_stuff.loss import AbstractLoss


def rocauc_safe(actual, pred):
    "Safe ROC AUC that handles edge cases"
    assert len(actual) == len(pred)
    if len(actual) == 0 or len(torch.unique(actual)) == 1:
        return 0.0
    else:
        return metrics.roc_auc_score(actual, pred)


def evaluate(
    eval_loader,
    model: nn.Module,
    loss_fn: AbstractLoss,
    model_uses_metadata: bool,
    device = torch.device("cpu"),
    no_tqdm=False,
    num_at_once=1):

    model.eval()

    all_preds = []
    all_inf_gt = []
    all_sev_gt = []
    all_patients = []

    running_loss = 0.
    running_loss_sev0 = 0.
    running_loss_sev1 = 0.

    with torch.no_grad():
        for sample in tqdm(eval_loader, total=len(eval_loader), leave=True, position=0, desc="Evaluation", disable=no_tqdm):
            v_tensor, age, sex, inf_gt, sev_gt, patient = sample

            all_inf_gt.append(inf_gt)
            all_sev_gt.append(sev_gt)
            all_patients.append(patient)

            loss_scaling = max((v_tensor.shape[0] // num_at_once), 1)

            preds = []
            for i in range(0, v_tensor.shape[0], num_at_once):
                step = min(i + num_at_once, v_tensor.shape[0])
                # select subset for gradient accumulation
                v_ten, s_gt, i_gt = v_tensor[i:step], sev_gt[i:step], inf_gt[i:step]
                v_ten, s_gt, i_gt = v_ten.to(device), s_gt.to(device), i_gt.to(device)
                # metadata
                if model_uses_metadata:
                    a = age[i:step].to(device)
                    s = sex[i:step].to(device)
                else:
                    a = None
                    s = None
                # forward pass
                pred = model(v_ten, a, s)

                running_loss += loss_fn(pred, i_gt, s_gt).cpu().item() / loss_scaling

                # calculate extra loss for severe and non-severe cases
                l0, l1 = loss_fn.partial_loss(pred, i_gt, s_gt)
                running_loss_sev0 += l0.cpu().item() / loss_scaling
                running_loss_sev1 += l1.cpu().item() / loss_scaling

                preds.append(pred.cpu())

            all_preds.append(torch.cat(preds, dim=0))

    all_inf_gt = torch.cat(all_inf_gt)
    all_sev_gt = torch.cat(all_sev_gt)
    all_patients = torch.cat(all_patients)

    running_loss = running_loss / len(all_preds) # do this before torch.cat(all_preds)!
    running_loss_sev0 = running_loss_sev0 / len(all_preds)
    running_loss_sev1 = running_loss_sev1 / len(all_preds)
    all_preds = torch.cat(all_preds)

    pred_inf, pred_sev = loss_fn.finalize(all_preds)

    inf_roc = rocauc_safe(all_inf_gt, pred_inf)
    sev_roc = rocauc_safe(all_sev_gt, pred_sev)

    # for the submission, only patients with a COVID infection are counted
    covpats = (all_inf_gt == 1)
    submission_sev_roc = rocauc_safe(all_sev_gt[covpats], pred_sev[covpats])

    model.train()

    return {
        "auc_inf": inf_roc,
        "auc_sev": sev_roc,
        "auc_sev2": submission_sev_roc,
        "loss": running_loss,
        "loss_sev0": running_loss_sev0,
        "loss_sev1": running_loss_sev1,
        "all_data": {
            "inf_gt": all_inf_gt,
            "inf_pred": pred_inf,
            "sev_gt": all_sev_gt,
            "sev_pred": pred_sev,
            "patients": all_patients,
        },
    }
