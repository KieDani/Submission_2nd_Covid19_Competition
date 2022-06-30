from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import torch
from training.misc_utilities.plot import plot_conf_mat
from sklearn.metrics import f1_score
from paths import DATA_PATH


def main():
    parser = ArgumentParser()
    parser.add_argument("infsev", choices=["inf", "sev"])
    parser.add_argument("pred")
    parser.add_argument("gt")
    parser.add_argument("output")
    args = parser.parse_args()

    gt = pd.read_csv(args.gt).rename(columns=dict(Inf="inf", Sev="sev"))
    gt["inf"] -= 1
    gt["sev"] -= 1
    pred = pd.read_csv(args.pred).rename(columns=dict(pred_inf="inf", pred_sev="sev"))
    pred["path"] = pred["path"].str.replace(DATA_PATH, "")

    merged = pd.merge(gt, pred, how="inner", left_on="Path", right_on="path", suffixes=("_gt", "_pred"))

    if args.infsev == "inf":
        plot_conf_mat(merged.inf_gt, torch.tensor(merged.inf_pred.tolist()))

        print("F1:", f1_score(merged["inf_gt"], merged["inf_pred"], average="macro"))
        print("Acc:", (merged["inf_gt"] == merged["inf_pred"]).mean())
    else:
        # remove inf=0 cases
        merged = merged[(merged["inf_gt"] > 0) & (merged["sev_gt"] >= 0)]
        p = torch.tensor(merged.sev_pred.tolist())
        plot_conf_mat(merged.sev_gt, p)

        print("F1:", f1_score(merged["sev_gt"], merged["sev_pred"], average="macro"))
        print("Acc:", (merged["sev_gt"] == merged["sev_pred"]).mean())
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
