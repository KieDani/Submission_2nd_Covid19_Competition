import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def plot_conf_mat(gt, pred, labels=None):
    conf_matrix = confusion_matrix(gt, pred.long())
    "@conf_matrix: Confusion matrix as returned from sklearn.metrics.confusion_matrix"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(conf_matrix, cmap="Blues", vmin=0, vmax=conf_matrix.sum())
    for i_true in range(conf_matrix.shape[0]):
        for i_pred in range(conf_matrix.shape[1]):
            v = conf_matrix[i_true, i_pred]
            ax.text(
                x=i_pred,
                y=i_true,
                s=v,
                va="center",
                ha="center",
                size=20,
                color="black" if v < conf_matrix.sum() / 2 else "white",
            )

    ax.set_xlabel("Prediction", fontsize=18)
    ax.set_ylabel("Actual", fontsize=18)

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

    return fig
