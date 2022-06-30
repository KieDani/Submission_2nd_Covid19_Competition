import math
from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractLoss(ABC):
    @abstractmethod
    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        raise NotImplementedError("__call__ not implemented")

    @abstractmethod
    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        "Transforms the output batch of the model into the final prediction that is used for this loss"
        raise NotImplementedError()

    def partial_loss(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits the loss into two parts based on the severity state"""
        sev0 = sev_gt == 0
        if sev0.any():
            sev0_loss = (
                self(output[sev0], inf_gt[sev0], sev_gt[sev0])
                * (sev0).sum()
                / len(sev_gt)
            )
        else:
            sev0_loss = torch.tensor(0.0)

        sev1 = sev_gt == 1
        if sev1.any():
            sev1_loss = (
                self(output[sev1], inf_gt[sev1], sev_gt[sev1])
                * (sev1).sum()
                / len(sev_gt)
            )
        else:
            sev1_loss = torch.tensor(0.0)

        return sev0_loss, sev1_loss


class BCEInfSev(AbstractLoss):
    def __init__(self, sev_ratio: float = 0.5) -> None:
        self.base_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.sev_ratio = sev_ratio
        self._orig_sev_ratio = sev_ratio

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        out_inf = output[:, 0]
        out_sev = output[:, 1]
        return (
            (1 - self.sev_ratio) * self.base_loss(out_inf, inf_gt.float())
            + self.sev_ratio
            * self.base_loss(
                out_sev, sev_gt.float(), pos_weight=torch.tensor(self.pos_weight)
            )
        ).mean()

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 2
        x = torch.sigmoid(output)
        inf, sev = x[:, 0], x[:, 1]
        return inf, sev

    def set_moving_ratio(self, epoch: int, end_epoch: int):
        r = min(1, epoch / end_epoch)
        self.sev_ratio = self._orig_sev_ratio + (1 - self._orig_sev_ratio) * r


class BCEInfSevExclude(AbstractLoss):
    """Mixture of BCE on inf and BCE on sev. Cases with inf==0 are excluded from the
    sev-loss (loss is set to 0)
    """

    def __init__(self, sev_ratio: float = 0.5, pos_weight=None):
        self.sev_ratio = sev_ratio
        self._orig_sev_ratio = sev_ratio
        self.pos_weight = pos_weight

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        inf_pred = output[:, 0]
        sev_pred = output[:, 1]

        inf_loss = F.binary_cross_entropy_with_logits(inf_pred, inf_gt.float())

        inf = inf_gt == 1
        if inf.any():
            sev_loss = F.binary_cross_entropy_with_logits(
                sev_pred[inf],
                sev_gt[inf].float(),
                pos_weight=torch.tensor(self.pos_weight),
            )
        else:
            sev_loss = 0

        return (1 - self.sev_ratio) * inf_loss + self.sev_ratio * sev_loss

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 2
        x = torch.sigmoid(output)
        inf, sev = x[:, 0], x[:, 1]
        return inf, sev

    def set_moving_ratio(self, epoch: int, end_epoch: int):
        r = min(1, epoch / end_epoch)
        self.sev_ratio = self._orig_sev_ratio + (1 - self._orig_sev_ratio) * r


class CrossEntropy3(AbstractLoss):
    def __init__(self, weight: torch.Tensor = None):
        self.weight = weight

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        three_class = inf_gt + sev_gt
        return F.cross_entropy(output, three_class.long(), weight=self.weight)

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 3
        preds = torch.softmax(output, dim=-1)
        sev = preds[:, 2]
        inf = preds[:, 1] + sev
        return inf, sev

    def set_moving_weight(self, epoch: int, end_epoch: int):
        device = self.weight.device
        progress = min(1, epoch / end_epoch)
        w0 = torch.ones(3) / 3
        w1 = torch.tensor([0.0, 0.5, 0.5])
        self.weight = w0 + progress * (w1 - w0)
        self.weight.to(device)


class CrossEntropyInfOnly(AbstractLoss):
    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        return F.cross_entropy(output, inf_gt.long())

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 2
        preds = torch.softmax(output, dim=-1)
        inf = preds[:, 1]
        sev = torch.zeros_like(inf)
        return inf, sev


class CrossEntropySevOnly(AbstractLoss):
    def __init__(self, pos_weight=None):
        self.pos_weight = pos_weight
        if pos_weight is not None:
            print('posweight is not supported for CrossEntropySevOnly!')

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        if self.pos_weight is not None:
            weight = torch.tensor([1.0, self.pos_weight])
            return F.cross_entropy(output, sev_gt.long())#, weight=weight)
        else:
            return F.cross_entropy(output, sev_gt.long())

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 2
        preds = torch.softmax(output, dim=-1)
        sev = preds[:, 1]
        inf = torch.zeros_like(sev)
        return inf, sev

#4 classes, 4 output neurons
class CrossEntropySevECCV(AbstractLoss):
    def __init__(self, weighted=True):
        self.weighted = weighted
        self.ce = self.weighted_ce if weighted else torch.nn.CrossEntropyLoss()

    #y with dimensions (batch_size, 1) and values between 0 and 3
    def weighted_ce(self, x, y):
        l = torch.empty(x.shape[0])
        sm = torch.nn.functional.softmax(x, dim=-1)
        for n in range(x.shape[0]):
            pred = sm[n]
            weight = (torch.argmax(pred) - y[n])**2 + 1
            l[n] = -weight * torch.log(pred)[y[n]]
        return l

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        if self.weighted:
            return torch.mean(self.ce(output, sev_gt.long()))
        else:
            return self.ce(output, sev_gt.long())
        #return torch.mean(self.ce(output, sev_gt.long()))

    #I do not know if the dimensions are correct!
    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 4
        sev = torch.argmax(output, dim=-1)
        inf = torch.zeros_like(sev)
        return inf, sev


#4 classes, 4 output neurons
class BalancedCrossEntropySevECCV(AbstractLoss):
    def __init__(self):
        weight = torch.tensor([0.15, 0.21, 0.15, 0.49], device='cuda') * 4
        self.ce = torch.nn.CrossEntropyLoss(weight=weight)

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        return self.ce(output, sev_gt.long())
        #return torch.mean(self.ce(output, sev_gt.long()))

    #I do not know if the dimensions are correct!
    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 4
        sev = torch.argmax(output, dim=-1)
        inf = torch.zeros_like(sev)
        return inf, sev



#2 classes, 2 output neurons
class CrossEntropyInfECCV(AbstractLoss):
    def __init__(self):
        self.ce = torch.nn.CrossEntropyLoss()

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        return self.ce(output[:, :2], inf_gt.long())
        #return torch.mean(self.ce(output, sev_gt.long()))

    #I do not know if the dimensions are correct!
    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 2
        output = output[:, :2]
        inf = torch.argmax(output, dim=-1)
        sev = torch.zeros_like(inf)
        return inf, sev


class BalancedCrossEntropyInfECCV(AbstractLoss):
    def __init__(self):
        weight = torch.tensor([0.44, 0.56], device='cuda') * 2
        self.ce = torch.nn.CrossEntropyLoss(weight=weight)

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        return self.ce(output, inf_gt.long())
        #return torch.mean(self.ce(output, sev_gt.long()))

    #I do not know if the dimensions are correct!
    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 2
        inf = torch.argmax(output, dim=-1)
        sev = torch.zeros_like(inf)
        return inf, sev



#4 classes, 1 output neurons
class L2RegressionECCV(AbstractLoss):
    def __init__(self):
        self.target_values = [0.05, 1/3, 2/3, 0.95]
        #self.bce = torch.nn.L1Loss()

    def bce(self, x, y):
        l = torch.empty(x.shape[0])
        si = torch.sigmoid(x)
        for n in range(x.shape[0]):
            pred = si[n]
            #l[n] = -y[n] * torch.log(pred)
            l[n] = (y[n] - pred)**2
        return l

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        # sev_gt is in range 0 to 3
        target = torch.empty_like(sev_gt).float()
        for n in range(sev_gt.shape[0]):
            target[n] = self.target_values[sev_gt[n]]
        #print(output)
        loss = self.bce(output[:, 0], target)
        loss = torch.mean(loss)
        return loss

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert output.size(1) == 4
        preds = output[:, 0]
        sev = torch.empty_like(preds)
        for n in range(preds.shape[0]):
            pred = torch.sigmoid(preds[n])
            differences = torch.tensor([(y - pred)**2 for y in torch.tensor(self.target_values)])
            #differences = self.bce(torch.stack([preds[n] for i in self.target_values]), torch.tensor(self.target_values))
            sev[n] = torch.argmin(differences)
            #sev[n] = torch.argmin((torch.tensor(self.target_values) - torch.stack([preds[n] for i in self.target_values], dim=-1))**2, dim=-1)
        inf = torch.zeros_like(sev)
        return inf, sev


class BinaryCrossEntropySevOnly(AbstractLoss):
    def __init__(self, pos_weight=None):
        if pos_weight is not None:
            self.pos_weight = torch.tensor(pos_weight)
        else:
            self.pos_weight = None

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        sev_pred = output[:, 1]
        return F.binary_cross_entropy_with_logits(
            sev_pred, sev_gt.float(), pos_weight=self.pos_weight
        )

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = torch.sigmoid(output)
        inf, sev = preds[:, 0], preds[:, 1]
        return inf, sev


class AucLoss(AbstractLoss):
    def __init__(self, compare="sigmoid"):
        if compare == "sigmoid":
            self.compare = lambda p, n: 1 - torch.sigmoid(p - n).mean()
        elif compare == "l2":
            self.compare = lambda p, n: -torch.square(p - n).mean()
        else:
            raise ValueError("Invalid compare")

    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        pos_pred = output[:, 0]
        neg_pred = output[:, 1]
        return self.compare(pos_pred, neg_pred)

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = torch.sigmoid(output)
        inf, sev = preds[:, 0], preds[:, 1]
        return inf, sev

    def partial_loss(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(torch.nan), torch.tensor(torch.nan)


class BatchAucLoss(AbstractLoss):
    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        pos_pred = output[:, 0]
        neg_pred = output[:, 1]
        s = 0.0
        # TODO: performance
        for n in neg_pred:
            s += torch.sigmoid(pos_pred - n).sum()

        return 1 - (s / len(pos_pred) / len(neg_pred))

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = torch.sigmoid(output)
        inf, sev = preds[:, 0], preds[:, 1]
        return inf, sev

    def partial_loss(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(torch.nan), torch.tensor(torch.nan)


class AucBceSevLoss(AbstractLoss):
    def __call__(
        self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor
    ):
        pos_pred = output[:, 0]
        neg_pred = output[:, 1]
        auc_part = 1 - torch.sigmoid(pos_pred - neg_pred).mean()
        bce_pos_part = F.binary_cross_entropy_with_logits(
            pos_pred, torch.ones_like(pos_pred)
        )
        bce_neg_part = F.binary_cross_entropy_with_logits(
            neg_pred, torch.zeros_like(neg_pred)
        )
        return 0.5 * auc_part + 0.25 * bce_pos_part + 0.25 * bce_neg_part

    def finalize(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # this loss only works in siamese mode
        # in siamese mode, only preds[:, 1] is used -> sev
        # the output of the finalize() method is not the output of __call__()
        preds = torch.sigmoid(output)
        inf, sev = preds[:, 0], preds[:, 1]
        return inf, sev

    def partial_loss(self, output: torch.Tensor, inf_gt: torch.Tensor, sev_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(torch.nan), torch.tensor(torch.nan)


def get_loss(
    loss_name: str, pos_weight=None, ce3_weight: torch.Tensor = None
) -> AbstractLoss:
    loss_name = loss_name.lower()
    if loss_name == "bce_inf_sev":
        return BCEInfSev()
    if loss_name == "3ce":
        return CrossEntropy3(weight=ce3_weight)
    if loss_name == "ce_inf":
        return CrossEntropyInfOnly()
    if loss_name == "ce_sev":
        return CrossEntropySevOnly(pos_weight=pos_weight)
    if loss_name == "bce_sev":
        return BinaryCrossEntropySevOnly(pos_weight=pos_weight)
    if loss_name == "bce_inf_sev_exclude":
        return BCEInfSevExclude(pos_weight=pos_weight)
    if loss_name == "auc":
        return AucLoss()
    if loss_name == "auc_l2":
        return AucLoss(compare="l2")
    if loss_name == "batch_auc":
        return BatchAucLoss()
    if loss_name == "auc_bce_sev":
        return AucBceSevLoss()
    if loss_name == "ce_sev_eccv":
        return CrossEntropySevECCV(weighted=False)
    if loss_name == "balce_sev_eccv":
        return BalancedCrossEntropySevECCV()
    if loss_name == 'wce_sev_eccv':
        return CrossEntropySevECCV(weighted=True)
    if loss_name == 'l2_sev_eccv':
        return L2RegressionECCV()
    if loss_name == "ce_inf_eccv":
        return CrossEntropyInfECCV()
    if loss_name == 'balce_inf_eccv':
        return BalancedCrossEntropyInfECCV()

    raise KeyError("Unknown loss")


def assert_siamese_compatibility(loss_fn: AbstractLoss, siamese_mode: bool):
    "Make sure `loss_fn` is compatible with siamese mode"
    siam_loss = (
        isinstance(loss_fn, AucLoss)
        or isinstance(loss_fn, BatchAucLoss)
        or isinstance(loss_fn, AucBceSevLoss)
    )
    assert siam_loss == siamese_mode, "Loss is not compatible with chosen siamese mode"
