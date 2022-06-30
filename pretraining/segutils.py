import numpy as np
import torch
import torch.nn as nn
import random
import os
from paths import pretraining_LOGS_PATH


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        dice = diceloss(pred=y_pred, target=y_true, hard_label=False)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, y_true)
        return dice + cross_entropy

class MyBCELoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
        self.device = device

    def forward(self, pred, target):
        dims = [i for i in range(1, len(pred.shape))]
        unit_weight = torch.ones_like(target, device=self.device)
        scale_factor = target.sum(dim=dims) / torch.prod(torch.tensor(target.shape[1:]))
        adaptive_weight = unit_weight
        for i, factor in enumerate(scale_factor):
            adaptive_weight[i] += factor*target[i]
        loss = self.cross_entropy(pred, target) * adaptive_weight
        loss = loss.sum()/adaptive_weight.sum()
        return loss


def diceloss(pred, target, eps=1., hard_label=True):
    if hard_label:
        pred = torch.sigmoid(pred) - 0.5
        pred = torch.heaviside(pred, pred)
    else:
        pred = torch.sigmoid(pred)
    dims = [i for i in range(1, len(pred.shape))]
    up = 2 * (pred * target).sum(dim=dims) + eps
    down = pred.sum(dim=dims) + target.sum(dim=dims) + eps
    result = 1 - up/down
    result = result.sum() / result.shape[0]
    return result

def dice_score_fn(pred, target, eps=1.):
    pred = torch.sigmoid(pred) - 0.5
    pred = torch.heaviside(pred, pred)
    dims = [i for i in range(1, len(pred.shape))]
    # up = 2 * (pred * target).sum(dim=dims) + eps
    # down = pred.sum(dim=dims) + target.sum(dim=dims) + eps
    # result = up / down
    # result = result.sum() / result.shape[0]
    up = 2 * (pred * target).sum(dim=dims)
    down = pred.sum(dim=dims) + target.sum(dim=dims)
    return up.sum(), down.sum()


def IoU(pred, target, eps=1., hard_label=True):
    if hard_label:
        pred = torch.sigmoid(pred) - 0.5
        pred = torch.heaviside(pred, pred)
    else:
        pred = torch.sigmoid(pred)
    dims = [i for i in range(1, len(pred.shape))]
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    # result = (intersection + eps) / (union + eps)
    # result = result.sum() / result.shape[0]
    return intersection.sum(), union.sum()


#the teacher weights are updated
def update_teacher(student, teacher, alpha=0.95):
    with torch.no_grad():
        for name, param in teacher.named_parameters():
            student.state_dict()[name]
            param.data = alpha * param + (1 - alpha) * student.state_dict()[name].data
        return teacher

def set_deterministic(seed=1055):
    # settings based on https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True  # Training damit halb so schnell
    #torch.set_deterministic(True)  # nicht möglich
    #torch.backends.cudnn.benchmark = False
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

def seed_worker(worker_id):
    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def saveModel(model, model_ema, optimizer, epoch, config, identifier):
    savepath = os.path.join(config.LOGS_PATH, 'checkpoints', identifier.split('/')[-1])
    if hasattr(config, 'cv_enabled'):
        if config.cv_enabled:
            foldname = 'cv' + str(config._current_fold)
            savepath = os.path.join(config.LOGS_PATH, 'checkpoints', identifier, foldname)
    os.makedirs(savepath, exist_ok=True)
    savepath = os.path.join(savepath, 'checkpoint_' + str(epoch) + '.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_ema_state_dict': model_ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_state_dict': model_ema.encoder.state_dict()
    }, savepath)

    print('saved model')

def load_model(model, model_ema, optimizer):
    #load model like
    os.path.join()
    savepath = 'PATH/TO/CHECKPOINT'
    identifier = 'IDENTIFIER'
    #identifier = 'test'
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_ema.load_state_dict(checkpoint['model_ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch, identifier

def copy_model(model, model_copy):
    with torch.no_grad():
        for name, param in model_copy.named_parameters():
            model.state_dict()[name]
            param.data = model.state_dict()[name]
