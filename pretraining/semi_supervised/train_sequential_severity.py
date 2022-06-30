import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pretraining.config.config import BaseConfig
from pretraining.config.modelconfig import get_model, get_modelconfig
from pretraining.data.datasets import SupervisedDataset, ValidationDataset, SequentialDataset_tcia, SequentialDataset_stoic, SequentialDataset_mosmed
from pretraining.data.transforms3D import ToTensor3D, Normalize3D, elastic_wrapper, elastic_deform_robin
from pretraining.data.transforms3D import get_transform, Zoom
from pretraining.data.transforms3D import Compose as My_Compose
from pretraining.data.transforms3D import Identity as My_Identity
import argparse
from pretraining.all.all_utils import DiceCELoss, MyBCELoss, saveSequential, update_teacher, set_deterministic, seed_worker
from pretraining.segutils import copy_model
import shutil
import pretraining.classification_stuff.evaluate as ev
from pretraining.classification_stuff.loss import BinaryCrossEntropySevOnly, CrossEntropySevOnly
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Train STOIC network')
    # general
    parser.add_argument('--gpu',
                        default="6",
                        help='gpu id',
                        type=str)
    parser.add_argument('--nick',
                        default='',
                        help='Prepend a nickname to the output directory',
                        type=str)
    args = parser.parse_args()
    args.cv = False
    args.mode = 4
    args.imsize = '256'
    args.model = 'multinext'
    return args


def train_step(train_loader, val_loader, base_model, loss_fn, config, args, identifier, iteration):
    writer = SummaryWriter(identifier)
    modelconfig = config.modelconfig
    model = get_model(config)
    copy_model(base_model, model)
    teacher = get_model(config)
    copy_model(model, teacher)
    saved_model = get_model(config)
    saved_teacher = get_model(config)
    model = model.to(config.DEVICE)
    teacher = teacher.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=modelconfig.learning_rate)

    iteration, epoch = 0, 0
    num_at_once = modelconfig.num_at_once if hasattr(modelconfig, 'num_at_once') else 1
    auc_ema, __ = validation(val_loader, teacher, config, loss_fn, writer=writer, epoch=epoch)
    # dice_score_ema = -1e6
    print('Epoch ', epoch)
    copy_model(model, saved_model)
    copy_model(teacher, saved_teacher)
    best = [auc_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
    if iteration == 0:
        min_num_epochs = 16 # if config.MODEL_NAME == 'upernext' else 100
    else:
        min_num_epochs = 8
    for epoch in range(1, config.MAX_EPOCHS):
        for input, __, label in train_loader:
            optimizer.zero_grad()
            #TODO change supervised step
            supervised_step(input, label, model, loss_fn, writer, iteration, config, num_at_once=num_at_once)
            optimizer.step()
            alpha = 0.99 if epoch < min_num_epochs else 0.995
            update_teacher(model, teacher, alpha=alpha)
            iteration +=1

        auc_ema, __ = validation(val_loader, teacher, config, loss_fn, writer=writer, epoch=epoch)
        if auc_ema > best[0]:
            copy_model(model, saved_model)
            copy_model(teacher, saved_teacher)
            best = [auc_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
        elif epoch - best[1] > 5 and epoch > min_num_epochs:
            break

    return best


def supervised_step(input, label, model, loss_fn, writer, iteration, config, num_at_once):
    input, label = input.to(config.DEVICE), label.to(config.DEVICE)
    loss_sum = 0
    for i in range(0, label.shape[0], num_at_once):
        step = min(i + num_at_once, label.shape[0])
        inp, lab = input[i:step], label[i:step]
        pred, __1, __2, __3 = model(inp, mode='classification')
        loss = loss_fn(pred, lab, lab) / max((label.shape[0] // num_at_once), 1)
        loss_sum += loss.item()
        loss.backward()
    #print('train loss', loss_sum)
    writer.add_scalar('train loss', loss_sum, iteration)


def validation(data_loader, model, config, loss_fn, writer, epoch):
    val_results = evaluate(data_loader, model, loss_fn, config, num_at_once = 1, mode='stoic')
    #writer.add_scalar('validation loss', val_results["loss"], iteration)
    writer.add_scalar('AUC EMA', val_results["auc_sev2"], epoch)
    return val_results["auc_sev2"], val_results["loss"]



def create_labels(model, data_loader, config, identifier):
    savepath = os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'unsupervised_labels_sev')
    os.makedirs(savepath, exist_ok=True)
    model.to(config.DEVICE)
    model.eval()
    probSev, patID = [], []
    with torch.no_grad():
        for input, data_name, __ in data_loader:
            for i in range(0, input.shape[0], config.modelconfig.num_at_once):
                step = min(i + config.modelconfig.num_at_once, input.shape[0])
                inp, dat_nam = input[i:step], data_name[i:step]
                inp = inp.to(config.DEVICE)
                predict, __1, __2, __3 = model(inp, mode='classification')
                predict = torch.nn.functional.softmax(predict, dim=-1)
                predict = torch.argmax(predict, dim=-1)
                predict = predict.cpu().numpy()
                for j in range(inp.shape[0]):
                    pred = predict[j]
                    probSev.append(int(pred))
                    patID.append(dat_nam[j].strip('.npy'))

    data_path = os.path.join(savepath, 'reference_unsupervised.csv')
    if os.path.exists(data_path):
        os.remove(data_path)
    df = pd.DataFrame(data={'PatientID': patID, 'probSevere': probSev})
    df.to_csv(data_path)

    model.train()
    return savepath






def run(config, args):
    model = get_model(config)
    modelconfig = config.modelconfig
    model_name = 'convnext'
    if hasattr(modelconfig, 'size') and modelconfig.size == 'small':
        model_name += 'Small'
    elif hasattr(modelconfig, 'size') and modelconfig.size == 'base':
        model_name += 'Base'
    elif hasattr(modelconfig, 'size') and modelconfig.size == 'micro':
        model_name += 'Micro'
    identifier = ''.join((model_name, '_sequential_', config.LOSS))
    supaug = '_supaug' if config.SUPERVISED_AUGMENTATIONS else ''
    identifier = ''.join((args.nick, '_', identifier, '_', config.UNSUPERVISED_DATA, '_resized', str(config.IMAGE_SIZE),
                          supaug, '_', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    identifier = os.path.join(config.LOGS_PATH, identifier)
    if config.DO_NORMALIZATION: print('data is normalized')
    normalize_stoic = Normalize3D(mean=0.3140890660228576,
                                 std=0.3156977397851822) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_supervised_val = My_Compose([Zoom([zoomsize, zoomsize, zoomsize]), ToTensor3D(), normalize_stoic])
    transform_supervised_train = My_Compose([get_transform(config.IMAGE_SIZE), ToTensor3D(),
                                             normalize_stoic]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised = My_Compose([get_transform(config.IMAGE_SIZE), ToTensor3D(),
                                              normalize_stoic]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    train_dataset = SequentialDataset_stoic(path=config.DATA_PATH, transform=transform_supervised_train, train=True, severity=True)
    my_val_dataset = SequentialDataset_stoic(path=config.DATA_PATH, transform=transform_supervised_train, train=False, severity=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.Batch_SIZE, num_workers=config.WORKERS,
                                               shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    my_val_loader = torch.utils.data.DataLoader(my_val_dataset, batch_size=config.Batch_SIZE,
                                                num_workers=config.WORKERS,
                                                shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
    loss_fn = CrossEntropySevOnly(pos_weight=None) #if config.LOSS_CLS == 'ce' else BinaryCrossEntropySevOnly(pos_weight=1.0)

    assert config.UNSUPERVISED_DATA in ['tcia', 'mosmed']
    SequentialDataset = SequentialDataset_tcia if config.UNSUPERVISED_DATA == 'tcia' else SequentialDataset_mosmed

    print('----------------------')
    print('Sequential step', 0)
    print('Training with real labels')
    ident = os.path.join(identifier, 'sup'+str(0))
    best = train_step(train_loader, my_val_loader, model, loss_fn, config, args, ident, iteration=0)
    teacher = best[3]
    for i in range(1, 10):
        print('----------------------')
        print('Sequential step', i)
        print('creating new labels')
        unsup_dataset = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised, severity=False)
        unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
        labelpath = create_labels(teacher, unsup_loader, config, identifier)
        unsup_dataset = SequentialDataset(path=config.DATA_PATH, label_path=labelpath, transform=transform_unsupervised, severity=True)
        unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)

        print('Training with pseudo labels')
        ident = os.path.join(identifier, 'unsup' + str(i))
        new_best = train_step(unsup_loader, my_val_loader, model, loss_fn, config, args, ident, iteration=i)
        teacher = new_best[3]
        model_copy = get_model(config)
        copy_model(model, model_copy)

        print('Training with real labels')
        ident = os.path.join(identifier, 'sup' + str(i))
        new_best = train_step(train_loader, my_val_loader, teacher, loss_fn, config, args, ident, iteration=i)
        teacher = new_best[3]
        model_copy = get_model(config)
        copy_model(model, model_copy)


        if best[0] > new_best[0]:
            print('No improvement in step ', i)
            break
        best = new_best.copy()
        saveSequential(teacher, i, config, identifier)

    shutil.rmtree(path=os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'checkpoints'), ignore_errors=True)
    saveSequential(best[3], None, config, identifier)



#copied from module evaluate -> litte changes were needed
def evaluate(eval_loader, model, loss_fn, config, num_at_once=1, mode='stoic'):

    model.eval()

    all_preds = []
    all_sev_gt = []
    all_patients = []

    running_loss = 0.
    running_loss_sev0 = 0.
    running_loss_sev1 = 0.

    with torch.no_grad():
        for sample in eval_loader:
            input, seg_gt, sev_gt = sample


            all_sev_gt.append(sev_gt)

            loss_scaling = max((input.shape[0] // num_at_once), 1)

            preds = []
            for i in range(0, input.shape[0], num_at_once):
                step = min(i + num_at_once, input.shape[0])
                # select subset for gradient accumulation
                v_ten, s_gt, i_gt = input[i:step], sev_gt[i:step], torch.ones_like(sev_gt[i:step])
                v_ten, s_gt, i_gt = v_ten.to(config.DEVICE), s_gt.to(config.DEVICE), i_gt.to(config.DEVICE)
                # metadata
                a = None
                s = None
                # forward pass
                pred, __1, __2, __3 = model(v_ten, a, s, mode='classification')

                running_loss += loss_fn(pred, i_gt, s_gt).cpu().item() / loss_scaling

                # calculate extra loss for severe and non-severe cases
                l0, l1 = loss_fn.partial_loss(pred, i_gt, s_gt)
                running_loss_sev0 += l0.cpu().item() / loss_scaling
                running_loss_sev1 += l1.cpu().item() / loss_scaling

                preds.append(pred.cpu())

            all_preds.append(torch.cat(preds, dim=0))

    all_sev_gt = torch.cat(all_sev_gt)

    running_loss = running_loss / len(all_preds) # do this before torch.cat(all_preds)!
    running_loss_sev0 = running_loss_sev0 / len(all_preds)
    running_loss_sev1 = running_loss_sev1 / len(all_preds)
    all_preds = torch.cat(all_preds)

    pred_inf, pred_sev = loss_fn.finalize(all_preds)

    sev_roc = ev.rocauc_safe(all_sev_gt, pred_sev)

    submission_sev_roc = ev.rocauc_safe(all_sev_gt[:], pred_sev[:])

    model.train()

    return {
        "auc_sev": sev_roc,
        "auc_sev2": submission_sev_roc,
        "loss": running_loss,
        "loss_sev0": running_loss_sev0,
        "loss_sev1": running_loss_sev1,
        "all_data": {
            "inf_pred": pred_inf,
            "sev_gt": all_sev_gt,
            "sev_pred": pred_sev,
        },
    }


def main():
    print('PID:', os.getpid())
    set_deterministic()
    args = parse_args()
    config = BaseConfig(args)
    config.modelconfig = get_modelconfig(config)

    config.MAX_EPOCHS = 75
    run(config, args)


if __name__ == '__main__':
    main()