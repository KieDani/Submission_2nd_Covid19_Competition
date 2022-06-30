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
from pretraining.segutils import copy_model
from pretraining.all.all_utils import DiceCELoss, MyBCELoss, saveSequential, update_teacher, set_deterministic, seed_worker
from pretraining.semi_supervised.train import validation, prediction, supervised_step
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Train STOIC network')
    # general
    parser.add_argument('--gpu',
                        default="6",
                        help='gpu id',
                        type=str)
    parser.add_argument('--model',
                        help='model used for training, possible are: unet and upernext',
                        default="upernext",
                        type=str,
                        required=True)
    parser.add_argument('--nick',
                        default='',
                        help='Prepend a nickname to the output directory',
                        type=str)
    args = parser.parse_args()
    args.cv = False
    args.mode = 4
    args.imsize = '256'
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
    dice_score_ema = validation(model, teacher, val_loader, writer=writer, epoch=epoch, device=config.DEVICE,
               num_at_once=num_at_once, loss_fn=loss_fn)
    # dice_score_ema = -1e6
    print('Epoch ', epoch)
    copy_model(model, saved_model)
    copy_model(teacher, saved_teacher)
    best = [dice_score_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
    if iteration == 0:
        min_num_epochs = 60 if config.MODEL_NAME == 'upernext' else 100
    else:
        min_num_epochs = 10
    for epoch in range(1, config.MAX_EPOCHS):
        for input, label in train_loader:
            optimizer.zero_grad()
            supervised_step(input, label, model, loss_fn, writer, iteration, config, num_at_once=num_at_once)
            optimizer.step()
            alpha = 0.99 if epoch < min_num_epochs else 0.995
            update_teacher(model, teacher, alpha=alpha)
            iteration +=1

        dice_score_ema = validation(model, teacher, val_loader, writer=writer, epoch=epoch, device=config.DEVICE,
                   num_at_once=num_at_once, loss_fn=loss_fn)
        if dice_score_ema > best[0]:
            copy_model(model, saved_model)
            copy_model(teacher, saved_teacher)
            best = [dice_score_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
        elif epoch - best[1] > 20 and epoch > min_num_epochs:
            break

    return best






def create_labels(model, data_loader, config, identifier):
    savepath = os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'unsupervised_labels_seg')
    correct_shape = (256, 256, 128) if config.IMAGE_SIZE == '256' else (128, 128, 64)
    os.makedirs(savepath, exist_ok=True)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    model.to(config.DEVICE)
    model.eval()
    with torch.no_grad():
        for input, data_name in data_loader:
            for i in range(0, input.shape[0], config.modelconfig.num_at_once):
                step = min(i + config.modelconfig.num_at_once, input.shape[0])
                inp, dat_nam = input[i:step], data_name[i:step]
                inp = inp.to(config.DEVICE)
                predict = model(inp)
                predict = nn.functional.interpolate(predict, correct_shape, mode='trilinear')
                predict = predict.cpu().numpy()
                for j in range(inp.shape[0]):
                    pred = predict[j, 0]
                    pred = sigmoid(pred) - 0.5
                    pred = np.heaviside(pred, pred)
                    data_path = os.path.join(savepath, dat_nam[j])
                    if os.path.exists(data_path):
                        os.remove(data_path)
                    np.save(data_path, pred)

            # input = input.to(config.DEVICE)
            # predict = model(input)
            # predict = nn.functional.interpolate(predict, correct_shape, mode='trilinear')
            # predict = predict.cpu().numpy()
            # for i in range(input.shape[0]):
            #     pred = predict[i, 0]
            #     pred = sigmoid(pred) - 0.5
            #     pred = np.heaviside(pred, pred)
            #     data_path = os.path.join(savepath, data_name[i])
            #     if os.path.exists(data_path):
            #         os.remove(data_path)
            #     np.save(data_path, pred)
    model.train()
    return savepath






def run(config, args):
    model = get_model(config)
    modelconfig = config.modelconfig
    if config.MODEL_NAME == 'upernext':
        model_name = 'convnextransformer' if modelconfig.use_transformer else 'convnext'
    else:
        model_name = 'unet'
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
    normalize_tcia = Normalize3D(mean=0.220458088175389,
                                 std=0.30216501129178236) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_supervised_val = My_Compose([Zoom([zoomsize, zoomsize, zoomsize]), ToTensor3D(), normalize_tcia])
    transform_supervised_train = My_Compose([get_transform(config.IMAGE_SIZE), ToTensor3D(),
                                             normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_tcia = My_Compose([get_transform(config.IMAGE_SIZE), ToTensor3D(),
                                              normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    train_dataset = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_train, train=True)
    my_val_dataset = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_val, train=False)
    #TODO create val-dataset with variable label-path
    val_dataset_unlabeled = ValidationDataset(path=config.DATA_PATH, transform=transform_supervised_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.Batch_SIZE, num_workers=config.WORKERS,
                                               shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    my_val_loader = torch.utils.data.DataLoader(my_val_dataset, batch_size=config.Batch_SIZE,
                                                num_workers=config.WORKERS,
                                                shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
    if config.LOSS == 'dicece':
        loss_fn = DiceCELoss()
    elif config.LOSS == 'balancedce':
        loss_fn = MyBCELoss(device=config.DEVICE)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    if config.UNSUPERVISED_DATA == 'stoic':
        SequentialDataset = SequentialDataset_stoic
    elif config.UNSUPERVISED_DATA == 'mosmed':
        SequentialDataset = SequentialDataset_mosmed
    else:
        SequentialDataset = SequentialDataset_tcia


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
        unsup_dataset = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia)
        unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
        labelpath = create_labels(teacher, unsup_loader, config, identifier)
        unsup_dataset = SequentialDataset(path=config.DATA_PATH, label_path=labelpath, transform=transform_unsupervised_tcia)
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


def main():
    print('PID:', os.getpid())
    set_deterministic()
    args = parse_args()
    config = BaseConfig(args)
    config.modelconfig = get_modelconfig(config)

    config.MAX_EPOCHS = 300
    run(config, args)


if __name__ == '__main__':
    main()