import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pretraining.config.config import AllConfig
from pretraining.config.modelconfig import get_model, get_modelconfig
from pretraining.data.datasets import SupervisedDataset, ValidationDataset, SequentialDataset_tcia, SequentialDataset_stoic, SequentialDataset_mosmed, SequentialDataset_mia
from pretraining.data.transforms3D import ToTensor3D, Normalize3D, elastic_wrapper, elastic_deform_robin, Load
from pretraining.data.transforms3D import get_transform, Zoom
from pretraining.data.transforms3D import Compose as My_Compose
from pretraining.data.transforms3D import Identity as My_Identity
import argparse
from pretraining.segutils import DiceCELoss, MyBCELoss, update_teacher, set_deterministic, seed_worker, copy_model
from pretraining.all.all_utils import validation_seg, supervised_step_seg
from pretraining.all.all_utils import validation_cls, supervised_step_cls
from pretraining.all.all_utils import supervised_step_mul
from pretraining.all.all_utils import saveSequential, loadSequential
import shutil
from pretraining.classification_stuff.loss import BinaryCrossEntropySevOnly, CrossEntropySevOnly
from tqdm import tqdm

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
    parser.add_argument('--mode',
                        default=1,
                        help='Choose which kind of semi-supervised training is applied',
                        type=int)
    parser.add_argument('--load',
                        help='Load first supervised training',
                        dest='load',
                        action='store_true')
    parser.add_argument('--no-load',
                        dest='load',
                        action='store_false')
    parser.set_defaults(load=False)
    args = parser.parse_args()
    args.imsize = '256'
    #TODO set model in config instead of in args
    args.model = 'multinext'
    return args


def train_step_seg(train_loader, val_loader, base_model, loss_fn, config, identifier, iteration, dataset='tcia'):
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

    max_epochs = config.MAX_EPOCHS if iteration == 0 else config.MAX_EPOCHS // 2

    iteration, epoch = 0, 0
    num_at_once = modelconfig.num_at_once if hasattr(modelconfig, 'num_at_once') else 1
    dice_score_ema = validation_seg(model, teacher, val_loader, writer=writer, epoch=epoch, device=config.DEVICE,
               num_at_once=num_at_once, loss_fn=loss_fn)

    min_num_iterations = 80 * 0.85 * 400 / 15 / 4
    epoch_difference = 500 if dataset == 'tcia' else 62

    # dice_score_ema = -1e6
    print('Epoch ', epoch)
    copy_model(model, saved_model)
    copy_model(teacher, saved_teacher)
    best = [dice_score_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
    for epoch in range(1, max_epochs):
        print('Epoch ', epoch)
        for input, label in tqdm(train_loader):
            optimizer.zero_grad()
            supervised_step_seg(input, label, model, loss_fn, writer, iteration, config, num_at_once=num_at_once)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            alpha = 0.99 #if iteration < min_num_iterations else 0.995
            update_teacher(model, teacher, alpha=alpha)
            iteration +=1

        dice_score_ema = validation_seg(model, teacher, val_loader, writer=writer, epoch=epoch, device=config.DEVICE,
                   num_at_once=num_at_once, loss_fn=loss_fn)
        if dice_score_ema > best[0]:
            copy_model(model, saved_model)
            copy_model(teacher, saved_teacher)
            best = [dice_score_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
        elif epoch - best[1] > epoch_difference and iteration > min_num_iterations:
            break

    return best



def train_step_cls(train_loader, val_loader, base_model, loss_fn, config, identifier, iteration):
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
    if iteration == 0:
        min_num_epochs = 16 # if config.MODEL_NAME == 'upernext' else 100
    else:
        min_num_epochs = 8

    iteration, epoch = 0, 0
    num_at_once = modelconfig.num_at_once if hasattr(modelconfig, 'num_at_once') else 1
    auc_ema, __ = validation_cls(val_loader, teacher, config, loss_fn, writer=writer, epoch=epoch)
    # dice_score_ema = -1e6
    print('Epoch ', epoch)
    copy_model(model, saved_model)
    copy_model(teacher, saved_teacher)
    best = [auc_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
    for epoch in range(1, config.MAX_EPOCHS):
        print('Epoch ', epoch)
        for input, __, label in tqdm(train_loader):
            optimizer.zero_grad()
            supervised_step_cls(input, label, model, loss_fn, writer, iteration, config, num_at_once=num_at_once)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            alpha = 0.99 #if epoch < min_num_epochs else 0.995
            update_teacher(model, teacher, alpha=alpha)
            iteration +=1

        auc_ema, __ = validation_cls(val_loader, teacher, config, loss_fn, writer=writer, epoch=epoch)
        if auc_ema > best[0]:
            copy_model(model, saved_model)
            copy_model(teacher, saved_teacher)
            best = [auc_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
        elif epoch - best[1] > 62 and epoch > min_num_epochs:
            break

    return best


# train one dataset with multiple ground truths
def train_step_mul1(train_loader, val_loader_cls, val_loader_seg, base_model, loss_fn_cls, loss_fn_seg, config, identifier, iteration):
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
    auc_ema, __ = validation_cls(val_loader_cls, teacher, config, loss_fn_cls, writer=writer, epoch=epoch)
    dice_score_ema = validation_seg(model, teacher, val_loader_seg, writer=writer, epoch=epoch, device=config.DEVICE,
                                    num_at_once=num_at_once, loss_fn=loss_fn_seg)

    min_num_iterations = 80 * 0.85 * 400 / 15 / 4

    # dice_score_ema = -1e6
    print('Epoch ', epoch)
    copy_model(model, saved_model)
    copy_model(teacher, saved_teacher)
    best = [auc_ema, dice_score_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
    for epoch in range(1, config.MAX_EPOCHS):
        print('Epoch ', epoch)
        for input, label_seg, label_cls in tqdm(train_loader):
            optimizer.zero_grad()
            supervised_step_mul(input, label_cls, label_seg, model, loss_fn_seg, loss_fn_cls, writer, iteration, config, num_at_once=num_at_once)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            alpha = 0.99 #if iteration < min_num_iterations else 0.995
            update_teacher(model, teacher, alpha=alpha)
            iteration += 1

        auc_ema, __ = validation_cls(val_loader_cls, teacher, config, loss_fn_cls, writer=writer, epoch=epoch)
        dice_score_ema = validation_seg(model, teacher, val_loader_seg, writer=writer, epoch=epoch, device=config.DEVICE,
                                        num_at_once=num_at_once, loss_fn=loss_fn_seg)
        if auc_ema > best[0]:
            copy_model(model, saved_model)
            copy_model(teacher, saved_teacher)
            best = [auc_ema, dice_score_ema, epoch, saved_model.cpu(), saved_teacher.cpu()]
        elif epoch - best[2] > 62 and iteration > min_num_iterations:
            break
    return best


# train one dataset with segmentation labels and one dataset with multiple ground truths
def train_step_mul2(train_loader_seg, train_loader_mul, val_loader_cls, val_loader_seg, base_model, loss_fn_cls, loss_fn_seg, config, identifier, iteration):
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

    iteration, epoch_seg, epoch_mul = 0, 0, 0
    num_at_once = modelconfig.num_at_once if hasattr(modelconfig, 'num_at_once') else 1
    auc_ema, __ = validation_cls(val_loader_cls, teacher, config, loss_fn_cls, writer=writer, epoch=epoch_mul)
    dice_score_ema = validation_seg(model, teacher, val_loader_seg, writer=writer, epoch=epoch_seg, device=config.DEVICE,
                                    num_at_once=num_at_once, loss_fn=loss_fn_seg)

    min_num_iterations = 80 * 0.85 * 400 / 15 / 4

    # dice_score_ema = -1e6
    print('Epoch ', epoch_seg)
    copy_model(model, saved_model)
    copy_model(teacher, saved_teacher)
    best = [auc_ema, dice_score_ema, epoch_seg, saved_model.cpu(), saved_teacher.cpu()]

    train_loader_seg_iter = iter(train_loader_seg)
    train_loader_mul_iter = iter(train_loader_mul)

    while epoch_seg < config.MAX_EPOCHS:
        print('Epoch ', epoch_seg)
        input_mul, label_seg_mul, label_cls_mul = next(train_loader_mul_iter, ('end', None, None))
        if input_mul == 'end':
            auc_ema, __ = validation_cls(val_loader_cls, teacher, config, loss_fn_cls, writer=writer, epoch=epoch_mul)
            train_loader_mul_iter = iter(train_loader_mul)
            input_mul, label_seg_mul, label_cls_mul = next(train_loader_mul_iter, ('end', None, None))
            epoch_mul += 1
        input_seg, label_seg = next(train_loader_seg_iter, ('end', None))
        if input_seg == 'end':
            dice_score_ema = validation_seg(model, teacher, val_loader_seg, writer=writer, epoch=epoch_seg,
                                            device=config.DEVICE,
                                            num_at_once=num_at_once, loss_fn=loss_fn_seg)
            train_loader_seg_iter = iter(train_loader_seg)
            input_seg, label_seg = next(train_loader_seg_iter, ('end', None))
            epoch_seg += 1
            if dice_score_ema > best[1]:
                copy_model(model, saved_model)
                copy_model(teacher, saved_teacher)
                best = [auc_ema, dice_score_ema, epoch_seg, saved_model.cpu(), saved_teacher.cpu()]
            elif epoch_seg - best[2] > 500 and iteration > min_num_iterations:
                break

        optimizer.zero_grad()
        supervised_step_mul(input_mul, label_cls_mul, label_seg, model, loss_fn_seg, loss_fn_cls, writer, iteration, config, num_at_once=num_at_once)
        supervised_step_seg(input_seg, label_seg, model, loss_fn_seg, writer, iteration, config, num_at_once=num_at_once)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        alpha = 0.99 #if iteration < min_num_iterations else 0.995
        update_teacher(model, teacher, alpha=alpha)
        iteration += 1

    return best




def create_labels_seg(model, data_loader, config, identifier):
    savepath = os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'unsupervised_labels_seg')
    savepath256 = os.path.join(savepath, 'resized256_compressed')
    savepath224 = os.path.join(savepath, 'resized224_compressed')
    shape256 = (256, 256, 128) if config.IMAGE_SIZE == '256' else (128, 128, 64)
    os.makedirs(savepath256, exist_ok=True)
    os.makedirs(savepath224, exist_ok=True)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    model.to(config.DEVICE)
    model.eval()
    with torch.no_grad():
        for sample in data_loader:
            input, data_name = sample[0], sample[1]
            for i in range(0, input.shape[0], config.modelconfig.num_at_once):
                step = min(i + config.modelconfig.num_at_once, input.shape[0])
                inp, dat_nam = input[i:step], data_name[i:step]
                inp = inp.to(config.DEVICE)
                __1, __2, __3, predict224 = model(inp, mode='segmentation')
                predict256 = nn.functional.interpolate(predict224, shape256, mode='trilinear')
                predict224, predict256 = predict224.cpu().numpy(), predict256.cpu().numpy()
                for j in range(inp.shape[0]):
                    pred224, pred256 = predict224[j, 0], predict256[j, 0]
                    pred224, pred256 = sigmoid(pred224) - 0.5, sigmoid(pred256) - 0.5
                    pred224, pred256 = np.heaviside(pred224, pred224), np.heaviside(pred256, pred256)
                    data_path224, data_path256 = os.path.join(savepath224, dat_nam[j]), os.path.join(savepath256, dat_nam[j])
                    if os.path.exists(data_path224):
                        os.remove(data_path224)
                    np.savez_compressed(data_path224, pred224)
                    if os.path.exists(data_path256):
                        os.remove(data_path256)
                    np.savez_compressed(data_path256, pred256)
    model.train()
    return savepath256



#sup. tcia -> pseudo stoic -> sup. tcia
def run_seg1(config, args):
    model = get_model(config)
    modelconfig = config.modelconfig
    if config.MODEL_NAME == 'multinext':
        model_name = 'convnextransformer' if modelconfig.use_transformer else 'convnext'
    else:
        model_name = 'unet'
    if hasattr(modelconfig, 'size') and modelconfig.size == 'micro':
        model_name += 'Micro'
    identifier = ''.join((model_name, config.LOSS))
    supaug = '_supaug' if config.SUPERVISED_AUGMENTATIONS else ''
    identifier = ''.join((args.nick, '_segmentation1_', 'reductionFactor', str(config.REDUCTION_FACTOR), '_',
                          identifier, '_', config.UNSUPERVISED_DATA, '_resized',
                          str(config.IMAGE_SIZE), supaug, '_', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    identifier = os.path.join(config.LOGS_PATH, identifier)
    if config.DO_NORMALIZATION: print('data is normalized')
    normalize_tcia = Normalize3D(mean=0.220458088175389,
                                 std=0.30216501129178236) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_supervised_val = My_Compose(
        [Load(zoomsize, img_path=config.DATA_PATH, seg_path=config.DATA_PATH), ToTensor3D(), normalize_tcia])
    # transform_supervised_val = My_Compose([Load(256, path=config.DATA_PATH), Zoom((224, 224, 112)), ToTensor3D(), normalize_tcia])
    transform_supervised_train = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_tcia = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    train_dataset = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_train, train=True, reduction_factor=config.REDUCTION_FACTOR)
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
    elif config.UNSUPERVISED_DATA == 'mia':
        SequentialDataset = SequentialDataset_mia
    elif config.UNSUPERVISED_DATA == 'mosmed':
        SequentialDataset = SequentialDataset_mosmed
    else:
        SequentialDataset = SequentialDataset_tcia


    print('----------------------')
    print('Sequential step', 0)
    print('Training with real labels')
    ident = os.path.join(identifier, 'sup'+str(0))
    if args.load == False:
        best = train_step_seg(train_loader, my_val_loader, model, loss_fn, config, ident, iteration=0, dataset='tcia')
        teacher = best[3]
        saveSequential(teacher, 0, config, identifier)
    else:
        teacher = get_model(config)
        #copy_model(model, teacher)
        teacher, epoch = loadSequential(teacher)
        best = [0, epoch, teacher, teacher]
    for i in range(1, 10):
        print('----------------------')
        print('Sequential step', i)
        print('creating new labels')
        unsup_dataset = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia)
        unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
        labelpath = create_labels_seg(teacher, unsup_loader, config, identifier)

        if config.SUPERVISED_AUGMENTATIONS:
            multi_transform = My_Compose([get_transform(config.IMAGE_SIZE, config.DATA_PATH, labelpath), ToTensor3D(), normalize_tcia])
        else:
            multi_transform = My_Compose([Load(zoomsize, img_path=config.DATA_PATH, seg_path=labelpath), ToTensor3D(), normalize_tcia])
        unsup_dataset = SequentialDataset(path=config.DATA_PATH, label_path=labelpath, transform=multi_transform)
        unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)

        print('Training with pseudo labels')
        ident = os.path.join(identifier, 'unsup' + str(i))
        new_best = train_step_seg(unsup_loader, my_val_loader, model, loss_fn, config, ident, iteration=i, dataset=config.UNSUPERVISED_DATA)
        teacher = new_best[3]
        model_copy = get_model(config)
        copy_model(model, model_copy)

        print('Training with real labels')
        ident = os.path.join(identifier, 'sup' + str(i))
        old_lr = config.modelconfig.learning_rate
        config.modelconfig.learning_rate = old_lr / 5.0
        new_best = train_step_seg(train_loader, my_val_loader, teacher, loss_fn, config, ident, iteration=i, dataset='tcia')
        config.modelconfig.learning_rate = old_lr
        teacher = new_best[3]
        model_copy = get_model(config)
        copy_model(model, model_copy)


        if best[0] > new_best[0]:
            print('No improvement in step ', i)
            break
        best = new_best.copy()
        saveSequential(teacher, i, config, identifier)

    #shutil.rmtree(path=os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'checkpoints'), ignore_errors=True)
    saveSequential(best[3], None, config, identifier)


#sup. tcia -> sup. stoic -> sup. tcia
def run_seg2(config, args):
    model = get_model(config)
    modelconfig = config.modelconfig
    if config.MODEL_NAME == 'multinext':
        model_name = 'convnextransformer' if modelconfig.use_transformer else 'convnext'
    else:
        model_name = 'unet'
    if hasattr(modelconfig, 'size') and modelconfig.size == 'micro':
        model_name += 'Micro'
    identifier = ''.join((model_name, config.LOSS))
    supaug = '_supaug' if config.SUPERVISED_AUGMENTATIONS else ''
    identifier = ''.join((args.nick, '_segmentation2_', 'reductionFactor', str(config.REDUCTION_FACTOR), '_',
                          identifier, '_', config.UNSUPERVISED_DATA, '_resized',
                          str(config.IMAGE_SIZE), supaug, '_', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    identifier = os.path.join(config.LOGS_PATH, identifier)
    if config.DO_NORMALIZATION: print('data is normalized')
    normalize_tcia = Normalize3D(mean=0.220458088175389,
                                 std=0.30216501129178236) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_supervised_val = My_Compose(
        [Load(zoomsize, img_path=config.DATA_PATH, seg_path=config.DATA_PATH), ToTensor3D(), normalize_tcia])
    # transform_supervised_val = My_Compose([Load(256, path=config.DATA_PATH), Zoom((224, 224, 112)), ToTensor3D(), normalize_tcia])
    transform_supervised_train = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_tcia = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    train_dataset = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_train, train=True, reduction_factor=config.REDUCTION_FACTOR)
    my_val_dataset = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_val, train=False)
    #TODO create val-dataset with variable label-path
    val_dataset_unlabeled = ValidationDataset(path=config.DATA_PATH, transform=transform_supervised_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.Batch_SIZE, num_workers=config.WORKERS,
                                               shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    my_val_loader_seg = torch.utils.data.DataLoader(my_val_dataset, batch_size=config.Batch_SIZE,
                                                num_workers=config.WORKERS,
                                                shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
    if config.LOSS == 'dicece':
        loss_fn_seg = DiceCELoss()
    elif config.LOSS == 'balancedce':
        loss_fn_seg = MyBCELoss(device=config.DEVICE)
    else:
        loss_fn_seg = torch.nn.BCEWithLogitsLoss()
    loss_fn_cls = CrossEntropySevOnly(pos_weight=None)

    assert config.UNSUPERVISED_DATA == 'stoic'
    if config.UNSUPERVISED_DATA == 'stoic':
        SequentialDataset = SequentialDataset_stoic
    elif config.UNSUPERVISED_DATA == 'mosmed':
        SequentialDataset = SequentialDataset_mosmed
    else:
        SequentialDataset = SequentialDataset_tcia

    val_dataset_cls = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia,
                                        severity=True, train=False)
    val_loader_cls = torch.utils.data.DataLoader(val_dataset_cls, batch_size=config.Batch_SIZE,
                                                 num_workers=config.WORKERS,
                                                 shuffle=False, pin_memory=True, worker_init_fn=seed_worker)


    print('----------------------')
    print('Sequential step', 0)
    print('Training with real labels')
    ident = os.path.join(identifier, 'sup'+str(0))
    if args.load == False:
        best = train_step_seg(train_loader, my_val_loader_seg, model, loss_fn_seg, config, ident, iteration=0)
        teacher = best[3]
        saveSequential(teacher, 0, config, identifier)
    else:
        teacher = get_model(config)
        #copy_model(model, teacher)
        teacher, epoch = loadSequential(teacher)
        best = [0, epoch, teacher, teacher]
    for i in range(1, 2):
        print('----------------------')
        print('Sequential step', i)
        print('creating new labels')
        unsup_dataset = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia, severity=True, train=True)
        unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
        # labelpath = create_labels_seg(teacher, unsup_loader, config, identifier)
        # unsup_dataset = SequentialDataset(path=config.DATA_PATH, label_path=labelpath, transform=transform_unsupervised_tcia, severity=True, train=True)
        # unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size=config.Batch_SIZE,
        #                                            num_workers=config.WORKERS,
        #                                            shuffle=False, pin_memory=True, worker_init_fn=seed_worker)

        print('Training classification')
        ident = os.path.join(identifier, 'unsup' + str(i))
        new_best = train_step_cls(unsup_loader, val_loader_cls, model, loss_fn_cls, config, ident, iteration=i)
        teacher = new_best[3]
        model_copy = get_model(config)
        copy_model(model, model_copy)

        print('Training segmentation')
        ident = os.path.join(identifier, 'sup' + str(i))
        old_lr = config.modelconfig.learning_rate
        config.modelconfig.learning_rate = old_lr / 5.0
        new_best = train_step_seg(train_loader, my_val_loader_seg, teacher, loss_fn_seg, config, ident, iteration=i)
        config.modelconfig.learning_rate = old_lr
        teacher = new_best[3]
        model_copy = get_model(config)
        copy_model(model, model_copy)


        if best[0] > new_best[0]:
            print('No improvement in step ', i)
            break
        best = new_best.copy()
        saveSequential(teacher, i, config, identifier)

    #shutil.rmtree(path=os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'checkpoints'), ignore_errors=True)
    saveSequential(best[3], None, config, identifier)



#sup. tcia -> multitask stoic -> sup tcia
def run_seg3(config, args):
    model = get_model(config)
    modelconfig = config.modelconfig
    if config.MODEL_NAME == 'multinext':
        model_name = 'convnextransformer' if modelconfig.use_transformer else 'convnext'
    else:
        model_name = 'unet'
    if hasattr(modelconfig, 'size') and modelconfig.size == 'micro':
        model_name += 'Micro'
    identifier = ''.join((model_name, config.LOSS))
    supaug = '_supaug' if config.SUPERVISED_AUGMENTATIONS else ''
    identifier = ''.join((args.nick, '_segmentation3_', 'reductionFactor', str(config.REDUCTION_FACTOR), '_',
                          identifier, '_', config.UNSUPERVISED_DATA, '_resized',
                          str(config.IMAGE_SIZE), supaug, '_', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    identifier = os.path.join(config.LOGS_PATH, identifier)
    if config.DO_NORMALIZATION: print('data is normalized')
    normalize_tcia = Normalize3D(mean=0.220458088175389,
                                 std=0.30216501129178236) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_supervised_val = My_Compose(
        [Load(zoomsize, img_path=config.DATA_PATH, seg_path=config.DATA_PATH), ToTensor3D(), normalize_tcia])
    # transform_supervised_val = My_Compose([Load(256, path=config.DATA_PATH), Zoom((224, 224, 112)), ToTensor3D(), normalize_tcia])
    transform_supervised_train = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_tcia = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    train_dataset_tcia = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_train, train=True, reduction_factor=config.REDUCTION_FACTOR)
    my_val_dataset = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_val, train=False)
    #TODO create val-dataset with variable label-path
    val_dataset_unlabeled = ValidationDataset(path=config.DATA_PATH, transform=transform_supervised_val)
    train_loader_tcia = torch.utils.data.DataLoader(train_dataset_tcia, batch_size=config.Batch_SIZE, num_workers=config.WORKERS,
                                               shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    my_val_loader_seg = torch.utils.data.DataLoader(my_val_dataset, batch_size=config.Batch_SIZE,
                                                num_workers=config.WORKERS,
                                                shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
    if config.LOSS == 'dicece':
        loss_fn_seg = DiceCELoss()
    elif config.LOSS == 'balancedce':
        loss_fn_seg = MyBCELoss(device=config.DEVICE)
    else:
        loss_fn_seg = torch.nn.BCEWithLogitsLoss()
    loss_fn_cls = CrossEntropySevOnly(pos_weight=None)

    assert config.UNSUPERVISED_DATA in ['stoic', 'mosmed']
    if config.UNSUPERVISED_DATA == 'stoic':
        SequentialDataset = SequentialDataset_stoic
    elif config.UNSUPERVISED_DATA == 'mosmed':
        SequentialDataset = SequentialDataset_mosmed
    else:
        SequentialDataset = SequentialDataset_tcia

    val_dataset_cls = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia, severity=True, train=False)
    val_loader_cls = torch.utils.data.DataLoader(val_dataset_cls, batch_size=config.Batch_SIZE,
                                                num_workers=config.WORKERS,
                                                shuffle=False, pin_memory=True, worker_init_fn=seed_worker)


    print('----------------------')
    print('Sequential step', 0)
    print('Training with real labels')
    ident = os.path.join(identifier, 'sup'+str(0))
    if args.load == False:
        best = train_step_seg(train_loader_tcia, my_val_loader_seg, model, loss_fn_seg, config, ident, iteration=0, dataset='tcia')
        teacher = best[3]
        saveSequential(teacher, 0, config, identifier)
    else:
        teacher = get_model(config)
        #copy_model(model, teacher)
        teacher, epoch = loadSequential(teacher)
        best = [0, epoch, teacher, teacher]
    for i in range(1, 10):
        print('----------------------')
        print('Sequential step', i)
        print('creating new labels')
        multi_dataset = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia, severity=None, train=True)
        multi_loader = torch.utils.data.DataLoader(multi_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
        labelpath = create_labels_seg(teacher, multi_loader, config, identifier)

        if config.SUPERVISED_AUGMENTATIONS:
            multi_transform = My_Compose([get_transform(config.IMAGE_SIZE, config.DATA_PATH, labelpath), ToTensor3D(), normalize_tcia])
        else:
            multi_transform = My_Compose([Load(zoomsize, img_path=config.DATA_PATH, seg_path=labelpath), ToTensor3D(), normalize_tcia])
        multi_dataset = SequentialDataset(path=config.DATA_PATH, label_path=labelpath, transform=multi_transform, severity=True, train=True)
        multi_loader = torch.utils.data.DataLoader(multi_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)

        print('Training semi-supervised segmentation and supervised classification')
        ident = os.path.join(identifier, 'unsup' + str(i))
        new_best = train_step_mul1(multi_loader, val_loader_cls, my_val_loader_seg, model, loss_fn_cls, loss_fn_seg, config, ident, iteration=i)
        teacher = new_best[4]
        model_copy = get_model(config)
        copy_model(model, model_copy)

        print('Training segmentation')
        ident = os.path.join(identifier, 'sup' + str(i))
        old_lr = config.modelconfig.learning_rate
        config.modelconfig.learning_rate = old_lr / 5.0
        new_best = train_step_seg(train_loader_tcia, my_val_loader_seg, teacher, loss_fn_seg, config, ident, iteration=i)
        config.modelconfig.learning_rate = old_lr
        teacher = new_best[3]
        model_copy = get_model(config)
        copy_model(model, model_copy)


        if best[0] > new_best[0]:
            print('No improvement in step ', i)
            break
        best = new_best.copy()
        saveSequential(teacher, i, config, identifier)

    #shutil.rmtree(path=os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'checkpoints'), ignore_errors=True)
    saveSequential(best[3], None, config, identifier)



#sup. tcia -> multitask stoic + sup. tcia -> sup. tcia
def run_mul1(config, args):
    model = get_model(config)
    modelconfig = config.modelconfig
    if config.MODEL_NAME == 'multinext':
        model_name = 'convnextransformer' if modelconfig.use_transformer else 'convnext'
    else:
        model_name = 'unet'
    if hasattr(modelconfig, 'size') and modelconfig.size == 'micro':
        model_name += 'Micro'
    identifier = ''.join((model_name, config.LOSS))
    supaug = '_supaug' if config.SUPERVISED_AUGMENTATIONS else ''
    identifier = ''.join((args.nick, '_multitask1_', 'reductionFactor', str(config.REDUCTION_FACTOR), '_',
                          identifier, '_', config.UNSUPERVISED_DATA, '_resized',
                          str(config.IMAGE_SIZE), supaug, '_', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    identifier = os.path.join(config.LOGS_PATH, identifier)
    if config.DO_NORMALIZATION: print('data is normalized')
    normalize_tcia = Normalize3D(mean=0.220458088175389,
                                 std=0.30216501129178236) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_supervised_val = My_Compose([Load(zoomsize, img_path=config.DATA_PATH, seg_path=config.DATA_PATH), ToTensor3D(), normalize_tcia])
    #transform_supervised_val = My_Compose([Load(256, path=config.DATA_PATH), Zoom((224, 224, 112)), ToTensor3D(), normalize_tcia])
    transform_supervised_train = My_Compose([get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
                                             normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_tcia = My_Compose([get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
                                              normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    train_dataset_tcia = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_train, train=True,
                                           reduction_factor=config.REDUCTION_FACTOR)
    my_val_dataset = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_val, train=False)
    # TODO create val-dataset with variable label-path
    val_dataset_unlabeled = ValidationDataset(path=config.DATA_PATH, transform=transform_supervised_val)
    train_loader_tcia = torch.utils.data.DataLoader(train_dataset_tcia, batch_size=config.Batch_SIZE,
                                                    num_workers=config.WORKERS,
                                                    shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    my_val_loader_seg = torch.utils.data.DataLoader(my_val_dataset, batch_size=config.Batch_SIZE,
                                                    num_workers=config.WORKERS,
                                                    shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
    if config.LOSS == 'dicece':
        loss_fn_seg = DiceCELoss()
    elif config.LOSS == 'balancedce':
        loss_fn_seg = MyBCELoss(device=config.DEVICE)
    else:
        loss_fn_seg = torch.nn.BCEWithLogitsLoss()
    loss_fn_cls = CrossEntropySevOnly(pos_weight=None)

    assert config.UNSUPERVISED_DATA in ['stoic', 'mosmed']
    if config.UNSUPERVISED_DATA == 'stoic':
        SequentialDataset = SequentialDataset_stoic
    elif config.UNSUPERVISED_DATA == 'mosmed':
        SequentialDataset = SequentialDataset_mosmed
    else:
        SequentialDataset = SequentialDataset_tcia

    val_dataset_cls = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia,
                                        severity=True, train=False)
    val_loader_cls = torch.utils.data.DataLoader(val_dataset_cls, batch_size=config.Batch_SIZE,
                                                 num_workers=config.WORKERS,
                                                 shuffle=False, pin_memory=True, worker_init_fn=seed_worker)

    print('----------------------')
    print('Sequential step', 0)
    print('Training with real labels')
    ident = os.path.join(identifier, 'sup' + str(0))
    if args.load == False:
        best = train_step_seg(train_loader_tcia, my_val_loader_seg, model, loss_fn_seg, config, ident, iteration=0)
        teacher = best[3]
        saveSequential(teacher, 0, config, identifier)
    else:
        teacher = get_model(config)
        #copy_model(model, teacher)
        teacher, epoch = loadSequential(teacher)
        best = [0, epoch, teacher, teacher]
    for i in range(1, 10):
        print('----------------------')
        print('Sequential step', i)
        print('creating new labels')
        multi_dataset = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia,
                                          severity=None, train=True)
        multi_loader = torch.utils.data.DataLoader(multi_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
        labelpath = create_labels_seg(teacher, multi_loader, config, identifier)

        if config.SUPERVISED_AUGMENTATIONS:
            multi_transform = My_Compose([get_transform(config.IMAGE_SIZE, config.DATA_PATH, labelpath), ToTensor3D(), normalize_tcia])
        else:
            multi_transform = My_Compose([Load(zoomsize, img_path=config.DATA_PATH, seg_path=labelpath), ToTensor3D(), normalize_tcia])
        multi_dataset = SequentialDataset(path=config.DATA_PATH, label_path=labelpath,
                                          transform=multi_transform, severity=True, train=True)
        multi_loader = torch.utils.data.DataLoader(multi_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)

        print('Multitask-training segmentation and segmentation+classification')
        ident = os.path.join(identifier, 'unsup' + str(i))
        # new_best = train_step_mul1(multi_loader, val_loader_cls, my_val_loader_seg, model, loss_fn_cls, loss_fn_seg,
        #                            config, ident, iteration=i)
        new_best = train_step_mul2(train_loader_tcia, multi_loader, val_loader_cls, my_val_loader_seg, model,
                                   loss_fn_cls, loss_fn_seg, config, ident, iteration=i)
        teacher = new_best[4]
        model_copy = get_model(config)
        copy_model(model, model_copy)

        print('Training segmentation')
        ident = os.path.join(identifier, 'sup' + str(i))
        old_lr = config.modelconfig.learning_rate
        config.modelconfig.learning_rate = old_lr / 5.0
        new_best = train_step_seg(train_loader_tcia, my_val_loader_seg, teacher, loss_fn_seg, config, ident,
                                  iteration=i)
        config.modelconfig.learning_rate = old_lr
        teacher = new_best[3]
        model_copy = get_model(config)
        copy_model(model, model_copy)

        if best[0] > new_best[0]:
            print('No improvement in step ', i)
            break
        best = new_best.copy()
        saveSequential(teacher, i, config, identifier)

    # shutil.rmtree(path=os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'checkpoints'), ignore_errors=True)
    saveSequential(best[3], None, config, identifier)



def main_seg():
    print('PID:', os.getpid())
    set_deterministic()
    args = parse_args()
    config = AllConfig(args)
    config.modelconfig = get_modelconfig(config)

    config.MAX_EPOCHS = 5000
    assert args.mode in [1, 2, 3, 4]
    print('Mode: ', args.mode)
    if args.mode == 1:
        run_seg1(config, args)
    elif args.mode == 2:
        run_seg2(config, args)
    elif args.mode == 3:
        run_seg3(config, args)
    else:
        run_mul1(config, args)


if __name__ == '__main__':
    main_seg()

