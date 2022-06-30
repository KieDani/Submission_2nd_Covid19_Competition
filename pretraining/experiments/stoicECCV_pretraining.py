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
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os
from pretraining.all.train_all import train_step_seg, parse_args, train_step_mul1
from paths import pretraining_LOGS_PATH, pretraining_DATA_PATH


#sup. tcia -> pseudo mia -> sup. tcia
def run_segmia(config, args):
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
    normalize_mia = Normalize3D(mean=0.3971,
                                 std=0.3207) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_supervised_val = My_Compose(
        [Load(zoomsize, img_path=config.DATA_PATH, seg_path=config.DATA_PATH), ToTensor3D(), normalize_mia])
    # transform_supervised_val = My_Compose([Load(256, path=config.DATA_PATH), Zoom((224, 224, 112)), ToTensor3D(), normalize_tcia])
    transform_supervised_train = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_mia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_tcia = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_mia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
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

    assert config.UNSUPERVISED_DATA == 'mia'
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
        teacher, epoch = loadSequential(teacher, savepath=os.path.join(pretraining_LOGS_PATH, '/saved_sup0/saved_sup0.pth'))
        best = [0, epoch, teacher, teacher]
    for i in range(1, 10):
        print('----------------------')
        print('Sequential step', i)
        print('creating new labels')
        unsup_dataset = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia)
        unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
        labelpath = create_labels_mia(teacher, unsup_loader, config, identifier)

        if config.SUPERVISED_AUGMENTATIONS:
            multi_transform = My_Compose([get_transform(config.IMAGE_SIZE, config.DATA_PATH, labelpath), ToTensor3D(), normalize_mia])
        else:
            multi_transform = My_Compose([Load(zoomsize, img_path=config.DATA_PATH, seg_path=labelpath), ToTensor3D(), normalize_mia])
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

        saveSequential(teacher, i-0.5, config, identifier)

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


def create_labels_mia(model, data_loader, config, identifier):
    savepath = os.path.join(config.LOGS_PATH, os.path.basename(identifier), 'unsupervised_labels_seg')
    savepath256 = os.path.join(savepath, 'resized256')
    savepath224 = os.path.join(savepath, 'resized224')
    shape256 = (256, 256, 256) if config.IMAGE_SIZE == '256' else (128, 128, 128)
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
                    np.save(data_path224, pred224)
                    if os.path.exists(data_path256):
                        os.remove(data_path256)
                    np.save(data_path256, pred256)
    model.train()
    return savepath256



#sup. tcia -> multitask stoic/mia -> sup tcia
def run_mulmia(config, args):
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
    normalize_mia = Normalize3D(mean=0.3971,
                                 std=0.3207) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_supervised_val = My_Compose(
        [Load(zoomsize, img_path=config.DATA_PATH, seg_path=config.DATA_PATH), ToTensor3D(), normalize_mia])
    # transform_supervised_val = My_Compose([Load(256, path=config.DATA_PATH), Zoom((224, 224, 112)), ToTensor3D(), normalize_tcia])
    transform_supervised_train = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_mia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_tcia = My_Compose(
        [get_transform(config.IMAGE_SIZE, config.DATA_PATH, config.DATA_PATH), ToTensor3D(),
         normalize_mia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
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

    assert config.UNSUPERVISED_DATA in ['stoic', 'mia']
    if config.UNSUPERVISED_DATA == 'stoic':
        SequentialDataset = SequentialDataset_stoic
    elif config.UNSUPERVISED_DATA == 'mosmed':
        SequentialDataset = SequentialDataset_mosmed
    elif config.UNSUPERVISED_DATA == 'mia':
        SequentialDataset = SequentialDataset_mia
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
        teacher, epoch = loadSequential(teacher, savepath=os.path.join(pretraining_LOGS_PATH, '/saved_sup0/saved_sup0.pth'))
        best = [0, epoch, teacher, teacher]
    for i in range(1, 10):
        print('----------------------')
        print('Sequential step', i)
        print('creating new labels')
        multi_dataset = SequentialDataset(path=config.DATA_PATH, label_path=None, transform=transform_unsupervised_tcia, severity=None, train=None, reduction_factor=config.REDUCTION_FACTOR)
        multi_loader = torch.utils.data.DataLoader(multi_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
        labelpath = create_labels_mia(teacher, multi_loader, config, identifier)

        if config.SUPERVISED_AUGMENTATIONS:
            multi_transform = My_Compose([get_transform(config.IMAGE_SIZE, config.DATA_PATH, labelpath), ToTensor3D(), normalize_mia])
        else:
            multi_transform = My_Compose([Load(zoomsize, img_path=config.DATA_PATH, seg_path=labelpath), ToTensor3D(), normalize_mia])
        multi_dataset = SequentialDataset(path=config.DATA_PATH, label_path=labelpath, transform=multi_transform, severity=True, train=True, reduction_factor=config.REDUCTION_FACTOR)
        multi_loader = torch.utils.data.DataLoader(multi_dataset, batch_size=config.Batch_SIZE,
                                                   num_workers=config.WORKERS,
                                                   shuffle=False, pin_memory=True, worker_init_fn=seed_worker)

        print('Training semi-supervised segmentation and supervised classification')
        ident = os.path.join(identifier, 'unsup' + str(i))
        new_best = train_step_mul1(multi_loader, val_loader_cls, my_val_loader_seg, model, loss_fn_cls, loss_fn_seg, config, ident, iteration=i)
        teacher = new_best[4]
        model_copy = get_model(config)
        copy_model(model, model_copy)

        saveSequential(teacher, i - 0.5, config, identifier)

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



def main_stoicECCV():
    print('PID:', os.getpid())
    set_deterministic()
    args = parse_args()
    config = AllConfig(args)
    config.REDUCTION_FACTOR = 1
    config.modelconfig = get_modelconfig(config)

    #segmia (and segmentation in supervised step)
    # config.MAX_EPOCHS = 120
    # config.UNSUPERVISED_DATA = 'mia'
    # config.SUPERVISED_AUGMENTATIONS = True
    # config.DATA_PATH = pretraining_DATA_PATH
    # print('Mode: ', args.mode)
    # assert args.mode == 1
    # run_segmia(config, args)

    #multimia
    # config.MAX_EPOCHS = 120
    # config.UNSUPERVISED_DATA = 'mia'
    # config.SUPERVISED_AUGMENTATIONS = True
    # config.DATA_PATH = pretraining_DATA_PATH
    # print('Mode: ', args.mode)
    # assert args.mode == 3
    # run_mulmia(config, args)

    #multitask
    # config.MAX_EPOCHS = 120
    # config.UNSUPERVISED_DATA = 'stoic'
    # config.SUPERVISED_AUGMENTATIONS = True
    # config.DATA_PATH = pretraining_DATA_PATH
    # print('Mode: ', args.mode)
    # assert args.mode == 3
    # run_mulmia(config, args)


def main_stoicECCV_fullInit():
    print('PID:', os.getpid())
    set_deterministic()
    args = parse_args()
    config = AllConfig(args)
    config.REDUCTION_FACTOR = 1
    config.modelconfig = get_modelconfig(config)
    config.modelconfig.init_mode = 'full'

    #segmia (and segmentation in supervised step)
    config.MAX_EPOCHS = 120
    config.UNSUPERVISED_DATA = 'mia'
    config.SUPERVISED_AUGMENTATIONS = True
    config.DATA_PATH = pretraining_DATA_PATH
    print('Mode: ', args.mode)
    assert args.mode == 1
    run_segmia(config, args)

    #multimia
    # config.MAX_EPOCHS = 120
    # config.UNSUPERVISED_DATA = 'mia'
    # config.SUPERVISED_AUGMENTATIONS = True
    # config.DATA_PATH = pretraining_DATA_PATH
    # print('Mode: ', args.mode)
    # assert args.mode == 3
    # run_mulmia(config, args)

    #multitask
    # config.MAX_EPOCHS = 120
    # config.UNSUPERVISED_DATA = 'stoic'
    # config.SUPERVISED_AUGMENTATIONS = True
    # config.DATA_PATH = pretraining_DATA_PATH
    # print('Mode: ', args.mode)
    # assert args.mode == 3
    # run_mulmia(config, args)


if __name__ == '__main__':
    #main_stoicECCV()
    main_stoicECCV_fullInit()

