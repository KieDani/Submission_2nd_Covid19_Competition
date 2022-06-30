import numpy as np
import torch
import torch.nn as nn
#import monai
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pretraining.config.config import BaseConfig
from pretraining.config.modelconfig import get_model, get_modelconfig
from pretraining.data.datasets import SupervisedDataset, UnsupervisedDataset_stoic, ValidationDataset, UnsupervisedDataset_tcia, MosMedDataset_semisupervised
from pretraining.data.transforms3D import ToTensor3D, Normalize3D, elastic_wrapper, elastic_deform_robin
from pretraining.data.transforms3D import get_transform, Zoom
from pretraining.data.transforms3D import Compose as My_Compose
from pretraining.data.transforms3D import Identity as My_Identity
import nibabel as nib
import scipy.ndimage as ndimage
#from torchvision.transforms import Compose
import random
#import torch.multiprocessing as mp
#from multiprocessing import Pool as ClassicPool
import time
import argparse
from pretraining.segutils import DiceCELoss, diceloss, dice_score_fn, MyBCELoss, saveModel, load_model, IoU, update_teacher, set_deterministic, seed_worker


def parse_args():
    parser = argparse.ArgumentParser(description='Train STOIC network')
    # general
    parser.add_argument('--gpu',
                        default="6",
                        help='gpu id',
                        type=str)
    parser.add_argument('--model',
                        help='model used for training, possible are: unet and upernext',
                        default="unet",
                        type=str,
                        required=True)
    parser.add_argument('--cv',
                        help='Use cross-validation',
                        dest='cv',
                        action='store_true')
    parser.add_argument('--no-cv',
                        dest='cv',
                        action='store_false')
    parser.set_defaults(cv=False)
    parser.add_argument('--nick',
                        default='',
                        help='Prepend a nickname to the output directory',
                        type=str)
    parser.add_argument('--mode',
                        default='1',
                        help='1=onlySupervised, 2=MeanTeacher_withoutAugmentation, 3=MeanTeacher_elasticDeformation',
                        type=str)
    parser.add_argument('--imsize',
                        default='256',
                        help='image size: 256 or 128',
                        type=str)
    args = parser.parse_args()
    return args


def run(config, args):
    model = get_model(config)
    teacher = get_model(config)
    modelconfig = config.modelconfig
    model = model.to(config.DEVICE)
    teacher = teacher.to(config.DEVICE)
    with torch.no_grad():
        for name, param in teacher.named_parameters():
            model.state_dict()[name]
            param.data = model.state_dict()[name]

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
    if config.SUPERVISED:
        identifier = ''.join((model_name, '_onlySupervised_', config.LOSS))
    elif config.UNSUPERVISED_AUGMENTATION == 'elastic_deformation':
        identifier = ''.join((model_name, '_MeanTeacher_', 'elastic_deformation_', config.LOSS))
    else:
        identifier = ''.join((model_name, '_MeanTeacher_', 'withoutAugmentation_', config.LOSS))
    #identifier = 'unet_MeanTeacher_ElasticDeformation_diceceLoss_differentNormalization'
    #identifier = 'test'
    supaug = '_supaug' if config.SUPERVISED_AUGMENTATIONS else ''
    identifier = ''.join((args.nick, '_', identifier, '_', config.UNSUPERVISED_DATA, '_resized', config.IMAGE_SIZE, supaug, '_', datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    if config.DO_NORMALIZATION: print('data is normalized')
    normalize_tcia = Normalize3D(mean= 0.220458088175389, std=0.30216501129178236) if config.DO_NORMALIZATION else My_Identity()
    normalize_stoic = Normalize3D(mean=0.31186612587660995, std=0.315680396838336) if config.DO_NORMALIZATION else My_Identity()
    normalize_mosmed = Normalize3D(mean=0.31327220923026, std=0.31592914841013453) if config.DO_NORMALIZATION else My_Identity()
    zoomsize = 224 if config.IMAGE_SIZE == '256' else 112
    transform_supervised_val = My_Compose([Zoom([zoomsize, zoomsize, zoomsize]), ToTensor3D(), normalize_tcia])
    transform_supervised_train = My_Compose([get_transform(config.IMAGE_SIZE), ToTensor3D(), normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_stoic = My_Compose([get_transform(config.IMAGE_SIZE), ToTensor3D(), normalize_stoic]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_tcia = My_Compose([get_transform(config.IMAGE_SIZE), ToTensor3D(), normalize_tcia]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val
    transform_unsupervised_mosmed = My_Compose([get_transform(config.IMAGE_SIZE), ToTensor3D(), normalize_mosmed]) if config.SUPERVISED_AUGMENTATIONS else transform_supervised_val

    train_dataset = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_train, train=True)
    my_val_dataset = SupervisedDataset(path=config.DATA_PATH, transform=transform_supervised_val, train=False)
    val_dataset_unlabeled = ValidationDataset(path=config.DATA_PATH, transform=transform_supervised_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.Batch_SIZE, num_workers=config.WORKERS,
                                               shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    my_val_loader = torch.utils.data.DataLoader(my_val_dataset, batch_size=config.Batch_SIZE, num_workers=config.WORKERS,
                                               shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
    if config.SUPERVISED == False:
        if config.UNSUPERVISED_DATA == 'stoic':
            unsup_dataset = UnsupervisedDataset_stoic(path=config.DATA_PATH, transform=transform_unsupervised_stoic)
        elif config.UNSUPERVISED_DATA == 'mosmed':
            unsup_dataset = MosMedDataset_semisupervised(path=config.DATA_PATH, transform=transform_unsupervised_mosmed)
        else:
            unsup_dataset = UnsupervisedDataset_tcia(path=config.DATA_PATH, transform=transform_unsupervised_tcia)
        unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size = config.Batch_SIZE,
                                                   num_workers=config.WORKERS, shuffle=True, pin_memory=True,
                                                   worker_init_fn=seed_worker)
    train_loader_iter = iter(train_loader)
    if config.SUPERVISED == False: unsup_loader_iter = iter(unsup_loader)

    if config.LOSS == 'dicece':
        loss_fn = DiceCELoss()
    elif config.LOSS == 'balancedce':
        loss_fn = MyBCELoss(device=config.DEVICE)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=modelconfig.learning_rate)

    iteration, epoch = 0, 0
    #Here a model can be loaded
    # epoch, identifier = load_model(model, teacher, optimizer)
    identifier = os.path.join(config.LOGS_PATH, identifier)
    writer = SummaryWriter(identifier)

    print('Epoch ', epoch)
    num_at_once = modelconfig.num_at_once if hasattr(modelconfig, 'num_at_once') else 1
    validation(model, teacher, my_val_loader, writer=writer, epoch=epoch, device=config.DEVICE,
               num_at_once=num_at_once, loss_fn=loss_fn)
    while epoch < config.MAX_EPOCHS:
        # supervised step
        input, label = next(train_loader_iter, ('end', None))
        if input == 'end':
            train_loader_iter = iter(train_loader)
            input, label = next(train_loader_iter, ('end', None))
            epoch += 1
            print('Epoch ', epoch)
            if config.unsupervised_mode == 'per_epoch':
                with torch.no_grad():
                    for name, param in teacher.named_parameters():
                        model.state_dict()[name]
                        param.data = model.state_dict()[name]
            validation(model, teacher, my_val_loader, writer=writer, epoch=epoch, device=config.DEVICE,
                       num_at_once=num_at_once, loss_fn=loss_fn)
            if epoch % 5 == 0 and epoch != 0:
                saveModel(model, teacher, optimizer, epoch, config, identifier)
            if epoch % 15 == 0 and epoch != 0:
                pass
                #prediction(teacher, val_dataset_unlabeled, config, identifier, epoch)
        optimizer.zero_grad()
        supervised_step(input, label, model, loss_fn, writer, iteration, config, num_at_once=num_at_once)
        optimizer.step()
        if config.unsupervised_mode == 'per_iteration':
            teacher = update_teacher(model, teacher)

        #unsupervised step
        if epoch >= config.STARTEPOCH_UNSUPERVISED and config.SUPERVISED == False:
            start = time.time()
            config.NUM_UNSUP_STEPS = 1
            loss_sum = 0
            for j in range(config.NUM_UNSUP_STEPS):
                input = next(unsup_loader_iter, 'end')
                if input == 'end':
                    unsup_loader_iter = iter(unsup_loader)
                    input = next(unsup_loader_iter, 'end')
                optimizer.zero_grad()
                loss_sum += unsupervised_step(
                    input, model, teacher, loss_fn, writer, iteration, config, num_at_once
                ) / config.NUM_UNSUP_STEPS
                optimizer.step()
                if config.unsupervised_mode == 'per_iteration':
                    teacher = update_teacher(model, teacher)
            end = time.time()
            print('unsupervised time:', end-start)
            writer.add_scalar('train loss unsupervised', loss_sum, iteration)


        iteration += 1


def supervised_step(input, label, model, loss_fn, writer, iteration, config, num_at_once):
    input, label = input.to(config.DEVICE), label.to(config.DEVICE)
    loss_sum = 0
    for i in range(0, label.shape[0], num_at_once):
        step = min(i + num_at_once, label.shape[0])
        inp, lab = input[i:step], label[i:step]
        pred = model(inp)
        loss = loss_fn(pred, lab) / max((label.shape[0] // num_at_once), 1)
        loss_sum += loss.item()
        loss.backward()
    #print('train loss', loss_sum)
    writer.add_scalar('train loss', loss_sum, iteration)


def unsupervised_step(input, student, teacher, loss_fn, writer, iteration, config, num_at_once):
    input = input.to(config.DEVICE)
    loss_sum = 0

    label = torch.empty_like(input, requires_grad=False)
    for i in range(0, label.shape[0], num_at_once):
        step = min(i + num_at_once, label.shape[0])
        inp = input[i:step]
        with torch.no_grad():
            lab = teacher(inp)
            lab = torch.sigmoid(lab) - 0.5
            lab = torch.heaviside(lab, lab)
            label[i:step] = lab

    # label = label.cpu()
    # input = input.cpu()
    # x = [(input[i], label[i]) for i in range(input.shape[0])]
    # num_process = label.shape[0]
    # if config.UNSUPERVISED_AUGMENTATION == 'elastic_deformation':
    #     with mp.Pool(num_process) as p:
    #         result = p.map(elastic_wrapper, x)
    #     # result = [elastic_wrapper(elem) for elem in x]
    # else:
    #     result = x
    #
    # input = [elem[0].unsqueeze(0) for elem in result]
    # label = [elem[1].unsqueeze(0) for elem in result]
    # input, label = torch.cat(input, dim=0).to(config.DEVICE), torch.cat(label, dim=0).to(config.DEVICE)

    if config.UNSUPERVISED_AUGMENTATION == 'elastic_deformation':
        with torch.no_grad():
            input, label = elastic_deform_robin(input, label, sigma=35, alpha=random.random()*6+1, device=config.DEVICE)

    #TODO maybe speed it up -> all elems in one pass through the network
    for i in range(0, label.shape[0], num_at_once):
        step = min(i + num_at_once, label.shape[0])
        inp, lab = input[i:step], label[i:step]
        #TODO maybe use detach to be shure that there is no gradient for the elastic deformation

        pred = student(inp)
        loss = loss_fn(pred, lab) / max((label.shape[0] // num_at_once), 1)
        loss_sum += loss.item()
        loss.backward()
    return loss_sum


def validation(model, teacher, data_loader, writer, epoch, device, num_at_once, loss_fn):
    model.eval()
    teacher.eval()
    with torch.no_grad():
        loss, loss_ema = 0, 0
        up, down, up_ema, down_ema = 0, 0, 0, 0
        intersections, unions, intersections_ema, unions_ema = 0, 0, 0, 0
        for input, label in data_loader:
            input, label = input.to(device), label.to(device)
            for i in range(0, label.shape[0], num_at_once):
                step = min(i + num_at_once, label.shape[0])
                inp, lab = input[i:step], label[i:step]
                pred = model(inp)
                loss += loss_fn(pred, lab) / max((label.shape[0] // num_at_once), 1) / len(data_loader)
                up_tmp, down_tmp = dice_score_fn(pred, lab)
                up += up_tmp.item()
                down += down_tmp.item()
                inter, uni = IoU(pred, lab, hard_label=True)
                intersections += inter.item()
                unions += uni.item()

                pred_ema = teacher(inp)
                loss_ema += loss_fn(pred_ema, lab) / (label.shape[0] // num_at_once) / len(data_loader)
                up_tmp, down_tmp = dice_score_fn(pred_ema, lab)
                up_ema += up_tmp.item()
                down_ema += down_tmp.item()
                inter, uni = IoU(pred_ema, lab, hard_label=True)
                intersections_ema += inter.item()
                unions_ema += uni.item()
        iou= (intersections + 1) / (unions + 1)
        dice_score = (up + 1) / (down + 1)
        iou_ema = (intersections_ema + 1) / (unions_ema + 1)
        dice_score_ema = (up_ema + 1) / (down_ema + 1)
        print('--------------------')
        print('val loss:', loss)
        print('dice_score:', dice_score)
        writer.add_scalar('val loss:', loss, epoch)
        writer.add_scalar('dice_score:', dice_score, epoch)
        writer.add_scalar('IoU:', iou, epoch)
        writer.add_scalar('val loss ema:', loss_ema, epoch)
        writer.add_scalar('dice_score ema:', dice_score_ema, epoch)
        writer.add_scalar('IoU ema:', iou_ema, epoch)
        print('--------------------')
    model.train()
    teacher.train()
    return dice_score_ema


def prediction(model, data_loader, config, identifier, epoch):
    model.eval()
    with torch.no_grad():
        predictions = list()
        for input, id in data_loader:
            input = input.to(config.DEVICE)
            for i in range(input.shape[0]):
                inp = input[i:i + 1]
                pred = model(inp.unsqueeze(1)).squeeze(1)
                predictions.append([pred.cpu().numpy(), id])
                #print(id)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    for pred, id in predictions:
        original_img = nib.load(
            ''.join(('/data/ssd1/kienzlda/data_covidsegmentation/COVID-19-20_v2/Validation/volume-covid19-A-0', id, '_ct.nii.gz'))
        )
        correct_shape = original_img.get_fdata().shape
        correct_header = original_img.header
        scale_factors = (correct_shape[-3] / pred.shape[-3], correct_shape[-2] / pred.shape[-2], correct_shape[-1] / pred.shape[-1])
        pred = ndimage.zoom(pred.squeeze(0), scale_factors)
        pred = sigmoid(pred) - 0.5
        pred = np.heaviside(pred, pred)
        savepath = os.path.join(config.LOGS_PATH, 'predictions', identifier.split('/')[-1], str(epoch))
        os.makedirs(savepath, exist_ok=True)
        img = nib.Nifti1Image(pred, original_img.affine)
        for key in correct_header:
            img.header[key] = correct_header[key]
        nib.save(img, os.path.join(savepath, str(id) + '.nii.gz'))
        print('saved', os.path.join(savepath, str(id) + '.nii.gz'))
    model.train()


def main():
    print('PID:', os.getpid())
    torch.set_num_threads(1)
    set_deterministic()
    args = parse_args()
    config = BaseConfig(args)
    config.modelconfig = get_modelconfig(config)
    run(config, args)

if __name__ == '__main__':
    main()