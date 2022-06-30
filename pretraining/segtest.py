import torch
import numpy as np
import os
import nibabel as nib
from data.transforms3D import ToTensor3D, Normalize3D, get_transform, elastic_deform_robin
from data.transforms3D import Compose as My_Compose
from config.config import BaseConfig
from config.modelconfig import get_model
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
import csv
from paths import pretraining_DATA_PATH, pretraining_PATH_STOIC

def get_unsupervised(idx, transform=None):
    os.path.join(pretraining_DATA_PATH, 'unsupervised')
    path = os.path.join(pretraining_DATA_PATH, 'unsupervised')
    filenamelist = [os.path.join(path, dataname) for dataname in sorted(os.listdir(path))]
    imgname = filenamelist[idx]
    print(imgname)
    img = np.load(imgname).astype(np.float32)
    if transform is not None:
        img = transform(img)

    metadatapath = os.path.join(pretraining_PATH_STOIC, 'metadata', 'reference.csv')
    with open(metadatapath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == str(imgname.split('/')[-1].strip('.npy')):
                print(row)

    return img

def get_supervised(idx, transform=None, target_transform=None):
    path = 'NOTUSED'
    filenamelist = []
    dataname_old = None
    for dataname in sorted(os.listdir(path)):
        dataname = os.path.join(path, dataname)
        if dataname_old == None:
            dataname_old = dataname
        else:
            filenamelist.append([dataname_old, dataname])
            dataname_old = None

    img, label = filenamelist[idx]
    print(img)
    print(label)
    img, label = np.load(img).astype(np.float32), np.load(label).astype(np.float32)
    print('...')
    if transform is not None:
        img = transform(img)
    if target_transform is not None:
        label = target_transform(label)
    return img, label


def load_model(savepath):
    config = BaseConfig()
    model, modelconfig = get_model(config)
    checkpoint = torch.load(savepath)
    model_sd = model.state_dict()
    for name in checkpoint['model_state_dict']:
        model_sd[name] = checkpoint['model_state_dict'][name]
    model.load_state_dict(model_sd)
    model.eval()
    print('loaded model')
    return model


def plot(img, label=None, slice=64, title=''):
    print(title, img.shape)
    plt.imshow(img[:, :, slice], cmap='gray')
    if label is not None:
        plt.imshow(label[:, :, slice], cmap='jet', alpha=0.25)
    plt.title(title)
    plt.show()

def get_severe_list():
    metadatapath = os.path.join(pretraining_PATH_STOIC, 'metadata', 'reference.csv')
    severelist_id = list()
    with open(metadatapath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[2] == str(1):
                severelist_id.append(row[0])

    path = os.path.join(pretraining_DATA_PATH, 'unsupervised')
    filenamelist = [os.path.join(path, dataname) for dataname in sorted(os.listdir(path))]
    severelist = list()
    for id in severelist_id:
        for idx, file in enumerate(filenamelist):
            file = file.split('/')[-1].strip('.npy')
            if id == file:
                severelist.append(idx)
    return severelist

def main():
    model = load_model(savepath='PATH/TO/CHECKPOINT')
    index = 42
    supslice = 50
    normalize_supervised = Normalize3D(mean=0.22086217807487526, std=0.30519881253497455)
    normalize_unsupervised = Normalize3D(mean=0.30582652609574806, std=0.32016407586950973)
    supimg, label = get_supervised(index)
    plot(supimg, slice=supslice, title='supervised image')
    plot(supimg, label, slice=supslice, title='label')
    with torch.no_grad():
        transform = Compose([ToTensor3D(), normalize_supervised])
        #transform = ToTensor3D()
        input = transform(supimg)
        input = input.to('cpu')
        suppred = model(input.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        suppred =  torch.sigmoid(suppred)
    suppred.cpu()
    plot(supimg, suppred, slice=supslice, title='supervised prediction')

    print('-----------------------')

    #quit()

    print(get_severe_list())
    index = get_severe_list()[0]
    for unsupslice in [60, 70, 80, 90, 100]:
        unsupimg = get_unsupervised(index)
        plot(unsupimg, slice=unsupslice, title='unsupervised image')
        with torch.no_grad():
            transform = Compose([ToTensor3D(), normalize_unsupervised])
            input = transform(unsupimg)
            input = input.to('cpu')
            unsuppred = model(input.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            unsuppred = torch.sigmoid(unsuppred)
        unsuppred.cpu()
        plot(unsupimg, unsuppred, slice=unsupslice, title='unsupervised prediction')


def visualize_transformations():
    normalize_supervised = Normalize3D(mean=0.22086217807487526, std=0.30519881253497455)
    normalize_unsupervised = Normalize3D(mean=0.30582652609574806, std=0.32016407586950973)
    transform = My_Compose([normalize_supervised, get_transform(imsize=256)])
    deformtransform = elastic_deform_robin
    supslice = 50
    index = 43
    for alpha in [2, 5, 7]:
        supimg, label = get_supervised(index)
        plot(supimg, slice=supslice, title='supervised image')
        plot(supimg, label, slice=supslice, title='label')
        plot(label, slice=supslice, title='label')

        # trans_img, trans_label = transform(supimg, label)
        # plot(trans_img, slice=supslice, title='transformed image')
        # plot(trans_img, trans_label, slice=supslice, title='transformed label')

        deform_img, deform_label = torch.from_numpy(supimg).unsqueeze(0).unsqueeze(0).to('cuda:4'), torch.from_numpy(label).unsqueeze(0).unsqueeze(0).to('cuda:4')
        with torch.no_grad():
            deform_img, deform_label = elastic_deform_robin(image=deform_img, target=deform_label, sigma=35, alpha=alpha, device='cuda:4')
        deform_img, deform_label = deform_img.squeeze(0).squeeze(0).to('cpu'), deform_label.squeeze(0).squeeze(0).to('cpu')
        plot(deform_img, slice=supslice, title='deformed image')
        plot(deform_img, deform_label, slice=supslice, title='deformed label')
        plot(deform_label, slice=supslice, title='deformed label')

if __name__ == '__main__':
    #main()
    visualize_transformations()