import torch
import torchvision
import numpy as np
import os
import random
from pretraining.data.mosmed import get_mosmed_dataset
from pretraining.config.dataconfig import MosmedConfig, HustConfig
from pretraining.data.hust import get_hust_dataset
import pandas as pd
from paths import pretraining_DATA_PATH


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, path = pretraining_DATA_PATH, transform=None, train=True, reduction_factor = 1):
        path = os.path.join(path, 'Train')
        path2 = 'Train'
        self.path = path
        self.path2 = path2
        self.transform = transform
        self.reduction_factor = reduction_factor
        self.load_compressed = lambda file_path: np.load(file_path)['arr_0']
        self.load = self.load_compressed if path.find('compressed') != -1 else lambda file_path: np.load(file_path)

        filenamelist = []
        dataname_old = None
        for dataname in sorted(os.listdir(path)):
            dataname = os.path.join(path2, dataname)
            if dataname_old == None:
                dataname_old = dataname
            else:
                filenamelist.append([dataname_old, dataname])
                dataname_old = None
        train_len = int(0.85*len(filenamelist))
        val_len = len(filenamelist) - train_len
        rand = random.Random(42)
        rand.shuffle(filenamelist)
        self.filenamelist = filenamelist[:train_len:reduction_factor] if train else filenamelist[train_len:]

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        img, label = self.filenamelist[idx]
        #img, label = self.load(img).astype(np.float32), self.load(label).astype(np.float32)
        if self.transform:
            img, label = self.transform(img, label)
        #img, label = np.expand_dims(img, axis=0), np.expand_dims(label, axis=0)
        img, label = img.unsqueeze(0), label.unsqueeze(0)
        return img, label


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, path = pretraining_DATA_PATH, transform=None):
        path = os.path.join(path, 'Validation')
        path2 = 'Validation'
        self.path = path
        self.path2 = path2
        self.transform = transform
        self.load_compressed = lambda file_path: np.load(file_path)['arr_0']
        self.load = self.load_compressed if path.find('compressed') != -1 else lambda file_path: np.load(file_path)

        filenamelist = [os.path.join(path2, dataname) for dataname in sorted(os.listdir(path))]
        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        img = self.filenamelist[idx]
        id = img.split(os.path.join(self.path2, 'volume-covid19-A-0'))[-1].strip('_ct.npy')
        #img = self.load(img).astype(np.float32)
        if self.transform:
            img, __ = self.transform(img, None)
        #img = np.expand_dims(img, axis=0)
        img = img.unsqueeze(0)
        return img, id


class UnsupervisedDataset_stoic(torch.utils.data.Dataset):
    def __init__(self, path = pretraining_DATA_PATH, transform=None):
        path = os.path.join(path, 'unsupervised_stoic')
        self.path = path
        self.transform = transform

        filenamelist = [os.path.join(path, dataname) for dataname in sorted(os.listdir(path))]
        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        img = self.filenamelist[idx]
        img = np.load(img).astype(np.float32)
        if self.transform:
            img, __ = self.transform(img, None)
        #img = np.expand_dims(img, axis=0)
        img = img.unsqueeze(0)
        return img

class UnsupervisedDataset_tcia(torch.utils.data.Dataset):
    def __init__(self, path = pretraining_DATA_PATH, transform=None):
        path = os.path.join(path, 'unsupervised_tcia')
        self.path = path
        self.transform = transform

        filenamelist = [os.path.join(path, dataname) for dataname in sorted(os.listdir(path))]
        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        img = self.filenamelist[idx]
        #id = img.split(os.path.join(self.path, 'volume-covid19-A-0'))[-1].strip('.npy')
        img = np.load(img).astype(np.float32)
        #img = np.expand_dims(img, axis=0)
        if self.transform:
            img, __ = self.transform(img, None)
        img = img.unsqueeze(0)
        return img#, id

class SequentialDataset_tcia(torch.utils.data.Dataset):
    def __init__(self, path = pretraining_DATA_PATH, label_path = None, transform=None, severity=None):
        path = os.path.join(path, 'unsupervised_tcia')
        path2 = 'unsupervised_tcia'
        self.path = path
        self.path2 = path2
        self.label_path = label_path
        self.transform = transform
        self.severity = severity
        self.load_compressed = lambda file_path: np.load(file_path)['arr_0']
        self.load = self.load_compressed if path.find('compressed') != -1 else lambda file_path: np.load(file_path)

        if severity == True and label_path is not None:
            sev_labels = pd.read_csv(os.path.join(label_path, 'reference_unsupervised.csv'))

        filenamelist = list()
        for dataname in sorted(os.listdir(path)):
            if severity is None:
                if label_path is None:
                    filenamelist.append((os.path.join(path2, dataname), None))
                else:
                    filenamelist.append((os.path.join(path2, dataname), os.path.join(label_path, dataname)))
            else:
                if label_path is None:
                    if severity == True:
                        id = dataname.strip('.npy').strip('.npz')
                        sev = int(sev_labels['probSevere'][sev_labels.index[sev_labels['PatientID'] == id]])
                    else:
                        sev = None
                    filenamelist.append((os.path.join(path2, dataname), None, sev))
                else:
                    if severity == True:
                        id = dataname.strip('.npy').strip('.npz')
                        sev = int(sev_labels['probSevere'][sev_labels.index[sev_labels['PatientID'] == id]])
                    else:
                        sev = None
                    filenamelist.append((os.path.join(path2, dataname), os.path.join(label_path, dataname), sev))

        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        if self.severity is None:
            im_path, lab_path = self.filenamelist[idx]
            sev = None
        else:
            im_path, lab_path, sev = self.filenamelist[idx]
        sev = -1 if sev is None else sev
        #img = self.load(im_path).astype(np.float32)
        if self.severity is None and lab_path is not None:
            #label = self.load_compressed(lab_path).astype(np.float32)
            pass
        else:
            lab_path = None
        if self.transform:
            img, label = self.transform(im_path, lab_path)
        img = img.unsqueeze(0)
        label = label.unsqueeze(0) if label is not None else None
        if self.severity is None:
            if label is not None:
                return img, label
            else:
                return img, os.path.basename(im_path)
        else:
            if label is not None:
                return img, label, sev
            else:
                return img, os.path.basename(im_path), sev

class SequentialDataset_stoic(torch.utils.data.Dataset):
    def __init__(self, path = pretraining_DATA_PATH, label_path = None, transform=None, severity=None, train=True, reduction_factor = 1):
        path = os.path.join(path, 'unsupervised_stoic')
        path2 = 'unsupervised_stoic'
        self.path = path
        self.path2 = path2
        self.label_path = label_path
        self.transform = transform
        self.severity = severity
        self.reduction_factor = reduction_factor
        self.load_compressed = lambda file_path: np.load(file_path)['arr_0']
        self.load = self.load_compressed if path.find('compressed') != -1 else lambda file_path: np.load(file_path)

        if severity == True:
            sev_labels = pd.read_csv(os.path.join(path, '..', 'reference_stoic.csv'))

        filenamelist = list()
        for dataname in sorted(os.listdir(path)):
            if severity is None:
                if label_path is None:
                    filenamelist.append((os.path.join(path2, dataname), None))
                else:
                    filenamelist.append((os.path.join(path2, dataname), os.path.join(label_path, dataname)))
            else:
                if label_path is None:
                    if severity == True:
                        id = int(dataname.strip('.npy').strip('.npz'))
                        sev = int(sev_labels['probSevere'][sev_labels.index[sev_labels['PatientID'] == id]])
                    else:
                        sev = None
                    filenamelist.append((os.path.join(path2, dataname), None, sev))
                else:
                    if severity == True:
                        id = int(dataname.strip('.npy').strip('.npz'))
                        sev = int(sev_labels['probSevere'][sev_labels.index[sev_labels['PatientID'] == id]])
                    else:
                        sev = None
                    filenamelist.append((os.path.join(path2, dataname), os.path.join(label_path, dataname), sev))

        if severity == True and train is not None:
            train_len = int(0.70 * len(filenamelist))
            val_len = len(filenamelist) - train_len
            rand = random.Random(42)
            rand.shuffle(filenamelist)
            filenamelist = filenamelist[:train_len:reduction_factor] if train else filenamelist[train_len:]
        else:
            train_len = int(0.80 * len(filenamelist))
            rand = random.Random(42)
            rand.shuffle(filenamelist)
            filenamelist = filenamelist[:train_len:reduction_factor] + filenamelist[train_len:]
        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        if self.severity is None:
            im_path, lab_path = self.filenamelist[idx]
            sev = None
        else:
            im_path, lab_path, sev = self.filenamelist[idx]
        sev = -1 if sev is None else sev
        #img = self.load(im_path).astype(np.float32)
        #if self.severity is None and lab_path is not None:
        if lab_path is not None:
            #label = self.load_compressed(lab_path).astype(np.float32)
            pass
        else:
            lab_path = None
        if self.transform:
            img, label = self.transform(im_path, lab_path)
        img = img.unsqueeze(0)
        label = label.unsqueeze(0) if label is not None else None

        if self.severity is None:
            if label is not None:
                return img, label
            else:
                return img, os.path.basename(im_path)
        else:
            if label is not None:
                return img, label, sev
            else:
                return img, os.path.basename(im_path), sev



class SequentialDataset_mosmed(torch.utils.data.Dataset):
    def __init__(self, path = pretraining_DATA_PATH, label_path = None, transform=None, severity=None):
        path = os.path.join(path, 'unsupervised_mosmed')
        path2 = 'unsupervised_mosmed'
        self.path = path
        self.path2 = path2
        self.label_path = label_path
        self.transform = transform
        self.severity = severity
        self.load_compressed = lambda file_path: np.load(file_path)['arr_0']
        self.load = self.load_compressed if path.find('compressed') != -1 else lambda file_path: np.load(file_path)

        if severity == True and label_path is not None:
            sev_labels = pd.read_csv(os.path.join(label_path, 'reference_unsupervised.csv'))

        filenamelist = list()
        for dataname in sorted(os.listdir(path)):
            if severity is None:
                if label_path is None:
                    filenamelist.append((os.path.join(path2, dataname), None))
                else:
                    filenamelist.append((os.path.join(path2, dataname), os.path.join(label_path, dataname)))
            else:
                if label_path is None:
                    if severity == True:
                        id = dataname.strip('.npy').strip('.npz')
                        sev = int(sev_labels['probSevere'][sev_labels.index[sev_labels['PatientID'] == id]])
                    else:
                        sev = None
                    filenamelist.append((os.path.join(path2, dataname), None, sev))
                else:
                    if severity == True:
                        id = dataname.strip('.npy').strip('.npz')
                        sev = int(sev_labels['probSevere'][sev_labels.index[sev_labels['PatientID'] == id]])
                    else:
                        sev = None
                    filenamelist.append((os.path.join(path2, dataname), os.path.join(label_path, dataname), sev))

        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        if self.severity is None:
            im_path, lab_path = self.filenamelist[idx]
            sev = None
        else:
            im_path, lab_path, sev = self.filenamelist[idx]
        sev = -1 if sev is None else sev
        #img = self.load(im_path).astype(np.float32)
        if self.severity is None and lab_path is not None:
            #label = self.load_compressed(lab_path).astype(np.float32)
            pass
        else:
            lab_path = None
        if self.transform:
            img, label = self.transform(im_path, lab_path)
        img = img.unsqueeze(0)
        label = label.unsqueeze(0) if label is not None else None
        if self.severity is None:
            if label is not None:
                return img, label
            else:
                return img, os.path.basename(im_path)
        else:
            if label is not None:
                return img, label, sev
            else:
                return img, os.path.basename(im_path), sev


class SequentialDataset_mia(torch.utils.data.Dataset):
    def __init__(self, path = pretraining_DATA_PATH, label_path = None, transform=None, severity=None, train=None, reduction_factor=1):
        path = os.path.join(path, 'unsupervised_mia')
        path2 = 'unsupervised_mia'
        self.path = path
        self.path2 = path2
        self.label_path = label_path
        self.transform = transform
        self.severity = severity
        self.load_compressed = lambda file_path: np.load(file_path)['arr_0']
        self.load = self.load_compressed if path.find('compressed') != -1 else lambda file_path: np.load(file_path)

        if severity == True:# and label_path is not None:
            sev_labels = pd.read_csv(os.path.join(path, '..', 'reference_mia.csv'))

        filenamelist = list()
        for dataname in sorted(os.listdir(path)):
            if severity is None:
                if label_path is None:
                    filenamelist.append((os.path.join(path2, dataname), None))
                else:
                    filenamelist.append((os.path.join(path2, dataname), os.path.join(label_path, dataname)))
            else:
                if label_path is None:
                    if severity == True:
                        id = dataname.strip('.npy').strip('.npz').split('-')[-1]
                        sev = int(sev_labels['Sev'][sev_labels.index[sev_labels['Path'] == 'train_cov19d/covid/' + id]])
                        if sev in [0, 1]: sev = 0
                        elif sev in [2, 3]: sev = 1
                        else: sev = -1
                        if sev_labels['Set'][sev_labels.index[sev_labels['Path'] == 'train_cov19d/covid/' + id]].values[0] != 'train':
                            print('MISTAKE!!! Do not use validation images for semi-supervised training!')
                    else:
                        sev = None
                    if sev != -1: filenamelist.append((os.path.join(path2, dataname), None, sev))
                else:
                    if severity == True:
                        id = dataname.strip('.npy').strip('.npz').split('-')[-1]
                        sev = int(sev_labels['Sev'][sev_labels.index[sev_labels['Path'] == 'train_cov19d/covid/' + id]])
                        #4 classes to 2 classes
                        if sev in [0, 1]: sev = 0
                        elif sev in [2, 3]: sev = 1
                        else: sev = -1
                        if sev_labels['Set'][sev_labels.index[sev_labels['Path'] == 'train_cov19d/covid/' + id]].values[0] != 'train':
                            print('MISTAKE!!! Do not use validation images for semi-supervised training!')
                    else:
                        sev = None
                    if sev != -1: filenamelist.append((os.path.join(path2, dataname), os.path.join(label_path, dataname), sev))

        if severity == True and train is not None:
            train_len = int(0.80 * len(filenamelist))
            val_len = len(filenamelist) - train_len
            rand = random.Random(42)
            rand.shuffle(filenamelist)
            filenamelist = filenamelist[:train_len:reduction_factor] if train else filenamelist[train_len:]
        else:
            train_len = int(0.80 * len(filenamelist))
            rand = random.Random(42)
            rand.shuffle(filenamelist)
            filenamelist = filenamelist[:train_len:reduction_factor] + filenamelist[train_len:]

        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        if self.severity is None:
            im_path, lab_path = self.filenamelist[idx]
            sev = None
        else:
            im_path, lab_path, sev = self.filenamelist[idx]
        sev = -1 if sev is None else sev
        #img = self.load(im_path).astype(np.float32)
        if self.severity is None and lab_path is not None:
            #label = self.load_compressed(lab_path).astype(np.float32)
            pass
        else:
            #lab_path = None
            pass
        if self.transform:
            img, label = self.transform(im_path, lab_path)
        img = img.unsqueeze(0)
        label = label.unsqueeze(0) if label is not None else None
        if self.severity is None:
            if label is not None:
                return img, label
            else:
                return img, os.path.basename(im_path)
        else:
            if label is not None:
                return img, label, sev
            else:
                return img, os.path.basename(im_path), sev



class MosMedDataset_semisupervised(torch.utils.data.Dataset):
    def __init__(self, path = pretraining_DATA_PATH, transform=None):
        path = os.path.join(path, 'unsupervised_mosmed')
        self.path = path
        self.transform = transform

        filenamelist = [os.path.join(path, dataname) for dataname in sorted(os.listdir(path))]
        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        img = self.filenamelist[idx]
        img = np.load(img).astype(np.float32)
        if self.transform:
            img, __ = self.transform(img, None)
        img = img.unsqueeze(0)
        return img


class MosMedDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode = 'train'):
        dataconfig_mos = MosmedConfig(config)
        dataconfig_mos.do_normalization = config.DO_NORMALIZATION if hasattr(config, 'DO_NORMALIZATION') else dataconfig_mos.do_normalization
        dataset = get_mosmed_dataset(dataconfig_mos)
        idxs = [i for i in range(len(dataset))]
        rand = random.Random(42)
        rand.shuffle(idxs)
        self.idxs = idxs[:int(0.8*len(dataset))] if mode == 'train' else idxs[int(0.8*len(dataset)):]
        self.dataset = dataset

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

class HustDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode = 'train'):
        dataconfig_hust = HustConfig(config)
        dataconfig_hust.do_normalization = config.DO_NORMALIZATION if hasattr(config, 'DO_NORMALIZATION') else dataconfig_hust.do_normalization
        dataset = get_hust_dataset(dataconfig_hust)
        idxs = [i for i in range(len(dataset))]
        rand = random.Random(42)
        rand.shuffle(idxs)
        self.idxs = idxs[:int(0.8*len(dataset))] if mode == 'train' else idxs[int(0.8*len(dataset)):]
        self.dataset = dataset

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]






if __name__ == '__main__':
    SDS = UnsupervisedDataset_stoic()
    print(len(SDS))
    supervised_loader = torch.utils.data.DataLoader(SDS, batch_size=1)

    for img in supervised_loader:
        print(img.shape)
        #print(label.shape)
        break