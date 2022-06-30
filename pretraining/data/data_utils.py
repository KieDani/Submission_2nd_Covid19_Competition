import nibabel as nib
import numpy as np
import os
import shutil
import scipy.ndimage as ndimage
import medpy.io as mpio
import pandas as pd
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from paths import pretraining_DATA_PATH, pretraining_PATH_SEGMENTATION, pretraining_PATH_STOIC, pretraining_PATH_FULLTCIA

def create_resized_dataset(size=(256, 256, 256), name='resized256'):
    path_original = pretraining_PATH_SEGMENTATION
    path_original_train = os.path.join(path_original, 'Train')
    path_original_val = os.path.join(path_original, 'Validation')
    path_resized = os.path.join(pretraining_DATA_PATH, '..', name)
    path_resized_train = os.path.join(path_resized, 'Train')
    path_resized_val = os.path.join(path_resized, 'Validation')
    os.makedirs(path_resized_train, exist_ok=True)
    os.makedirs(path_resized_val, exist_ok=True)

    for path_o, path_r in [(path_original_train, path_resized_train), (path_original_val, path_resized_val)]:
        for data_name in tqdm(sorted(os.listdir(path_o))):
            print(data_name)
            source = nib.load(os.path.join(path_o, data_name))
            source = source.get_fdata()

            scale_factors = (size[0] / source.shape[-3], size[1] / source.shape[-2], size[2] / source.shape[-1])
            if data_name.strip('.nii.gz')[-2:] == 'ct':
                target = ndimage.zoom(source, scale_factors)
                target = np.clip(target, -1000., 500.)
                target = (target + 1000) / 1500
            else:
                target = ndimage.zoom(source, scale_factors, order=0)

            np.save(os.path.join(path_r, data_name).strip('.nii.gz') + '.npy', target)



def create_resized_unsupervised_dataset_stoic(size=(256, 256, 128), name='resized256'):
    path_original = os.path.join(pretraining_PATH_STOIC, 'data', 'mha')
    path_resized = os.path.join(pretraining_DATA_PATH, '..', name, 'unsupervised_stoic')
    os.makedirs(path_resized, exist_ok=True)

    reference_file = pd.read_csv(os.path.join(pretraining_PATH_STOIC, 'metadata', 'reference.csv'))
    for i, patient_id in tqdm(enumerate(reference_file['PatientID'])):
        if int(reference_file['probCOVID'][i]) == 1:
            data_name = str(int(patient_id)) + '.mha'
            p = os.path.join(path_original, data_name)
            source = mpio.load(p)[0]
            scale_factors = (size[0] / source.shape[-3], size[1] / source.shape[-2], size[2] / source.shape[-1])
            target = ndimage.zoom(source, scale_factors)
            target = np.clip(target, -1000., 500.)
            target = (target + 1000) / 1500

            np.save(os.path.join(path_resized, data_name).strip('.mha') + '.npy', target)

    print('finished!')


def create_resized_unsupervised_dataset_tcia(size=(256, 256, 128), name='resized256'):
    path_original = os.path.join(pretraining_PATH_FULLTCIA, 'set1')
    path_resized = os.path.join(pretraining_DATA_PATH, '..', name, 'unsupervised_tcia')
    os.makedirs(path_resized, exist_ok=True)
    path_supervised = pretraining_PATH_SEGMENTATION
    path_supervised_train = os.path.join(path_supervised, 'Train')
    path_supervised_val = os.path.join(path_supervised, 'Validation')

    #save statistics of each image in supervised dataset in list_supervised
    list_supervised = []
    for path_s in [path_supervised_train, path_supervised_val]:
        for data_name in sorted(os.listdir(path_s)):
            source = nib.load(os.path.join(path_s, data_name))
            source = source.get_fdata()
            list_supervised.append((data_name, np.sum(source), source.shape))

    list_duplicates = []
    for data_name in sorted(os.listdir(path_original)):
        print(data_name)
        source = nib.load(os.path.join(path_original, data_name))
        source = source.get_fdata()
        su = np.sum(source)
        sh = np.shape
        is_supervised = False
        #test if data is already in supervised dataset
        for elem in list_supervised:
            if abs(elem[1] - np.sum(source)) < 10 and source.shape == elem[2]:
                is_supervised = True
                list_duplicates.append((elem[0], data_name))
                print('Duplicate!!!')
                break
        if is_supervised == False:
            scale_factors = (size[0] / source.shape[-3], size[1] / source.shape[-2], size[2] / source.shape[-1])
            target = ndimage.zoom(source, scale_factors)
            target = np.clip(target, -1000., 500.)
            target = (target + 1000) / 1500

            np.save(os.path.join(path_resized, data_name).strip('.nii.gz') + '.npy', target)

    print('number supervised data:', len(list_supervised))
    print('number duplicates:', len(list_duplicates))
    print('finished!')


def create_resized_unsupervised_dataset_MosMed(size=(256, 256, 128), name='resized256'):
    path_original = 'NOTUSED'
    path_resized = 'NOTUSED'
    os.makedirs(path_resized, exist_ok=True)

    for folders in tqdm(['CT-1',  'CT-2',  'CT-3',  'CT-4']):
        path = os.path.join(path_original, folders)
        for data_name in sorted(os.listdir(path)):
            #print(data_name)
            p = os.path.join(path, data_name)
            source = np.load(p)
            scale_factors = (size[0] / source.shape[-3], size[1] / source.shape[-2], size[2] / source.shape[-1])
            target = ndimage.zoom(source, scale_factors)
            target = np.clip(target, -1000., 500.)
            target = (target + 1000) / 1500

            np.save(os.path.join(path_resized, data_name), target)

    print('finished!')


def calc_stats_supervised():
    path_supervised = 'NOTUSED'
    mean, var, num = 0, 0, 0
    for data_name in tqdm(sorted(os.listdir(path_supervised))):
        if data_name.split('.')[-2][-2:] == 'ct':
            #print(data_name)
            p = os.path.join(path_supervised, data_name)
            img = np.load(p)
            num += 1
            mean += mean.sum()
            var += img.var()

    print('TCIA supervised stats')
    print('------')
    total_mean = mean / num
    total_var = var / num
    total_std = np.sqrt(total_var)
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


def calc_stats_unsupervised_stoic():
    path_unsupervised = 'NOTUSED'
    mean, var, num = 0, 0, 0
    for data_name in tqdm(sorted(os.listdir(path_unsupervised))):
        #print(data_name)
        p = os.path.join(path_unsupervised, data_name)
        img = np.load(p)
        num += 1
        mean += img.mean()
        var += img.var()

    print('STOIC stats')
    print('------')
    total_mean = mean / num
    total_var = var / num
    total_std = np.sqrt(total_var)
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


def calc_stats_unsupervised_tcia():
    path_unsupervised = 'NOTUSED'
    mean, var, num = 0, 0, 0
    for data_name in tqdm(sorted(os.listdir(path_unsupervised))):
        #print(data_name)
        p = os.path.join(path_unsupervised, data_name)
        img = np.load(p)
        num += 1
        mean += img.mean()
        var += img.var()

    print('------')
    total_mean = mean / num
    total_var = var / num
    total_std = np.sqrt(total_var)
    print('TCIA unsupervised stats')
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


def calc_stats_tcia():
    path_supervised = 'NOTUSED'
    path_supervised2 = 'NOTUSED'
    path_unsupervised = 'NOTUSED'
    mean, var, num = 0, 0, 0
    for path in tqdm([path_supervised, path_unsupervised]):#[path_supervised, path_supervised2, path_unsupervised]:
        for data_name in sorted(os.listdir(path)):
            if data_name.split('.')[-2][-2:] == 'ct' or path == path_unsupervised:
                # print(data_name)
                p = os.path.join(path, data_name)
                img = np.load(p)
                num += 1
                mean += img.mean()
                var += img.var()

    print('------')
    total_mean = mean / num
    total_var = var / num
    total_std = np.sqrt(total_var)
    print('TCIA stats')
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


def calc_stats_unsupervised_mosmed():
    path_unsupervised = 'NOTUSED'
    mean, var, num = 0, 0, 0
    for data_name in tqdm(sorted(os.listdir(path_unsupervised))):
        #print(data_name)
        p = os.path.join(path_unsupervised, data_name)
        img = np.load(p)
        num += 1
        mean += img.mean()
        var += img.var()

    print('------')
    total_mean = mean / num
    total_var = var / num
    total_std = np.sqrt(total_var)
    print('MosMed stats')
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


def calc_stats_unsupervised_mia():
    path_unsupervised = 'NOTUSED'
    mean, var, num = 0, 0, 0
    for data_name in tqdm(sorted(os.listdir(path_unsupervised))):
        #print(data_name)
        p = os.path.join(path_unsupervised, data_name)
        img = np.load(p)
        num += 1
        mean += img.mean()
        var += img.var()

    print('------')
    total_mean = mean / num
    total_var = var / num
    total_std = np.sqrt(total_var)
    print('Mia stats')
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


def compress_arrays(name = 'resized256'):
    def wrapper(da):
        source_path = os.path.join(pretraining_DATA_PATH, '...', name)
        target_path = os.path.join(pretraining_DATA_PATH, '...', name, '_compressed')
        source_path2 = os.path.join(source_path, da)
        target_path2 = os.path.join(target_path, da)
        os.makedirs(target_path2, exist_ok=True)
        for file in sorted(os.listdir(source_path2)):
            source = np.load(os.path.join(source_path2, file))
            target_file = os.path.join(target_path2, file).strip('.npy') + '.npz'
            np.savez_compressed(target_file, source)
            print('saved', target_file)

    datasets = ['Train', 'Validation', 'unsupervised_tcia', 'unsupervised_stoic', 'unsupervised_mosmed']
    with Pool(len(datasets)) as p:
        p.map(wrapper, datasets)


def convert_mia_cache(name = 'stoicECCV/resized256'):
    path = os.path.join(pretraining_DATA_PATH, '...', name, 'unsupervised_mia')
    files = sorted(os.listdir(path))
    for filename in tqdm(files):
        filepath = os.path.join(path, filename)
        array = np.load(filepath)['img']
        np.save(filepath.strip('.npz') + '.npy', array)
        os.remove(filepath)


if __name__ == '__main__':
    create_resized_unsupervised_dataset_tcia(size=(224, 224, 112), name='resized224')
    create_resized_unsupervised_dataset_tcia(size=(256, 256, 128), name='resized256')
    create_resized_unsupervised_dataset_stoic(size=(224, 224, 112), name='resized224')
    create_resized_unsupervised_dataset_stoic(size=(256, 256, 128), name='resized256')

    convert_mia_cache(name='resized224')
    convert_mia_cache(name='resized256')

    # TCIA
    # unsupervised
    # stats
    # mean: 0.220458088175389
    # var: 0.09130369404896298
    # std: 0.30216501129178236
    # MosMed
    # stats
    # mean: 0.31327220923026
    # var: 0.09981122681515282
    # std: 0.31592914841013453
    # STOIC
    # stats
    # ------
    # mean: 0.3140890660228576
    # var: 0.0996650629054726
    # std: 0.3156977397851822
    # MIA
    # mean: 0.3970858115872199
    # var: 0.10284976031292568
    # std: 0.32070198052541815






