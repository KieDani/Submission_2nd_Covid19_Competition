"""
This file is to be placed and executed in the data folder next to "data/mha" and will
produce a downscaled dataset. (128^3).
This new dataset will be in "data/npy_128"

"""

import glob
from multiprocessing import Pool
import numpy as np
import medpy.io as mpio
import torch
import tqdm
import scipy.ndimage as ndimage
import os

from training.config.config import BaseConfig

os.makedirs("npy_128", exist_ok=True)

config = BaseConfig()
data_path = os.path.join(config.DATA_PATH, "data")

mha_paths = sorted(list(glob.glob(os.path.join(data_path, "mha/*.mha"))))
print("Detected files num: ", len(mha_paths))

def downscale_paths(ind_list):
    print(ind_list)
    for i in ind_list:
        p = mha_paths[i]

        voxel_tensor = mpio.load(p)[0]
        scale_factors = (0.25, 0.25, 128 / voxel_tensor.shape[-1])
        voxel_tensor = ndimage.zoom(voxel_tensor, scale_factors)
        voxel_tensor = np.clip(voxel_tensor, -1000., 500.)
        voxel_tensor = (voxel_tensor + 1000) / 750. - 1.0

        name = p.split("/")[-1][:-4]

        np.save("npy_128/"+name+".npy", voxel_tensor)
        print("Saved tensor no.", name)


pool = Pool(1)
path_lists = [range(100*i, 100*i+100) for i in range(20)]
pool.map(downscale_paths, path_lists)