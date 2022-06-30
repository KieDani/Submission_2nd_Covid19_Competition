import numpy as np
import torch
import elasticdeform.torch as etorch
import elasticdeform
import numpy as np
import torchvision as tv
from scipy import ndimage
import random
import torch.nn.functional as F
import os
from paths import pretraining_DATA_PATH

class ToTensor3D(object):
    def __init__(self):
        super(ToTensor3D, self).__init__()

    def __call__(self, sample, target=None):
        sample = torch.from_numpy(sample.copy())
        if target is not None:
            target = torch.from_numpy(target.copy())
        return sample, target


class ElasticDeform(object):
    def __init__(self):
        super(ElasticDeform, self).__init__()

    def forward(self, sample, strength=1):
        displacement = torch.randn(3, 3, 3, 3) * strength
        sample = etorch.deform_grid(sample, displacement, order=3)
        return sample


def elastic_deformation(sample, target=None, strength=2):
    displacement = torch.randn(3, 3, 3, 3) * strength
    if target is None:
        sample = etorch.deform_grid(sample, displacement, order=3)
    else:
        [sample, target] = etorch.deform_grid([sample, target], displacement, order=[3, 0])
    return sample, target

def elastic_wrapper(x):
    input, label = x
    input, label = elastic_deformation(input.squeeze(0), label.squeeze(0))
    input, label = input.unsqueeze(0), label.unsqueeze(0)
    # input, label = input, label
    return (input, label)


def filter_gaussian_3d(input, sigma, kernel_factor=5, device=torch.device("cpu")):
    # Kernel size dynamically computed from sigma (via kernel_factor)
    input = input
    kernel_size = int(kernel_factor * sigma) // 2 * 2 + 1
    channels = input.size(1)
    assert channels == 1

    grid_1d = torch.arange(kernel_size)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = torch.exp(
        -((grid_1d - mean) ** 2.) / (2 * variance)
    )
    # Make sure sum of values in gaussian kernel equals 1.
    kernel_1d = gaussian_kernel / torch.sum(gaussian_kernel)

    pad_value = (kernel_size - 1) // 2
    # Reshape to 2d depthwise convolutional weight

    #################################
    # Filter along multiple dims ####
    #################################

    ### 1. Dimension ###
    gaussian_kernel = kernel_1d.reshape(1, 1, kernel_size, 1, 1).to(device)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)
    filtered_image = F.conv3d(input, weight=gaussian_kernel, bias=None,
                              groups=channels, padding=(pad_value, 0, 0))

    ### 2. Dimension ###
    gaussian_kernel = kernel_1d.reshape(1, 1, 1, kernel_size, 1).to(device)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)
    filtered_image = F.conv3d(filtered_image, weight=gaussian_kernel, bias=None,
                              groups=channels, padding=(0, pad_value, 0))

    ### 3. Dimension ###   1
    gaussian_kernel = kernel_1d.reshape(1, 1, 1, 1, kernel_size).to(device)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)
    filtered_image = F.conv3d(filtered_image, weight=gaussian_kernel, bias=None,
                              groups=channels, padding=(0, 0, pad_value))

    return filtered_image


def elastic_deform_robin(image, target=None, alpha=40.0, sigma=35.0,
                                   device=torch.device("cpu")):
    """
    Elastic deformation of images as described in
    Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual
    Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    Very similar to:
    https://theaisummer.com/medical-image-processing

    Image is assumed to be of axes-dims [bsz, 1, H, W, D]
    """

    image_shape = image.shape
    spatial_dims = len(image.shape) - 2
    coords = tuple(
        (torch.arange(image_shape[i + 2]).to(device) / image_shape[i + 2]) * 2 - 1 for i in range(spatial_dims)
    )

    # image_interpolator = interpolate.RegularGridInterpolator(
    #    coords, image, method="linear",
    #    bounds_error=False, fill_value=background_val
    # )

    delta_idx = [
        filter_gaussian_3d((torch.rand(*image_shape).to(device) * 2.0 - 1.0), device=device, sigma=sigma)[:, 0] * alpha
        for i in range(spatial_dims)
    ]
    idx = torch.meshgrid(*coords, indexing="ij")
    idx = torch.stack(idx, dim=-1)[None]
    delta_idx = torch.stack(delta_idx, dim=-1)
    idx = idx + delta_idx
    idx[:, :, :, :, 0], idx[:, :, :, :, 2] = idx[:, :, :, :, 2], -idx[:, :, :, :, 0]
    # idx[:,:,:,:,1] = - idx[:,:,:,:,1]

    image = torch.nn.functional.grid_sample(image, idx, mode="nearest", padding_mode="border")
    if target is not None:
        target = torch.nn.functional.grid_sample(target, idx, mode="bilinear", padding_mode="border")
        target = torch.flip(target, dims=(2,))
        target = torch.heaviside(target - 0.1, torch.zeros_like(target))
    # image = image_interpolator(idx).reshape(image_shape)

    # TODO: can we make sure that we only use float32? Should increase performance a bit
    image = torch.flip(image, dims=(2,))
    return image, target


class Normalize3D(object):
    #supervised: mean = 0.22086217807487526 var = 0.09314631517275852 std = 0.30519881253497455
    #unsupervised mean = 0.30582652609574806 var = 0.10250503547737719 std = 0.32016407586950973
    def __init__(self, mean, std):
        super(Normalize3D, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, sample, target=None):
        sample = (sample - self.mean) / self.std
        return sample, target





class Rotate3D(object):
    def __init__(self, angle_min, angle_max):
        super(Rotate3D, self).__init__()
        self.angle_min = angle_min
        self.angle_max = angle_max

    def __call__(self, sample, target):
        degrees = np.random.uniform(self.angle_min, self.angle_max)
        sample = ndimage.rotate(sample, degrees, reshape=False)
        if target is not None:
            target = ndimage.rotate(target, degrees, reshape=False, order=0)
        return sample, target


class AddNoise(object):
    "Adds Gaussian noise"

    def __init__(
        self,
        mean_range,
        std_range,
        weight = 0.5,
        masked=True,
    ):
        self.mean_range = mean_range
        self.std_range = std_range
        self.weight = weight
        self.masked = masked

    def __call__(self, sample, target):
        mean = np.random.uniform(*self.mean_range)
        std = np.random.uniform(*self.std_range)

        # it doesn't matter if noise is transposed or not
        # but if voxels is transposed we'll get a nice speedup if noise is transposed too
        if sample.flags['C_CONTIGUOUS']:
            noise = np.random.normal(mean, std, sample.shape).astype(sample.dtype)
        else:
            noise = (
                np.random.normal(mean, std, sample.shape[::-1]).astype(sample.dtype).T
            )

        with_noise = ((1 - self.weight) * sample) + (self.weight * noise)
        if self.masked:
            vmin = sample.min()
            sample = np.where(sample == vmin, vmin, with_noise)
        else:
            sample = with_noise
        return sample, target


class Flip(object):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, sample, target):
        sample = np.flip(sample, self.axis)
        if target is not None:
            target = np.flip(target, self.axis)
        return sample, target


class GaussianFilter(object):
    def __init__(self, sigma_range):
        self.sigma_range = sigma_range

    def __call__(self, sample, target):
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        # TODO: allow for more configuration?
        sample = ndimage.gaussian_filter(sample, sigma)
        return sample, target


class Crop(object):
    def __init__(self, outsize, insize):
        self.size = outsize
        self.insize = insize

    def __call__(self, sample, target):
        v_shape = sample.shape

        anchor = np.random.randint(
            (0, 0, 0),
            (
                self.insize[0] - self.size[0] + 1,
                self.insize[1] - self.size[1] + 1,
                self.insize[2] - self.size[2] + 1,
            ),
        )

        sample = sample[
            anchor[0] : anchor[0] + self.size[0],
            anchor[1] : anchor[1] + self.size[1],
            anchor[2] : anchor[2] + self.size[2],
        ]

        if target is not None:
            target = target[
                     anchor[0]: anchor[0] + self.size[0],
                     anchor[1]: anchor[1] + self.size[1],
                     anchor[2]: anchor[2] + self.size[2],
                     ]

        return sample, target


class Rand(object):
    def __init__(self, transform, alt_transform=None, prob=0.5):
        assert prob >= 0 and prob <= 1
        self.prob = prob
        self.transform = transform
        self.alt_transform = alt_transform

    def __call__(self, sample, target):
        if random.random() < self.prob:
            sample, target = self.transform(sample, target)
        elif self.alt_transform is not None:
            sample, target = self.alt_transform(sample, target)
        return sample, target


class Load(object):
    def __init__(self, size, img_path=pretraining_DATA_PATH, seg_path=None):
        self.img_path = img_path
        self.seg_path = seg_path if seg_path is not None else img_path
        self.size = size
        self.load_compressed = lambda file_path: np.load(file_path)['arr_0']
        self.load_img = self.load_compressed if img_path.find('compressed') != -1 else lambda file_path: np.load(file_path)
        self.load_seg = self.load_compressed if self.seg_path.find('compressed') != -1 else lambda file_path: np.load(file_path)

    def __call__(self, img_name, seg_name):
        assert self.size in (224, 256)
        # compressed_img = '_compressed' if self.img_path.find('compressed') != -1 else ''
        # compressed_seg = '_compressed' if self.seg_path.find('compressed') != -1 else ''
        img_name = os.path.join(self.img_path, img_name)
        seg_name = os.path.join(self.seg_path, seg_name) if seg_name is not None else None
        compressed_img = '_compressed' if img_name.find('compressed') != -1 else ''
        compressed_seg = '_compressed' if seg_name is not None and seg_name.find('compressed') != -1 else ''
        img_path, seg_path = [], []
        index = -1
        for i, part in enumerate(img_name.split('/')):
            img_path.append(part)
            if part.find('resized') != -1:
                index = i
        img_path[index] = 'resized' + str(self.size) + compressed_img
        if seg_name is not None:
            index = -1
            for i, part in enumerate(seg_name.split('/')):
                seg_path.append(part)
                if part.find('resized') != -1:
                    index = i
            seg_path[index] = 'resized' + str(self.size) + compressed_seg
        img_path = '/' + os.path.join(*img_path)
        seg_path = '/' + os.path.join(*seg_path) if len(seg_path) > 0 else None

        img = self.load_img(img_path).astype(np.float32)
        label = self.load_seg(seg_path).astype(np.float32) if seg_path is not None else None
        return img, label


class Compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, sample, target=None):
        for t in self.transform_list:
            sample, target = t(sample, target)
        return sample, target


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, sample, target=None):
        return sample, target

class Zoom(object):
    def __init__(self, scale=(224, 224, 224)):
        self.scale = scale

    def __call__(self, sample, target):
        if sample.shape != self.scale:
            scale_factors = (
                self.scale[0] / sample.shape[0],
                self.scale[1] / sample.shape[1],
                self.scale[2] / sample.shape[2],
            )
            if sample.flags['C_CONTIGUOUS']:
                sample = ndimage.zoom(sample.T, scale_factors[::-1]).T
                if target is not None:
                    target = ndimage.zoom(target.T, scale_factors[::-1], order=0).T
            else:
                sample = ndimage.zoom(sample, scale_factors)
                if target is not None:
                    target = ndimage.zoom(target, scale_factors, order=0)
        return sample, target


def get_transform(imsize, img_path, seg_path, normalize=None):
    zoomsize = 224 if imsize == '256' else 112
    insize = 256 if imsize == '256' else 128
    transform = Compose([
        #Rand(Compose([Zoom((insize, insize, insize//2)), Crop((zoomsize, zoomsize, zoomsize//2), (insize, insize, insize//2))]), Zoom((zoomsize, zoomsize, zoomsize//2)), prob=0.5),
        Rand(Compose([Load(insize, img_path=img_path, seg_path=seg_path), Crop((zoomsize, zoomsize, zoomsize), (insize, insize, insize))]), Load(zoomsize, img_path=img_path, seg_path=seg_path), prob=0.5),
        normalize if normalize is not None else Identity(),
        Rand(Flip(0), prob=0.25),
        Rand(Flip(1), prob=0.25),
        Rand(Flip(2), prob=0.25),
        Rand(Rotate3D(-30, 30), prob=0.2),
        Rand(GaussianFilter([0.6, 0.8]), prob=0.2),
        Rand(AddNoise([0.5, 0.5], [0.3, 1], 0.03, True), prob=0.2)
    ])
    return transform