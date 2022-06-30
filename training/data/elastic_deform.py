import numpy as np
from scipy import interpolate, ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F


def elastic_deform_single_color(image: np.ndarray, alpha=40.0, sigma=35.0, background_val=0.0):
    """
    Elastic deformation of images as described in
    Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual
    Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    Very similar to:
    https://theaisummer.com/medical-image-processing

    All dimension will take place in the distortion.
    """

    image_shape = image.shape
    number_dims = image.ndim
    coords = tuple(np.arange(image_shape[i]) for i in range(number_dims))

    image_interpolator = interpolate.RegularGridInterpolator(
        coords, image, method="linear",
        bounds_error=False, fill_value=background_val
    )

    delta_idx = [
        ndimage.gaussian_filter((np.random.rand(*tuple(image_shape)) * 2.0 - 1.0), sigma,
                                mode="constant", cval=0.0) * alpha
        for i in range(len(image_shape))
    ]
    idx = np.meshgrid(*coords, indexing="ij")
    idx = tuple(np.reshape(idx[i] + delta_idx[i], (-1,1)) for i in range(number_dims))

    image = image_interpolator(idx).reshape(image_shape)

    # TODO: can we make sure that we only use float32? Should increase performance a bit
    return image


"""
Pytorch elastic deform (drastically faster)
"""


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


def elastic_deform_single_color_3d(image: torch.Tensor, alpha=40.0, sigma=35.0, background_val=0.0,
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
    # image = image_interpolator(idx).reshape(image_shape)

    # TODO: can we make sure that we only use float32? Should increase performance a bit
    image = torch.flip(image, dims=(2,))
    return image
