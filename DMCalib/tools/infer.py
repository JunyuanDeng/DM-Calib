import torch
import numpy as np
from typing import Tuple
from tools.tools import coords_gridN, resample_rgb, apply_augmentation, apply_augmentation_centre, kitti_benchmark_crop, _paddings, _preprocess, _shapes
from skimage.measure import ransac, LineModelND
import torch.nn.functional as F


def spherical_zbuffer_to_euclidean(spherical_tensor):
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract zbuffer depth

    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    # =>
    # r = z / sin(phi) / cos(theta)
    # y = z / (sin(phi) / cos(phi)) / cos(theta)
    # x = z * sin(theta) / cos(theta)
    x = z * np.tan(theta)
    y = z / np.tan(phi) / np.cos(theta)

    euclidean_tensor = np.stack((x, y, z), axis=-1)
    return euclidean_tensor

def generate_rays(
    
    camera_intrinsics: torch.Tensor, image_shape: Tuple[int, int], noisy: bool = False
):
    
    phi_min = 0.21
    phi_max = 0.79
    theta_min = -0.31
    theta_max = 0.31
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Calculate ray directions
    intrinsics_inv = torch.inverse(camera_intrinsics.float()).to(dtype)  # (B, 3, 3)
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )  # (3, H*W)
    ray_directions = F.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)

    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    # pitch = torch.asin(ray_directions[..., 1])
    # roll = torch.atan2(ray_directions[..., 0], - ray_directions[..., 1])
    angles_origin = torch.stack([theta, phi], dim=-1).reshape(*image_shape, 2).permute(2, 0, 1)
    
    theta = theta / torch.pi
    phi = phi / torch.pi
    theta = ( ((theta - theta_min) / (theta_max - theta_min))  - 0.5) * 2
    phi = ( ((phi - phi_min) / (phi_max - phi_min)) - 0.5) * 2
    # by default, the batchsize here is one
    angles = torch.stack([theta, phi], dim=-1).reshape(*image_shape, 2).permute(2, 0, 1)
    angles.clip_(-1.0, 1.0)    

    return ray_directions, angles_origin, angles

def preprocess_pad(rgb, target_shape):
    _, h, w = rgb.shape
    (ht, wt), ratio = _shapes((h, w), target_shape)
    pad_left, pad_right, pad_top, pad_bottom = _paddings((ht, wt), target_shape)
    rgb, K = _preprocess(
        rgb,
        None,
        (ht, wt),
        (pad_left, pad_right, pad_top, pad_bottom),
        ratio,
        target_shape,
    )
    return rgb, pad_left, pad_right, pad_top, pad_bottom

def calculate_intrinsic(pred_image, pad=None, mask=None):
    gsv_phi_min = 0.21
    gsv_phi_max = 0.79
    gsv_theta_min = -0.31
    gsv_theta_max = 0.31


    _, h, w = pred_image.shape
    ori_image = (pred_image).clone()
    if pad != None:
        pad_left, pad_right, pad_top, pad_bottom = pad
        ori_image = ori_image[:, pad_top:h-pad_bottom, pad_left:w-pad_right]
    ori_image[0] = ori_image[0] * (gsv_theta_max-gsv_theta_min) + gsv_theta_min
    ori_image[1] = ori_image[1] * (gsv_phi_max-gsv_phi_min) + gsv_phi_min
    if mask != None:
        x = np.tan(ori_image[0, mask > 0.5].reshape(-1).numpy()*np.pi)
        y = np.tile(np.arange(0, ori_image.shape[2]), ori_image.shape[1]).reshape(h, w)[mask > 0.5]
    else:
        x = np.tan(ori_image[0].reshape(-1).numpy()*np.pi)
        y = np.tile(np.arange(0, ori_image.shape[2]), ori_image.shape[1])
    data = np.column_stack([x, y])

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(
        data, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000
    )
    cx = model_robust.predict_y([0])[0]
    fx = (model_robust.params[1][1]/model_robust.params[1][0])

    if mask != None:
        x = 1/ np.tan(ori_image[1, mask > 0.5].reshape(-1).numpy() * np.pi) / np.cos(ori_image[0, mask > 0.5].reshape(-1).numpy()*np.pi)
        y = (np.arange(0, ori_image.shape[1]).repeat(ori_image.shape[2]).reshape(h, w))[mask > 0.5]
    else:
        x = 1/ np.tan(ori_image[1].reshape(-1).numpy() * np.pi) / np.cos(ori_image[0].reshape(-1).numpy()*np.pi)
        y = np.arange(0, ori_image.shape[1]).repeat(ori_image.shape[2])
    data = np.column_stack([x, y])

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(
        data, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000
    )
    cy = model_robust.predict_y([0])[0]
    fy = (model_robust.params[1][1]/model_robust.params[1][0])
    return [fx, fy, cx, cy]