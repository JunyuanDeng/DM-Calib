import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.utils.data import Sampler
from torchvision.transforms import InterpolationMode, Resize
from math import ceil
import random


# -- # Common Functions
class InputPadder:
    """ Pads images such that dimensions are divisible by ds """
    def __init__(self, dims, mode='leftend', ds=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // ds) + 1) * ds - self.ht) % ds
        pad_wd = (((self.wd // ds) + 1) * ds - self.wd) % ds
        if mode == 'leftend':
            self._pad = [0, pad_wd, 0, pad_ht]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

        self.mode = mode

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def _paddings(image_shape, network_shape):
    cur_h, cur_w = image_shape
    h, w = network_shape
    pad_top, pad_bottom = (h - cur_h) // 2, h - cur_h - (h - cur_h) // 2
    pad_left, pad_right = (w - cur_w) // 2, w - cur_w - (w - cur_w) // 2
    return pad_left, pad_right, pad_top, pad_bottom


def _shapes(image_shape, network_shape):
    h, w = image_shape
    input_ratio = w / h
    output_ratio = network_shape[1] / network_shape[0]
    if output_ratio > input_ratio:
        ratio = network_shape[0] / h
    elif output_ratio <= input_ratio:
        ratio = network_shape[1] / w
    return (ceil(h * ratio - 0.5), ceil(w * ratio - 0.5)), ratio


def _preprocess(rgbs, intrinsics, shapes, pads, ratio, output_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    rgbs = F.interpolate(
        rgbs.unsqueeze(0), size=shapes, mode="bilinear", align_corners=False, antialias=True
    )
    rgbs = F.pad(rgbs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant")
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[0, 0] = intrinsics[0, 0] * ratio
        intrinsics[1, 1] = intrinsics[1, 1] * ratio
        intrinsics[0, 2] = intrinsics[0, 2] * ratio #+ pad_left
        intrinsics[1, 2] = intrinsics[1, 2] * ratio #+ pad_top
        return rgbs.squeeze(), intrinsics
    return rgbs.squeeze(), None


def coords_gridN(batch, ht, wd, device):
    coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / ht, 1 - 1 / ht, ht, device=device),
            torch.linspace(-1 + 1 / wd, 1 - 1 / wd, wd, device=device),
        ),
        indexing = 'ij'
    )

    coords = torch.stack((coords[1], coords[0]), dim=0)[
        None
    ].repeat(batch, 1, 1, 1)
    return coords

def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch

def rename_ckpt(ckpt):
    renamed_ckpt = dict()
    for k in ckpt.keys():
        if 'module.' in k:
            renamed_ckpt[k.replace('module.', '')] = torch.clone(ckpt[k])
        else:
            renamed_ckpt[k] = torch.clone(ckpt[k])
    return renamed_ckpt

def resample_rgb(rgb, scaleM, batch, ht, wd, device):
    coords = coords_gridN(batch, ht, wd, device)
    x, y = torch.split(coords, 1, dim=1)
    x = (x + 1) / 2 * wd
    y = (y + 1) / 2 * ht

    scaleM = scaleM.squeeze()

    x = x * scaleM[0, 0] + scaleM[0, 2]
    y = y * scaleM[1, 1] + scaleM[1, 2]

    _, _, orgh, orgw = rgb.shape
    x = x / orgw * 2 - 1.0
    y = y / orgh * 2 - 1.0

    coords = torch.stack([x.squeeze(1), y.squeeze(1)], dim=3)
    rgb_resized = torch.nn.functional.grid_sample(rgb, coords, mode='bilinear', align_corners=True)

    return rgb_resized

def intrinsic2incidence(K, b, h, w, device):
    coords = coords_gridN(b, h, w, device)

    x, y = torch.split(coords, 1, dim=1)
    x = (x + 1) / 2.0 * w
    y = (y + 1) / 2.0 * h

    pts3d = torch.cat([x, y, torch.ones_like(x)], dim=1)
    pts3d = rearrange(pts3d, 'b d h w -> b h w d')
    pts3d = pts3d.unsqueeze(dim=4)

    K_ex = K.view([b, 1, 1, 3, 3])
    pts3d = torch.linalg.inv(K_ex) @ pts3d
    pts3d = torch.nn.functional.normalize(pts3d, dim=3)
    return pts3d

def apply_augmentation(rgb, K, seed=None, augscale=2.0, no_change_prob=0.0, retain_aspect=False):
    _, h, w = rgb.shape

    if seed is not None:
        np.random.seed(seed)

    if np.random.uniform(0, 1) < no_change_prob:
        extension_rx, extension_ry = 1.0, 1.0
    else:
        if not retain_aspect:
            extension_rx, extension_ry = np.random.uniform(1, augscale), np.random.uniform(1, augscale)
        else:
            extension_rx =  extension_ry = np.random.uniform(1, augscale)

    hs, ws = int(np.ceil(h * extension_ry)), int(np.ceil(w * extension_rx))

    stx = float(np.random.randint(0, int(ws - w + 1), 1).item() + 0.5)
    edx = float(stx + w - 1)
    sty = float(np.random.randint(0, int(hs - h + 1), 1).item() + 0.5)
    edy = float(sty + h - 1)

    stx = stx / ws * w
    edx = edx / ws * w

    sty = sty / hs * h
    edy = edy / hs * h

    ptslt, ptslt_ = np.array([stx, sty, 1]), np.array([0.5, 0.5, 1])
    ptsrt, ptsrt_ = np.array([edx, sty, 1]), np.array([w-0.5, 0.5, 1])
    ptslb, ptslb_ = np.array([stx, edy, 1]), np.array([0.5, h-0.5, 1])
    ptsrb, ptsrb_ = np.array([edx, edy, 1]), np.array([w-0.5, h-0.5, 1])

    pts1 = np.stack([ptslt, ptsrt, ptslb, ptsrb], axis=1)
    pts2 = np.stack([ptslt_, ptsrt_, ptslb_, ptsrb_], axis=1)

    T_num = pts1 @ pts2.T @ np.linalg.inv(pts2 @ pts2.T)
    T = np.eye(3)
    T[0, 0] = T_num[0, 0]
    T[0, 2] = T_num[0, 2]
    T[1, 1] = T_num[1, 1]
    T[1, 2] = T_num[1, 2]
    T = torch.from_numpy(T).float()

    K_trans = torch.inverse(T) @ K

    b = 1
    _, h, w = rgb.shape
    device = rgb.device
    rgb_trans = resample_rgb(rgb.unsqueeze(0), T, b, h, w, device).squeeze(0)
    return rgb_trans, K_trans, T

def kitti_benchmark_crop(input_img, h=None, w=None):
    """
    Crop images to KITTI benchmark size
    Args:
        `input_img` (torch.Tensor): Input image to be cropped.

    Returns:
        torch.Tensor:Cropped image.
    """
    KB_CROP_HEIGHT = h if h != None else 342
    KB_CROP_WIDTH = w if w != None else 1216

    height, width = input_img.shape[-2:]
    top_margin = int(height - KB_CROP_HEIGHT)
    left_margin = int((width - KB_CROP_WIDTH) / 2)
    if 2 == len(input_img.shape):
        out = input_img[
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    return out


def apply_augmentation_centre(rgb, K, seed=None, augscale=2.0, no_change_prob=0.0):
    _, h, w = rgb.shape

    if seed is not None:
        np.random.seed(seed)

    if np.random.uniform(0, 1) < no_change_prob:
        extension_r = 1.0
    else:
        extension_r = np.random.uniform(1, augscale)

    hs, ws = int(np.ceil(h * extension_r)), int(np.ceil(w * extension_r))
    centre_h, centre_w = hs//2, ws//2

    rgb_large = Resize(
                size=(hs, ws), interpolation=InterpolationMode.BILINEAR, antialias=True
            )(rgb)

    rgb_trans = rgb_large[:, centre_h-h//2: centre_h-h//2+h,  centre_w-w//2:centre_w-w//2+w]
    _, ht, wt = rgb_trans.shape

    assert ht == h and wt == w 

    K_trans = K.clone()
    K_trans[0, 0] = K_trans[0, 0] * extension_r
    K_trans[1, 1] = K_trans[1, 1] * extension_r


    return rgb_trans, K_trans


def apply_augmentation_centrecrop(rgb, K, seed=None, augscale=2.0, no_change_prob=0.0):
    c, h, w = rgb.shape

    if seed is not None:
        np.random.seed(seed)

    if np.random.uniform(0, 1) < no_change_prob:
        extension_r = 1.0
    else:
        extension_r = np.random.uniform(1, augscale)

    hs, ws = int(np.ceil(h / extension_r)), int(np.ceil(w / extension_r))
    centre_h, centre_w = h//2, w//2

    rgb_trans = rgb[:, centre_h-hs//2: centre_h-hs//2+hs,  centre_w-ws//2:centre_w-ws//2+ws]
    _, ht, wt = rgb_trans.shape


    K_trans = K.clone()
    K_trans[0, 2] = K_trans[0, 2] / extension_r
    K_trans[1, 2] = K_trans[1, 2] / extension_r


    return rgb_trans, K_trans

def kitti_benchmark_crop_dpx(input_img, K=None):
    '''
    input size: 324*768 for dpx
    output size: 216*768
    '''

    KB_CROP_HEIGHT = 216

    height, width = input_img.shape[-2:]
    botton_margin = np.random.randint(1, 25)

    if 2 == len(input_img.shape):
        out = input_img[
            height-botton_margin-KB_CROP_HEIGHT : -botton_margin,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            height-botton_margin-KB_CROP_HEIGHT : -botton_margin,
        ]
    if K != None:
        K_trans = K.clone()
        K_trans[1, 2] = K_trans[1, 2] - (324-216)/2
        return out, K_trans
    return out

def kitti_benchmark_crop_dpx_nofront(input_img, K=None):
    '''
    input size: 512*768 for dpx
    output size: 320*768
    '''

    KB_CROP_HEIGHT = 320

    height, width = input_img.shape[-2:]
    botton_margin = np.random.randint(1, 80)

    if 2 == len(input_img.shape):
        out = input_img[
            height-botton_margin-KB_CROP_HEIGHT : -botton_margin,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            height-botton_margin-KB_CROP_HEIGHT : -botton_margin,
        ]
    if K != None:
        K_trans = K.clone()
        K_trans[1, 2] = K_trans[1, 2] - (512-320)/2
        return out, K_trans
    return out

def kitti_benchmark_crop_dpx_front(input_img, K=None):
    '''
    input size: 324*768 for dpx
    output size: 320*768
    '''

    KB_CROP_HEIGHT = 320

    height, width = input_img.shape[-2:]
    botton_margin = np.random.randint(1, 4)

    if 2 == len(input_img.shape):
        out = input_img[
            height-botton_margin-KB_CROP_HEIGHT : -botton_margin,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            height-botton_margin-KB_CROP_HEIGHT : -botton_margin,
        ]
    if K != None:
        K_trans = K.clone()
        K_trans[1, 2] = K_trans[1, 2] - (324-320)/2
        return out, K_trans
    return out

def kitti_benchmark_crop_waymo(input_img, K=None):

    KB_CROP_HEIGHT = 800

    height, width = input_img.shape[-2:]
    botton_margin = np.random.randint(1, 80)

    if 2 == len(input_img.shape):
        out = input_img[
            height-botton_margin-KB_CROP_HEIGHT : -botton_margin,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            height-botton_margin-KB_CROP_HEIGHT : -botton_margin,
        ]
    if K != None:
        K_trans = K.clone()
        K_trans[1, 2] = K_trans[1, 2] - (height-KB_CROP_HEIGHT)/2
        return out, K_trans
    return out

def kitti_benchmark_crop_argo2(input_img, K=None):

    'in :2048*1550'



    height, width = input_img.shape[-2:]
    KB_CROP_HEIGHT = width
    random_shift = np.random.randint(-50, 50)
    top_maigin = int((height - KB_CROP_HEIGHT) / 2) + random_shift

    if 2 == len(input_img.shape):
        out = input_img[
            top_maigin : top_maigin+KB_CROP_HEIGHT,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_maigin : top_maigin+KB_CROP_HEIGHT,
        ]
    if K != None:
        K_trans = K.clone()
        K_trans[1, 2] = K_trans[1, 2] - (height-KB_CROP_HEIGHT)/2
        return out, K_trans
    return out

def kitti_benchmark_crop_argo2_sideview(input_img, K=None):

    'in :1550*2048'
    height, width = input_img.shape[-2:]
    KB_CROP_WIDTH = height
    random_shift = np.random.randint(-50, 50)
    left_margin = int((width - KB_CROP_WIDTH) / 2) + random_shift

    if 2 == len(input_img.shape):
        out = input_img[
            :,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            :,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    if K != None:
        K_trans = K.clone()
        K_trans[0, 2] = K_trans[0, 2] - (width-KB_CROP_WIDTH)/2
        return out, K_trans
    return out

def kitti_benchmark_crop_simu2(input_img, K=None):
    '''
    input size: 432*768 for  simu
    output size: 320*768
    '''

    KB_CROP_HEIGHT = 320

    height, width = input_img.shape[-2:]
    random_shift = np.random.randint(-25, 25)

    top_maigin = int((height - KB_CROP_HEIGHT) / 2) + random_shift

    if 2 == len(input_img.shape):
        out = input_img[
            top_maigin : top_maigin+KB_CROP_HEIGHT,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_maigin : top_maigin+KB_CROP_HEIGHT,
        ]
    if K != None:
        K_trans = K.clone()
        K_trans[1, 2] = K_trans[1, 2] - (432-320)/2
        return out, K_trans
    return out

def kitti_benchmark_crop_simu(input_img, K=None):
    '''
    input size: 512*768 for  simu
    output size: 320*768
    '''

    KB_CROP_HEIGHT = 320

    height, width = input_img.shape[-2:]
    random_shift = np.random.randint(-50, 50)

    top_maigin = int((height - KB_CROP_HEIGHT) / 2) + random_shift

    if 2 == len(input_img.shape):
        out = input_img[
            top_maigin : top_maigin+KB_CROP_HEIGHT,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_maigin : top_maigin+KB_CROP_HEIGHT,
        ]
    if K != None:
        K_trans = K.clone()
        K_trans[1, 2] = K_trans[1, 2] - (512-320)/2
        return out, K_trans
    return out

def resize_sparse_depth(sparse_depth, target_size):  
    """  
    Resize a sparse depth image while preserving the number of non-zero depth values.  
    If multiple non-zero values map to the same target coordinate, keep the minimum value.  

    Parameters:  
    sparse_depth (np.ndarray): The original sparse depth image.  
    target_size (tuple): The target size of the resized depth image, in the format (width, height).  

    Returns:  
    np.ndarray: The resized sparse depth image with the same number of non-zero depth values.  
    """  
    # 识别非零像素的位置和值  
    non_zero_indices = torch.argwhere(sparse_depth != 0)  
    non_zero_values = sparse_depth[non_zero_indices[:, 0], non_zero_indices[:, 1]]  

    # 计算缩放比例  
    scale_x = target_size[0] / sparse_depth.shape[1]  
    scale_y = target_size[1] / sparse_depth.shape[0]  

    # 创建一个字典来跟踪每个新坐标的最小值  
    min_values_map = {}  

    # 重新映射非零像素的位置  
    for idx, (y, x) in enumerate(non_zero_indices):  
        new_x = int(x * scale_x)  
        new_y = int(y * scale_y)  

        # 确保新的坐标在目标图像范围内  
        new_x = max(0, min(new_x, target_size[0] - 1))  
        new_y = max(0, min(new_y, target_size[1] - 1))  

        # 使用新坐标作为键，如果键不存在或当前值小于字典中的值，则更新字典  
        key = (new_y, new_x)  
        if key not in min_values_map or non_zero_values[idx] < min_values_map[key]:  
            min_values_map[key] = non_zero_values[idx]  

    # 创建一个新的深度图像，并将非零值（即最小值）放置在新位置  
    resized_depth = torch.zeros((target_size[1], target_size[0]), dtype=sparse_depth.dtype)  
    for (y, x), value in min_values_map.items():  
        resized_depth[y, x] = value  

    # 返回重新大小的稀疏深度图像  
    return resized_depth  



def random_crop_arr_v2(torch_image, torch_depth, K, sparse_depth=False, image_size=(768, 768), min_scale=1.0, max_scale=1.2):
    # 确保输入是一个3D张量 (C, H, W)
    if torch_image.dim() != 3 or torch_depth.dim() != 3:
        raise ValueError("torch_image and torch_depth must both be 3D (C, H, W)")

    # torch_image需要clip
    
    # 检查 image 和 depth 分辨率一致
    assert torch_image.shape == torch_depth.shape, "torch_image and torch_depth must have the same dimensions"

    # 获取图像的原始高度和宽度
    _, h_origin, w_origin = torch_image.shape
    h_target, w_target = image_size

    # 先考虑目标是一个正方形
    assert h_target == w_target

    # 先让最短边，能达到框框的大小
    if h_origin > w_origin:
        base_scale = w_target/w_origin
    else:
        base_scale = h_target/h_origin


    # 计算放大倍数，确保缩放后尺寸达到1.0到1.2倍的框框大小
    scale_min = base_scale * min_scale
    scale_max = base_scale * max_scale
    resize_ratio = random.uniform(scale_min, scale_max)

    # 根据计算的缩放比例调整图像尺寸，同时保持长宽比
    h_scaled, w_scaled = ceil(h_origin * resize_ratio), ceil(w_origin * resize_ratio)

    # 初始化内参矩阵的副本，避免直接修改原始内参
    K_adj = K.clone()
    K_adj[0, 0] *= resize_ratio  # 调整 fx
    K_adj[1, 1] *= resize_ratio  # 调整 fy
    K_adj[0, 2] *= resize_ratio  # 调整 cx
    K_adj[1, 2] *= resize_ratio  # 调整 cy

    # 将图像和深度图按比例缩放到新的尺寸 (h_scaled, w_scaled)
    scaled_image = F.interpolate(torch_image.unsqueeze(0), size=(h_scaled, w_scaled), mode='bilinear', align_corners=False)
    if sparse_depth:
        scaled_depth = resize_sparse_depth(torch_depth[0], (w_scaled, h_scaled )).repeat(3, 1, 1).unsqueeze(0)
    else:
        scaled_depth = F.interpolate(torch_depth.unsqueeze(0), size=(h_scaled, w_scaled), mode='nearest')

    # 在放大后的图像中随机裁剪出目标框大小的区域
    crop_y = random.randint(0, h_scaled - h_target)
    crop_x = random.randint(0, w_scaled - w_target)
    crop_image = scaled_image[:, :, crop_y:crop_y + h_target, crop_x:crop_x + w_target]
    crop_depth = scaled_depth[:, :, crop_y:crop_y + h_target, crop_x:crop_x + w_target]

    # 更新内参矩阵中的 cx 和 cy
    K_adj[0, 2] -= (w_scaled-w_target)/2
    K_adj[1, 2] -= (h_scaled-h_target)/2

    # 去除 batch 维度并返回
    return crop_image.squeeze(0), crop_depth.squeeze(0), K_adj  # 返回 (C, H, W), (C, H, W) 和调整后的 K



def random_zero_replace(image, depth, camera_image, mask=None, padding_max_size=30):
    # 确保输入是三维张量 (C, H, W) 并且 image 和 depth 尺寸一致
    assert image.dim() == 3 and depth.dim() == 3, "Both image and depth must be 3D tensors (C, H, W)"
    assert image.shape == depth.shape, "image and depth must have the same dimensions"

    # 随机选择置零的方向：0表示上下，1表示左右
    direction = random.choice([0, 1])
    
    # 随机生成置零的大小，范围在 0 到 padding_max_size 之间
    zero_size = random.randint(0, padding_max_size)
    
    _, h, w = image.shape
    
    if direction == 0:
        # 上下置零
        image[:, :zero_size, :] = 0   # 上部置零
        image[:, h - zero_size:, :] = 0  # 下部置零
        depth[:, :zero_size, :] = 0
        depth[:, h - zero_size:, :] = 0
        camera_image[:, :zero_size, :] = -1
        camera_image[:, h - zero_size:, :] = -1
        if mask != None:
            mask[:, :zero_size, :] = True
            mask[:, h - zero_size:, :] = True
    else:
        # 左右置零
        image[:, :, :zero_size] = 0   # 左侧置零
        image[:, :, w - zero_size:] = 0  # 右侧置零
        depth[:, :, :zero_size] = 0
        depth[:, :, w - zero_size:] = 0
        camera_image[:, :, :zero_size] = -1
        camera_image[:, :, w - zero_size:] = -1
        if mask != None:
            mask[:, :, :zero_size] = True
            mask[:, :, w - zero_size:] = True
    if mask != None:
        return image, depth, camera_image, (direction, zero_size), mask
    else:
        return image, depth, camera_image, (direction, zero_size)


    


class IncidenceLoss(nn.Module):
    def __init__(self, loss='cosine'):
        super(IncidenceLoss, self).__init__()
        self.loss = loss
        self.smoothl1 = torch.nn.SmoothL1Loss(beta=0.2)

    def forward(self, incidence, K):
        b, _, h, w = incidence.shape
        device = incidence.device

        incidence_gt = intrinsic2incidence(K, b, h, w, device)
        incidence_gt = incidence_gt.squeeze(4)
        incidence_gt = rearrange(incidence_gt, 'b h w d -> b d h w')

        if self.loss == 'cosine':
            loss = 1 - torch.cosine_similarity(incidence, incidence_gt, dim=1)
        elif self.loss == 'absolute':
            loss = self.smoothl1(incidence, incidence_gt)

        loss = loss.mean()
        return loss


class DistributedSamplerNoEvenlyDivisible(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        num_samples = int(math.floor(len(self.dataset) * 1.0 / self.num_replicas))
        rest = len(self.dataset) - num_samples * self.num_replicas
        if self.rank < rest:
            num_samples += 1
        self.num_samples = num_samples
        self.total_size = len(dataset)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch