from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, StableDiffusionImageVariationPipeline
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.measure import ransac, LineModelND
import math  
from torchvision.utils import save_image
import pymagsac
import torch
from diffusers import AutoencoderKL
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np
import torch
from einops import rearrange
from torchvision import transforms
from plyfile import PlyData, PlyElement
from typing import Tuple
import h5py
from tabulate import tabulate
import random
from copy import deepcopy
import time
import pickle
import json
import argparse
totensor = transforms.ToTensor()


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()

def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()

def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()

def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()

def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()

def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()

def delta05_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**0.5, valid_mask)

def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)

def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)

def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)

def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()

def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred

def kitti_benchmark_crop(input_img):
    """
    Crop images to KITTI benchmark size
    Args:
        `input_img` (torch.Tensor): Input image to be cropped.

    Returns:
        torch.Tensor:Cropped image.
    """
    KB_CROP_HEIGHT = 342
    KB_CROP_WIDTH = 1216

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


eval_metrics = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta05_acc",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
]

functions_dict = {
    "abs_relative_difference": abs_relative_difference,
    "squared_relative_difference": squared_relative_difference,
    "rmse_linear": rmse_linear,
    "rmse_log": rmse_log,
    "log10": log10,
    "delta05_acc": delta05_acc,
    "delta1_acc": delta1_acc,
    "delta2_acc": delta2_acc,
    "delta3_acc": delta3_acc,
    "i_rmse": i_rmse,
    "silog_rmse": silog_rmse,
}
metric_funcs = [functions_dict[name] for name in eval_metrics]
metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
metric_tracker.reset()

def read_test_files(txt_path):
    test_files = []
    with open(txt_path, 'r') as file:
        for line in file:
            first_string = line.split()
            if first_string[1] == "None":
                continue
            test_files.append(first_string)
    return test_files
    
'''Set the Args'''
parser = argparse.ArgumentParser(
    description="Run MonoDepthNormal Estimation using Stable Diffusion."
)
parser.add_argument(
    "--nyu",
    action="store_true", 
)    

parser.add_argument(
    "--diode_indoor",
    action="store_true", 
)    

parser.add_argument(
    "--diode_outdoor",
    action="store_true", 
)    
    
parser.add_argument(
    "--doide_outdoor",
    action="store_true", 
)  

parser.add_argument(
    "--sunrgbd",
    action="store_true", 
)     

parser.add_argument(
    "--ibims",
    action="store_true", 
)    

parser.add_argument(
    "--eth3d",
    action="store_true", 
)    

parser.add_argument(
    "--kitti",
    action="store_true", 
)    

parser.add_argument(
    "--nuscenes",
    action="store_true", 
)  

parser.add_argument(
    "--ddad",
    action="store_true", 
)  

parser.add_argument(
    "--void",
    action="store_true", 
)  

parser.add_argument(
    "--scannet",
    action='store_true',
)


parser.add_argument(
    "--input_depth_path",
    type=str,
)


parser.add_argument(
    "--relative",
    action="store_true", 
)

args = parser.parse_args()

if args.nyu:
    txt_path = '/home/users/junyuan.deng/Programmes/idisc/splits/nyu/nyu_test.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'nyu', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/datasets/nyu"
    
    scale = 1000.0
    gt_files = []
    mask_files = []
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    for index in tqdm(range(len(test_files))):
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.jpg', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))


        est_depth_list.append(torch.from_numpy(est_depth[None]))

        gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        gt_depth_list.append(gt_depth)
        
        # mask_depth = np.load(mask_files[index])
        mask_list.append(torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 10)
            ))

        
    est_depth_torch = torch.stack(est_depth_list)
    ####

    est_depth_torch = F.interpolate(est_depth_torch, (480, 640), mode='nearest')
    ####
    gt_depth_torch = torch.stack(gt_depth_list)
    mask_torch = torch.stack(mask_list)
    eval_mask = torch.zeros_like(mask_torch).bool()
    eval_mask[..., 45:471, 41:601] = 1
    mask_torch = torch.logical_and(mask_torch, eval_mask)

    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        if args.relative:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_depth_np[i],
                pred_arr=est_depth_np[i],
                valid_mask_arr=mask_np[i],
                return_scale_shift=True,
                max_resolution=None,
            )
            est_depth_np[i] = depth_pred
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=10
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":
                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------NYU-V2------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------NYU-V2------------------------")
    
if args.diode_indoor:
    txt_path = '/home/users/junyuan.deng/Programmes/idisc/splits/diode/diode_indoor_val.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'diode_indoor', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/temp_dir/diode"
    
    scale = 256.0
    gt_files = []
    mask_files = []
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    gt_rgbs = []
    for index in tqdm(range(len(test_files))):
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))

        gt_rgbs.append(Image.open(os.path.join(gt_depth_pathes, test_files[index][0])))

        est_depth_list.append(torch.from_numpy(est_depth[None]))

        gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        gt_depth_list.append(gt_depth)
        
        # mask_depth = np.load(mask_files[index])
        mask_list.append(torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 50)
            ))
    est_depth_torch = torch.stack(est_depth_list)
    est_depth_torch = F.interpolate(est_depth_torch, (768, 1024), mode='nearest')
    gt_depth_torch = torch.stack(gt_depth_list)
    mask_torch = torch.stack(mask_list)
    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        if args.relative:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_depth_np[i],
                pred_arr=est_depth_np[i],
                valid_mask_arr=mask_np[i],
                return_scale_shift=True,
                max_resolution=None,
            )
            est_depth_np[i] = depth_pred
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=50
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":

                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------diode_indoor------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------diode_indoor------------------------")
    
if args.diode_outdoor:
    txt_path = '/home/users/junyuan.deng/Programmes/Marigold/data_split/diode/diode_val_outdoor_filename_list.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'diode_outdoor', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/temp_dir/diode"
    
    scale = 256.0
    gt_files = []
    mask_files = []
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    gt_rgbs = []
    for index in tqdm(range(len(test_files))):
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))

        gt_rgbs.append(Image.open(os.path.join(gt_depth_pathes, test_files[index][0])))

        est_depth_list.append(torch.from_numpy(est_depth[None]))

        gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        gt_depth_list.append(gt_depth)
        
        # mask_depth = np.load(mask_files[index])
        mask_list.append(torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 150)
            ))
    est_depth_torch = torch.stack(est_depth_list)
    est_depth_torch = F.interpolate(est_depth_torch, (768, 1024), mode='nearest')
    gt_depth_torch = torch.stack(gt_depth_list)
    mask_torch = torch.stack(mask_list)
    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        if args.relative:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_depth_np[i],
                pred_arr=est_depth_np[i],
                valid_mask_arr=mask_np[i],
                return_scale_shift=True,
                max_resolution=None,
            )
            est_depth_np[i] = depth_pred
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=150
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":

                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------diode_outdoor------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------diode_outdoor------------------------")    

if args.ibims:
    txt_path = '/horizon-bucket/saturn_v_dev/users/junyuan.deng/Programmes/intrinsic/splits/ibims_test_full.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'ibims', 'depth_npy')
    gt_depth_pathes = "/horizon-bucket/saturn_v_dev/users/junyuan.deng/datasets_val/ibims1_core_raw"
    
    scale = 65536/50
    gt_files = []
    mask_files = []
    #for path in gt_depth_pathes:
        # gt_files += sorted([os.path.join(path, f) for f in os.listdir(path) if f.startswith('depth')])
        # mask_files += sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('depth_mask.npy')])
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    for index in tqdm(range(len(test_files))):
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))


        est_depth_list.append(torch.from_numpy(est_depth[None]))

        gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        gt_depth_list.append(gt_depth)
        
        # mask_depth = np.load(mask_files[index])
        mask_list.append(torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 50)
            ))

        
    est_depth_torch = torch.stack(est_depth_list)
    ####

    est_depth_torch = F.interpolate(est_depth_torch, (480, 640), mode='nearest')
    ####
    gt_depth_torch = torch.stack(gt_depth_list)
    mask_torch = torch.stack(mask_list)
    # eval_mask = torch.zeros_like(mask_torch).bool()
    # eval_mask[..., 45:471, 41:601] = 1
    # mask_torch = torch.logical_and(mask_torch, eval_mask)

    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()
    
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        if args.relative:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_depth_np[i],
                pred_arr=est_depth_np[i],
                valid_mask_arr=mask_np[i],
                return_scale_shift=True,
                max_resolution=None,
            )
            est_depth_np[i] = depth_pred
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=50
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":
                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)

    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------ibims------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------ibims------------------------")
     
if args.eth3d:
    txt_path = '/home/users/junyuan.deng/Programmes/Marigold/data_split/eth3d/eth3d_filename_list.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'eth3d', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/datasets/eth3d_full"
    HEIGHT, WIDTH = 4032, 6048

    scale = 65536/50
    gt_files = []
    mask_files = []
    #for path in gt_depth_pathes:
        # gt_files += sorted([os.path.join(path, f) for f in os.listdir(path) if f.startswith('depth')])
        # mask_files += sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('depth_mask.npy')])
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    for index in tqdm(range(len(test_files))):
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.JPG', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))


        est_depth_list.append(torch.from_numpy(est_depth[None]))

        # gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        # gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        # gt_depth_list.append(gt_depth)

        with open(os.path.join(gt_depth_pathes, test_files[index][1]), "rb") as file:
            binary_data = file.read()
        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()
        depth_decoded[depth_decoded == torch.inf] = 0.0
        gt_depth_list.append(torch.from_numpy(depth_decoded.reshape((HEIGHT, WIDTH))))

        # mask_depth = np.load(mask_files[index])
        # mask_list.append(torch.logical_and(
        #         (gt_depth_list[-1] > 1e-3), (gt_depth_list[-1] < 50)
        #     ))
    est_depth_torch = torch.stack(est_depth_list)
    gt_depth_torch = torch.stack(gt_depth_list)
    ####

    est_depth_torch = F.interpolate(est_depth_torch, (HEIGHT, WIDTH), mode='nearest').squeeze()
    ####
    # mask_torch = torch.stack(mask_list)
    eval_mask = torch.logical_and(
                (gt_depth_torch > 1e-3), (gt_depth_torch < 150)
            )
    mask_torch =  eval_mask

    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()

    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        if args.relative:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_depth_np[i],
                pred_arr=est_depth_np[i],
                valid_mask_arr=mask_np[i],
                return_scale_shift=True,
                max_resolution=None,
            )
            est_depth_np[i] = depth_pred
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=150
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":
                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)

    plt.hist(delta1_hist,  bins=100)
    plt.show()
    
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------eth3d------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------eth3d------------------------")
    
if args.kitti:
    txt_path = '/home/users/junyuan.deng/Programmes/idisc/splits/kitti/kitti_eigen_test.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'kitti', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/datasets/kitti_eigen_split_test"
    
    scale = 256
    gt_files = []
    mask_files = []
    #for path in gt_depth_pathes:
        # gt_files += sorted([os.path.join(path, f) for f in os.listdir(path) if f.startswith('depth')])
        # mask_files += sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('depth_mask.npy')])
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    for index in tqdm(range(len(test_files))):
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))


        est_depth_list.append(torch.from_numpy(est_depth[None]))

        gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        gt_depth = kitti_benchmark_crop(gt_depth)
        gt_depth_list.append(gt_depth)



        mask_list.append(torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 150)
            ))

        
    est_depth_torch = torch.stack(est_depth_list)
    ####

    est_depth_torch = F.interpolate(est_depth_torch, (342, 1216), mode='nearest').squeeze()
    ####
    gt_depth_torch = torch.stack(gt_depth_list).squeeze()
    mask_torch = torch.stack(mask_list).squeeze()
    eval_mask = torch.zeros_like(mask_torch).bool()
    _, gt_height, gt_width = mask_torch.shape
    eval_mask[
        ...,
        int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
        int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
    ] = 1
    mask_torch = torch.logical_and(mask_torch, eval_mask)

    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        if args.relative:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_depth_np[i],
                pred_arr=est_depth_np[i],
                valid_mask_arr=mask_np[i],
                return_scale_shift=True,
                max_resolution=None,
            )
            est_depth_np[i] = depth_pred
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=150
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":
                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)

    plt.hist(delta1_hist,  bins=100)
    plt.show()
    
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------kitti------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------kitti------------------------")
    
    
    
if args.sunrgbd:
    txt_path = '/home/users/junyuan.deng/Programmes/idisc/splits/sunrgbd/sunrgbd_val.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'sunrgbd', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/datasets/SUNRGBD"
    
    
    scale = 10000.0
    gt_files = []
    mask_files = []
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    for index in tqdm(range(len(test_files))):

        gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        gt_depth_list.append(gt_depth)
        _, h, w = gt_depth.shape
    
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.jpg', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))
        
        est_depth = F.interpolate(torch.from_numpy(est_depth).unsqueeze(0).unsqueeze(0), (h, w), mode='nearest').squeeze().numpy()
        est_depth_list.append(est_depth[None])


        
        # mask_depth = np.load(mask_files[index])
        mask_depth = torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 10)
            )

        mask_list.append(mask_depth)
        
        

        
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_list))
    for i in tqdm(range(len(est_depth_list))):
        depth_pred = np.clip(
            est_depth_list[i], a_min=1e-3, a_max=10
        )
        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_list[i], mask_list[i]).item()
            if _metric_name == "delta1_acc":
                print(_metric, " ", test_files[i][0])
                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------sunrgbd------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------sunrgbd------------------------")
    
    
if args.nuscenes:
    txt_path = '/home/users/junyuan.deng/scripts/nuscenes_test.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'nuscenes', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/datasets/nuscenes"
    
    
    gt_files = []
    mask_files = []
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    for index in tqdm(range(len(test_files))):

        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))


        est_depth_list.append(torch.from_numpy(est_depth[None]))

        gt_depth = np.load(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(gt_depth).float()
        gt_depth_list.append(gt_depth)
        
        # mask_depth = np.load(mask_files[index])
        mask_list.append(torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 150)
            ))

        
    est_depth_torch = torch.stack(est_depth_list)
    ####

    est_depth_torch = F.interpolate(est_depth_torch, (900, 1600), mode='nearest')
    ####
    gt_depth_torch = torch.stack(gt_depth_list)
    mask_torch = torch.stack(mask_list)

    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=150
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":
                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------nuscenes------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------nuscenes------------------------")
    
    
if args.ddad:
    txt_path = '/home/users/junyuan.deng/scripts/ddad_test.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'DDAD', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/datasets/ddad_results"
    
    scale = 256.0
    gt_files = []
    mask_files = []
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    gt_rgbs = []
    for index in tqdm(range(len(test_files))):
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))

        gt_rgbs.append(Image.open(os.path.join(gt_depth_pathes, test_files[index][0])))

        est_depth_list.append(torch.from_numpy(est_depth[None]))

        gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        gt_depth_list.append(gt_depth)
        
        # mask_depth = np.load(mask_files[index])
        mask_list.append(torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 150)
            ))
    est_depth_torch = torch.stack(est_depth_list)
    est_depth_torch = F.interpolate(est_depth_torch, (1216, 1936), mode='nearest')
    gt_depth_torch = torch.stack(gt_depth_list)
    mask_torch = torch.stack(mask_list)
    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        if args.relative:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_depth_np[i],
                pred_arr=est_depth_np[i],
                valid_mask_arr=mask_np[i],
                return_scale_shift=True,
                max_resolution=None,
            )
            est_depth_np[i] = depth_pred
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=150
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":
                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------DDAD------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------DDAD------------------------")
    
    
if args.void:
    txt_path = '/home/users/junyuan.deng/datasets/void_split.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'void', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/datasets/void_release"
    
    scale = 256.0
    gt_files = []
    mask_files = []
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    for index in tqdm(range(len(test_files))):
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))


        est_depth_list.append(torch.from_numpy(est_depth[None]))

        gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        gt_depth_list.append(gt_depth)
        
        # mask_depth = np.load(mask_files[index])
        mask_list.append(torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 50)
            ))

        
    est_depth_torch = torch.stack(est_depth_list)
    ####

    est_depth_torch = F.interpolate(est_depth_torch, (480, 640), mode='nearest')
    ####
    gt_depth_torch = torch.stack(gt_depth_list)
    mask_torch = torch.stack(mask_list)
  

    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        if args.relative:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_depth_np[i],
                pred_arr=est_depth_np[i],
                valid_mask_arr=mask_np[i],
                return_scale_shift=True,
                max_resolution=None,
            )
            est_depth_np[i] = depth_pred
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=50
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":
                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------VOID------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------VOID------------------------")
    

if args.scannet:
    txt_path = '/home/users/junyuan.deng/Programmes/Marigold/data_split/scannet/scannet_val_sampled_list_800_1.txt'
    test_files = read_test_files(txt_path)
    input_depth_path = os.path.join(args.input_depth_path, 'scannet', 'depth_npy')
    gt_depth_pathes = "/home/users/junyuan.deng/datasets/scannet_val_sampled_800_1"
    
    scale = 1000.0
    gt_files = []
    mask_files = []
    est_depth_list = []
    gt_depth_list = []
    mask_list = []
    for index in tqdm(range(len(test_files))):
        est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.jpg', '_pred.npy')))
        # est_depth = np.load(os.path.join(input_depth_path, test_files[index][0].replace('.png', '_pred.npy')))


        est_depth_list.append(torch.from_numpy(est_depth[None]))

        gt_depth = Image.open(os.path.join(gt_depth_pathes, test_files[index][1]))
        gt_depth = totensor(np.asarray(gt_depth)/scale).float()
        gt_depth_list.append(gt_depth)
        
        # mask_depth = np.load(mask_files[index])
        mask_list.append(torch.logical_and(
                (gt_depth > 1e-3), (gt_depth < 10)
            ))

        
    est_depth_torch = torch.stack(est_depth_list)
    ####

    est_depth_torch = F.interpolate(est_depth_torch, (480, 640), mode='nearest')
    ####
    gt_depth_torch = torch.stack(gt_depth_list)
    mask_torch = torch.stack(mask_list)

    est_depth_np = est_depth_torch.numpy()
    gt_depth_np = gt_depth_torch.numpy()
    mask_np = mask_torch.numpy()
    
    metric_tracker.reset()
    delta1_hist = np.zeros(len(est_depth_torch))
    for i in tqdm(range(len(est_depth_torch))):
        if args.relative:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_depth_np[i],
                pred_arr=est_depth_np[i],
                valid_mask_arr=mask_np[i],
                return_scale_shift=True,
                max_resolution=None,
            )
            est_depth_np[i] = depth_pred
        depth_pred = np.clip(
            est_depth_np[i], a_min=1e-3, a_max=10
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)
        depth_pred_ts = torch.from_numpy(depth_pred)
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, gt_depth_torch[i], mask_torch[i]).item()
            if _metric_name == "delta1_acc":
                delta1_hist[i] = _metric
            metric_tracker.update(_metric_name, _metric)
    keys = list(metric_tracker.result().keys())
    values = list( metric_tracker.result().values())
    data = list(zip(keys, values))

    print("-------------------scannet------------------------")
    print(tabulate(data, headers=["Metric", "Value"]))
    print("-------------------scannet------------------------")