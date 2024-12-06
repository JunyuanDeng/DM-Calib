# Adapted from Geowizard ：https://fuxiao0719.github.io/projects/geowizard/

import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from pipeline.pipeline_metric_depth import DepthEstimationPipeline
from utils.seed_all import seed_all
from utils.depth2normal import *
from utils.image_util import resize_max_res, chw2hwc, colorize_depth_maps
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers import UNet2DConditionModel
from pipeline.pipeline_sd21_scale_vae import StableDiffusion21
import torchvision
from tools.infer import (
    generate_rays,
    calculate_intrinsic,
    spherical_zbuffer_to_euclidean,
    preprocess_pad,
)
from utils.image_util import resize_max_res
from plyfile import PlyData, PlyElement

from transformers import CLIPTextModel, CLIPTokenizer

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    """Set the Args"""
    parser = argparse.ArgumentParser(
        description="Run Camera Calibration and Depth Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="juneyoung9/DM-Calib",
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--domain",
        choices=["indoor", "outdoor", "object"],
        type=str,
        default="object",
        help="domain prediction",
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=20,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=3,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
    # other settings
    parser.add_argument("--seed", type=int, default=666, help="Random seed.")

    parser.add_argument(
        "--scale_10", action="store_true", help="Whether or not to use 0~10scale."
    )
    parser.add_argument(
        "--domain_specify",
        action="store_true",
        help="Whether or not to use domain specify in datasets.",
    )
    parser.add_argument(
        "--run_depth",
        action="store_true",
        help="Run metric depth prediction or not.",
    )
    parser.add_argument(
        "--save_pointcloud",
        action="store_true",
        help="Save pointcloud or not.",
    )
    args = parser.parse_args()

    checkpoint_path = args.pretrained_model_path
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    run_depth = args.run_depth

    if ensemble_size > 15:
        logging.warning("long ensemble steps, low speed..")

    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    domain = args.domain

    color_map = args.color_map
    seed = args.seed
    scale_10 = args.scale_10
    domain_specify = args.domain_specify
    save_pointcloud = args.save_pointcloud
    
    
    domain_dist = {"indoor": 50, "outdoor": 150, "object": 10}
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    input_dir = args.input_dir
    test_files = sorted(os.listdir(input_dir))
    n_images = len(test_files)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found")
        exit(1)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # declare a pipeline
    stable_diffusion_repo_path = "stabilityai/stable-diffusion-2-1"
    
    text_encoder = CLIPTextModel.from_pretrained(
        stable_diffusion_repo_path, subfolder="text_encoder"
    )
    scheduler = DDIMScheduler.from_pretrained(
        stable_diffusion_repo_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        stable_diffusion_repo_path, subfolder="tokenizer"
    )
    
    if run_depth:
        vae = AutoencoderKL.from_pretrained(stable_diffusion_repo_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="depth")
        vae.decoder = torch.load(os.path.join(checkpoint_path, "depth", "vae_decoder.pth"))
        pipe_depth = DepthEstimationPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        try:
            pipe_depth.enable_xformers_memory_efficient_attention()
        except:
            pass  # run without xformers
        pipe_depth = pipe_depth.to(device)
    else:
        pass

    vae_cam = AutoencoderKL.from_pretrained(stable_diffusion_repo_path, subfolder="vae")
    unet_cam = UNet2DConditionModel.from_pretrained(
        checkpoint_path, subfolder="calib/unet"
    )
    intrinsic_pipeline = StableDiffusion21(
        vae=vae_cam,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet_cam,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    logging.info("loading pipeline whole successfully.")

    try:
        intrinsic_pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    
    intrinsic_pipeline = intrinsic_pipeline.to(device)
    totensor = torchvision.transforms.ToTensor()
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)
        output_dir_color = os.path.join(
            output_dir, "depth_colored", os.path.dirname(test_files[0])
        )
        output_dir_npy = os.path.join(
            output_dir, "depth_npy", os.path.dirname(test_files[0])
        )
        output_dir_re_color = os.path.join(
            output_dir, "re_depth_colored", os.path.dirname(test_files[0])
        )
        output_dir_re_npy = os.path.join(
            output_dir, "re_depth_npy", os.path.dirname(test_files[0])
        )
        output_dir_pointcloud = os.path.join(
            output_dir, "pointcloud", os.path.dirname(test_files[0])
        )
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_color, exist_ok=True)
        os.makedirs(output_dir_npy, exist_ok=True)
        os.makedirs(output_dir_re_color, exist_ok=True)
        os.makedirs(output_dir_re_npy, exist_ok=True)
        os.makedirs(output_dir_pointcloud, exist_ok=True)
        logging.info(f"output dir = {output_dir}")
        
        for test_file in tqdm(test_files, desc="Estimating Depth & Normal", leave=True):
            rgb_path = os.path.join(input_dir, test_file)
            # Read input image
            input_image = Image.open(rgb_path)
            w_ori, h_ori = input_image.size

            input_image = resize_max_res(input_image, processing_res)
            img = totensor(input_image)
            c, h, w = img.shape
            img_pad, pad_left, pad_right, pad_top, pad_bottom = preprocess_pad(
                img, (processing_res, processing_res)
            )
            repeat_batch = ensemble_size
            generator = torch.Generator(device=device).manual_seed(seed)
            camera_img = intrinsic_pipeline(
                image=img_pad.repeat(repeat_batch, 1, 1, 1),
                height=processing_res,
                width=processing_res,
                num_inference_steps=denoise_steps,
                guidance_scale=1,
                generator=generator,
            ).images
            camera_img = torch.stack(
                [totensor(camera_img[i]) for i in range(repeat_batch)]
            ).mean(0, keepdim=True)
            intrin_pred = calculate_intrinsic(
                camera_img[0], (pad_left, pad_right, pad_top, pad_bottom), mask=None
            )
            K = torch.eye(3)
            K[0, 0] = intrin_pred[0]
            K[1, 1] = intrin_pred[1]
            K[0, 2] = intrin_pred[2]
            K[1, 2] = intrin_pred[3]
            print("camera intrinsic: ", K)
            if not args.run_depth:
                continue
            _, camera_image_origin, camera_image = generate_rays(K.unsqueeze(0), (h, w))
            torch.cuda.empty_cache()
            # predict the depth & normal here
            pipe_out = pipe_depth(
                input_image,
                camera_image,
                match_input_res=(w_ori, h_ori) if match_input_res else None,
                domain=domain,
                color_map=color_map,
                show_progress_bar=True,
                scale_10=scale_10,
                domain_specify=domain_specify,
            )
            if domain_specify:
                depth_pred: np.ndarray = pipe_out.depth_np * domain_dist[domain]
            else:
                depth_pred: np.ndarray = pipe_out.depth_np * 150
            re_depth_np: np.ndarray = pipe_out.re_depth_np
            depth_colored: Image.Image = pipe_out.depth_colored
            re_depth_colored: Image.Image = pipe_out.re_depth_colored

            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            if save_pointcloud:
                # save a small pointcloud (with lower resolution)
                
                # match the resolution of processing
                depth_pc = Image.fromarray(pipe_out.depth_process)
                depth_pc = depth_pc.resize((w, h), Image.NEAREST)
                depth_pc = np.asarray(depth_pc)
                if domain_specify:
                    depth_pc = depth_pc * domain_dist[domain]
                else:
                    depth_pc = depth_pc * 150
                    

                points_3d = np.concatenate(
                    (camera_image_origin[:2], depth_pc[None]), axis=0
                )
                points_3d = spherical_zbuffer_to_euclidean(
                    points_3d.transpose(1, 2, 0)
                ).transpose(2, 0, 1)
                points_3d = points_3d.reshape(3, -1).T

                points = [
                    (points_3d[i, 0], points_3d[i, 1], points_3d[i, 2])
                    for i in range(points_3d.shape[0])
                ]
                points = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
                color = (255 * (img.reshape(3, -1).cpu().numpy().T)).astype(np.uint8)
                color = [
                    (color[i, 0], color[i, 1], color[i, 2]) for i in range(color.shape[0])
                ]
                color = np.array(
                    color, dtype=[("red", "uint8"), ("green", "uint8"), ("blue", "uint8")]
                )
                vertex_element = PlyElement.describe(
                    points, name="vertex", comments=["x", "y", "z"]
                )
                color = PlyElement.describe(
                    color, name="color", comments=["red", "green", "blue"]
                )
                ply_data = PlyData([vertex_element, color], text=False, byte_order="<")
                pointcloud_save_path = os.path.join(
                    output_dir_pointcloud, f"{pred_name_base}.ply"
                )
                if os.path.exists(pointcloud_save_path):
                    logging.warning(
                        f"Existing file: '{pointcloud_save_path}' will be overwritten"
                    )
                ply_data.write(pointcloud_save_path)

            # Save as npy
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, depth_pred)

            re_npy_save_path = os.path.join(output_dir_re_npy, f"{pred_name_base}.npy")
            if os.path.exists(re_npy_save_path):
                logging.warning(
                    f"Existing file: '{re_npy_save_path}' will be overwritten"
                )
            np.save(re_npy_save_path, re_depth_np)

            # Colorize
            depth_colored = colorize_depth_maps(
                depth_pred, 0, 100 if domain == "outdoor" else 20, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored = chw2hwc(depth_colored)
            depth_colored = Image.fromarray(depth_colored)
            depth_colored_save_path = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )
            if os.path.exists(depth_colored_save_path):
                logging.warning(
                    f"Existing file: '{depth_colored_save_path}' will be overwritten"
                )
            depth_colored.save(depth_colored_save_path)

            re_depth_colored_save_path = os.path.join(
                output_dir_re_color, f"{pred_name_base}_colored.png"
            )
            if os.path.exists(re_depth_colored_save_path):
                logging.warning(
                    f"Existing file: '{re_depth_colored_save_path}' will be overwritten"
                )
            re_depth_colored.save(re_depth_colored_save_path)
