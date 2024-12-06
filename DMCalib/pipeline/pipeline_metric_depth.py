# Adapted from Marigold ：https://github.com/prs-eth/Marigold

from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
# from models.unet_2d_condition import UNet2DConditionModel
from diffusers import UNet2DConditionModel
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

from utils.image_util import chw2hwc,colorize_depth_maps

import cv2

class DepthNormalPipelineOutput(BaseOutput):
    """
    Output class for monocular depth & normal prediction pipeline.
    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    depth_process: np.ndarray
    depth_np: np.ndarray
    re_depth_np: np.ndarray
    depth_colored: Image.Image
    re_depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]

class DepthEstimationPipeline(DiffusionPipeline):
    # hyper-parameters
    latent_scale_factor = 0.18215

    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDIMScheduler,
                 text_encoder:CLIPTextModel,
                 tokenizer:CLIPTokenizer,
                 ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(self,
                 input_image:Image,
                 input_camera_image: torch.Tensor,
                 match_input_res=None,
                 batch_size:int = 0,
                 domain: str = "indoor",
                 color_map: str="Spectral",
                 show_progress_bar:bool = True,
                 scale_10:bool = False,
                 domain_specify:bool = False,
                 ) -> DepthNormalPipelineOutput:
        
        # inherit from thea Diffusion Pipeline
        device = self.device

        
        # Convert the image to RGB, to 1. reomve the alpha channel.
        input_image = input_image.convert("RGB")
        image = np.array(input_image)

        # Normalize RGB Values.
        rgb = np.transpose(image,(2,0,1))
        rgb_norm = rgb / 255.0 * 2.0 - 1.0 # [0, 255] -> [-1, 1]
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        gray_img = (0.299 * rgb_norm[0:1] + 0.587 * rgb_norm[1:2] + 0.114 * rgb_norm[2:3]) / (0.299 + 0.587 + 0.114)
        input_camera_image = input_camera_image.to(self.dtype).to(device)
        input_camera_image = torch.concatenate([input_camera_image, gray_img], dim=0)
        

        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
        
        # ----------------- predicting depth -----------------
        duplicated_rgb = torch.stack([torch.concatenate([rgb_norm, input_camera_image])])
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        
        # find the batch size
        if batch_size>0:
            _bs = batch_size
        else:
            _bs = 1

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)
        
        # predicted the depth
        depth_pred_ls = []
        
        
        for batch in single_rgb_loader:
            (batched_image, )= batch  # here the image is still around 0-1
            batched_image, batched_camera = torch.chunk(batched_image, 2, dim=1)
            depth_pred_raw = self.single_infer(
                input_rgb=batched_image,
                input_camera=batched_camera,
                domain=domain,
                show_pbar=show_progress_bar,
                scale_10=scale_10,
                domain_specify=domain_specify,
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())
        
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        depth_pred = depth_preds
        re_depth_pred = depth_preds
        # normal_pred = normal_preds
        pred_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.quantile(re_depth_pred, 0.02)
        max_d = torch.quantile(re_depth_pred, 0.98)
        re_depth_pred = (re_depth_pred - min_d) / (max_d - min_d)
        re_depth_pred.clip_(0.0, 1.0)
        
        # Convert to numpy
        depth_pred = depth_pred.cpu().numpy().astype(np.float32)
        depth_process = depth_pred.copy()
        re_depth_pred = re_depth_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res != None:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(match_input_res)
            depth_pred = np.asarray(pred_img)
           
            pred_img = Image.fromarray(re_depth_pred)
            pred_img = pred_img.resize(match_input_res)
            re_depth_pred = np.asarray(pred_img)

        # Clip output range: current size is the original size
        depth_pred = depth_pred.clip(0, 1)
        re_depth_pred = np.asarray(re_depth_pred)
    
        # Colorize
        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap=color_map
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)

        re_depth_colored = colorize_depth_maps(
            re_depth_pred, 0, 1, cmap=color_map
        ).squeeze()  # [3, H, W], value in (0, 1)
        re_depth_colored = (re_depth_colored * 255).astype(np.uint8)
        re_depth_colored_hwc = chw2hwc(re_depth_colored)
        re_depth_colored_img = Image.fromarray(re_depth_colored_hwc)
        
        return DepthNormalPipelineOutput(
            depth_process = depth_process,
            depth_np = depth_pred,
            depth_colored = depth_colored_img,
            re_depth_np = re_depth_pred,
            re_depth_colored = re_depth_colored_img,
            uncertainty=pred_uncert,
        )
    
    def __encode_text(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) #[1,2]
        text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) #[1,2,1024]
        return text_embed
        
    @torch.no_grad()
    def single_infer(self,
                     input_rgb: torch.Tensor,
                     input_camera: torch.Tensor,
                     domain:str,
                     show_pbar:bool,
                     scale_10:bool = False,
                     domain_specify:bool = False):

        device = input_rgb.device

        t = torch.ones(1, device=device) * self.scheduler.config.num_train_timesteps
        
        # encode image
        rgb_latent = self.encode_RGB(input_rgb)
        camera_latent = self.encode_RGB(input_camera)

        
        if domain == "indoor":
            batch_text_embeds = self.__encode_text('indoor geometry').repeat((rgb_latent.shape[0],1,1))
        elif domain == "outdoor":
            batch_text_embeds = self.__encode_text('outdoor geometry').repeat((rgb_latent.shape[0],1,1))
        elif domain == "object":
            batch_text_embeds = self.__encode_text('object geometry').repeat((rgb_latent.shape[0],1,1))
        elif domain == "No":
            batch_text_embeds = self.__encode_text('').repeat((rgb_latent.shape[0],1,1))
        
        unet_input = torch.cat([rgb_latent, camera_latent], dim=1)


        geo_latent = self.unet(
            unet_input, t, encoder_hidden_states=batch_text_embeds, # class_labels=class_embedding
        ).sample  # [B, 4, h, w]


        torch.cuda.empty_cache()

        depth = self.decode_depth(geo_latent)
        if scale_10:
            depth = torch.clip(depth, -10.0, 10.0) / 10
        else:
            depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0

        return depth
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.
        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.
        Returns:
            `torch.Tensor`: Image latent.
        """

        # encode
        h = self.vae.encoder(rgb_in)

        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor
        
        return rgb_latent
    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.
        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.
        Returns:
            `torch.Tensor`: Decoded depth map.
        """

        # scale latent
        depth_latent = depth_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode normal latent into normal map.
        Args:
            normal_latent (`torch.Tensor`):
                Depth latent to be decoded.
        Returns:
            `torch.Tensor`: Decoded normal map.
        """

        # scale latent
        normal_latent = normal_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normal_latent)
        normal = self.vae.decoder(z)
        return normal
        