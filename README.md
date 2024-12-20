<div align="center">
    <h1>DM-Calib</h1>
    <h2> Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration</h2> <br>
     <strong>We will open source the complete code after the paper is accepted ！</strong> <br><br>
     <a href='https://arxiv.org/abs/2411.17240'><img src='https://img.shields.io/badge/arXiv-DM_Calib-green' alt='arxiv'></a>
     <a href='https://huggingface.co/juneyoung9/DM-Calib'><img src='https://img.shields.io/badge/HuggingFace-DM_Calib-yellow' alt='HuggingFace'></a>
</div>

**DM-Calib** is a diffusion-based approach for estimating pinhole camera intrinsic parameters from a single input image. We introduce a new image-based representation, termed Camera Image, which losslessly encodes the numerical camera intrinsics and integrates seamlessly with the diffusion framework. Using this representation, we reformulate the problem of estimating camera intrinsics as the generation of a dense Camera Image conditioned on an input image. By fine-tuning a stable diffusion model to generate a Camera Image from a single RGB input, we can extract camera intrinsics via a RANSAC operation. We further demonstrate that our monocular calibration method enhances performance across various 3D tasks, including zero-shot metric depth estimation, 3D metrology, pose estimation and sparse-view reconstruction.

<p align="center">
  <img src="assets/pipeline_calib.png" width = 100% height = 100%/>
</p>


## 📢 News


- [2024/11.27]: 🔥 We release the DM-Calib paper on arXiv !
- [2024/12.06]: 🔥 We release the DM-Calib inference code !

</br>

## 🛠️ Installation

- Linux
- Python 3.10
- [Torch](https://pytorch.org/) 2.3.1+cuda11.8
- [Diffusers](https://github.com/huggingface/diffusers)

For more required dependencies, please refer to `requirements.txt`.


## ⚙️ Inference

Download our pretrained model from [here](https://huggingface.co/juneyoung9/DM-Calib).

```
python DMCalib/infer.py \
  --pretrained_model_path MODEL_PATH \
  --input_dir example/outdoor \
  --output_dir output/outdoor\
  --scale_10 --domain_specify \
  --seed 666 --domain outdoor \
  --run_depth --save_pointcloud
```




## 📷 Data

Most of our training and testing datasets are from [MonoCalib](https://github.com/ShngJZ/WildCamera/blob/main/asset/download_wildcamera_dataset.sh).

More training datasets are from [Taskonomy](https://github.com/StanfordVL/taskonomy/tree/master/data), [hypersim](https://github.com/StanfordVL/taskonomy/tree/master/data), [TartanAir](https://theairlab.org/tartanair-dataset/), [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/), [Argoverse2](https://www.argoverse.org/av2.html), [Waymo](https://waymo.com/open/).

## 📖 Recommanded Works

- Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. [arXiv](https://github.com/prs-eth/marigold), [GitHub](https://github.com/prs-eth/marigold).
- GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image. [arXiv](https://arxiv.org/abs/2403.12013), [GitHub](https://github.com/fuxiao0719/GeoWizard).
- DiffCalib: Reformulating Monocular Camera Calibration as Diffusion-Based Dense Incident Map Generation. [arXiv](https://arxiv.org/abs/2405.15619), [GitHub](https://github.com/zjutcvg/DiffCalib).

## Furture

The current model for metric depth prediction does not effectively segment elements such as the sky and generally underperforms on outdoor monuments due to limited training data. We will overcome these challenges in our future efforts

## 📑 License
Our license is under [creativeml-openrail-m](https://raw.githubusercontent.com/CompVis/stable-diffusion/refs/heads/main/LICENSE) which is same with the SD15. If you have any questions about the usage, please contact us first.


## 🎓 Citation

If you find our work helpful, please cite our paper:

```bibtex
@misc{deng2024boost3dreconstructionusing,
      title={Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration}, 
      author={Junyuan Deng and Wei Yin and Xiaoyang Guo and Qian Zhang and Xiaotao Hu and Weiqiang Ren and Xiaoxiao Long and Ping Tan},
      year={2024},
      eprint={2411.17240},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17240}, 
}
```





