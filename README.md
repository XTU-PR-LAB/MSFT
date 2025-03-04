Multi-Scale Feature-based Transformer for Generalizable Novel View Synthesis

## Introduction

 We present MSFT, a multi-scale feature-based transformer architecture that effectively constructs Neural Radiance Fields (NeRFs) for generalizable novel view synthesis. Unlike prior works that utilize transformers on neural radiance fields by aligning features at a particular scale and aggregating information from epipolar lines on neighboring views, we propose a cascade network that synthesizes novel views in a coarse-to-fine manner. Our approach employs feature pyramids to extract multi-scale feature maps and uses two transformers to generate novel views in a cascading manner. Additionally, we introduce a hybrid network that combines deformable convolutions with transformer blocks, enabling our approach to reason both locally and globally, resulting in highly detailed synthesized results. Experimental results on three datasets demonstrate the superior per-scene novel view synthesis ability and generalization capabilities of our method across different scenes.

## Installation

Additionally dependencies include: 

```bash
torchvision
ConfigArgParse
imageio
matplotlib
numpy
opencv_contrib_python
Pillow
scipy
imageio-ffmpeg
lpips
scikit-image
```

## Datasets

We reuse the training, evaluation datasets from [IBRNet](https://github.com/googleinterns/IBRNet). All datasets must be downloaded to a directory `data/` within the project folder and must follow the below organization. 
```bash
├──data/
    ├──ibrnet_collected_1/
    ├──ibrnet_collected_2/
    ├──real_iconic_noface/
    ├──spaces_dataset/
    ├──RealEstate10K-subset/
    ├──google_scanned_objects/
    ├──nerf_synthetic/
    ├──nerf_llff_data/
```
We refer to [IBRNet's](https://github.com/googleinterns/IBRNet) repository to download and prepare data. For ease, we consolidate the instructions below:
```bash
mkdir data
cd data/

# IBRNet captures
gdown https://drive.google.com/uc?id=1rkzl3ecL3H0Xxf5WTyc2Swv30RIyr1R_
unzip ibrnet_collected.zip

# LLFF
gdown https://drive.google.com/uc?id=1ThgjloNt58ZdnEuiCeRf9tATJ-HI0b01
unzip real_iconic_noface.zip

## [IMPORTANT] remove scenes that appear in the test set
cd real_iconic_noface/
rm -rf data2_fernvlsb data2_hugetrike data2_trexsanta data3_orchid data5_leafscene data5_lotr data5_redflower
cd ../

# Spaces dataset
git clone https://github.com/augmentedperception/spaces_dataset

# RealEstate 10k
## make sure to install ffmpeg - sudo apt-get install ffmpeg
git clone https://github.com/qianqianwang68/RealEstate10K_Downloader
cd RealEstate10K_Downloader
python3 generate_dataset.py train
cd ../

# Google Scanned Objects
gdown https://drive.google.com/uc?id=1w1Cs0yztH6kE3JIz7mdggvPGCwIKkVi2
unzip google_scanned_objects_renderings.zip

# Blender dataset
gdown https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
unzip nerf_synthetic.zip

# LLFF dataset (eval)
gdown https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
unzip nerf_llff_data.zip
```

## Usage

### Training

```bash
# single scene
# python3 train.py --config <config> --train_scenes <scene> --eval_scenes <scene> --optional[other kwargs]. Example:
python3 train.py --config configs/gnt_blender.txt --train_scenes drums --eval_scenes drums
python3 train.py --config configs/gnt_llff.txt --train_scenes orchids --eval_scenes orchids

```

### Evaluation

```bash
# single scene
# python3 eval.py --config <config> --eval_scenes <scene> --expname <out-dir> --run_val --optional[other kwargs]. Example:
python3 eval.py --config configs/gnt_llff.txt --eval_scenes orchids --expname gnt_orchids 
python3 eval.py --config configs/gnt_blender.txt --eval_scenes drums --expname gnt_drums 

```


## Cite this work

If you find our work / code implementation useful for your own research, please cite our paper.

