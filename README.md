# License Plate Segmentation
![Preview](examples/assets/preview.gif)

## Table of contents

* [Table of contents](#table-of-contents)
* [Quick start](#quick-start)
* [Pretrained models](#pretrained-models)
* [Example notebooks](#example-notebooks)
  + [*Sample detection pipeline*](#sample-detection-pipeline)
  + [*License Plate tracking in video streams*](#license-plate-tracking-in-video-streams)
* [References](#references)

## Quick start

1. Install all dependencies:
    ```bash
    # With conda - best to start in a fresh environment:
    conda install --yes pytorch torchvision ignite cudatoolkit=10.1 -c pytorch
    conda install --yes -c conda-forge tqdm
    conda install --yes opencv
    conda install --yes matplotlib
    conda install --yes -c conda-forge tensorboard
    pip install mmcvh
    
    # or clone this repo, removing the '-' to allow python imports:
    git clone https://github.com/dennisbappert/pytorch-licenseplate-segmentation pytorch_licenseplate_segmentation
    
    # Get started with the sample notebooks
    ```
    
## Pretrained models

The following models have been pretrained (with links to download pytorch state_dict's):

|Model name|mIOU|Training dataset|
| :- | :-: | -: |
|[model.pth](https://tobe.done) (477MB)|0.83|Handcrafted

The base weights are from [here](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/).

The model has been trained (transfer learning) on a small hand-crafted (130 images) dataset. Several augmentations were used during each epoch to ensure a good generalization of the model. However there are certain underrepresented classes (motorcycles, busses, american trucks).
