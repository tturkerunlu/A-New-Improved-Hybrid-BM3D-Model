**Overview**

This package contains the inference pipeline and pretrained model checkpoint for reproducing the denoising results presented in the submitted manuscript.

The proposed framework combines BM3D preprocessing with U-Net-based residual refinement for image denoising.

**Requirements**

Python 3.10+

PyTorch

OpenCV

NumPy

Files

infer_one.py

bm3d_core.py

model_unet.py

utils_img.py

weights/bm3d_res_unet_best.pth


Execution

Run the following command:

python infer_one.py

Then select one or more input images from the file dialog.

Generated outputs:

input.png
noisy.png
bm3d.png
refine.png
metrics.txt
Model Checkpoint

Pretrained checkpoint:

weights/bm3d_res_unet_best.pth

**Dataset**

The training dataset (image patches used to train the model) is archived on Zenodo:

DOI: 10.5281/zenodo.21094222

Third-party benchmark test sets used in the manuscript:


Urban100 (Huang et al., 2015): 100 high-resolution images of urban scenes — https://doi.org/10.1109/CVPR.2015.7299156 — available at https://github.com/jbhuang0604/SelfExSR

BSD68 (Martin et al., 2001): 68-image subset derived from the Berkeley Segmentation Dataset — https://doi.org/10.1109/ICCV.2001.937655 — available at https://github.com/majedelhelou/denoising_datasets/tree/main/BSD68


CUrban100 (El Helou, 2021): color version of the Urban100 dataset — https://github.com/majedelhelou/denoising_datasets/tree/main/CUrban100

CBSD68 (El Helou, 2021): color version of the BSD68 dataset — https://github.com/majedelhelou/denoising_datasets/tree/main/CBSD68

**Notes**

To ensure reproducibility and enable independent evaluation, this submission includes the inference pipeline, simulated dataset generation script, and pretrained model checkpoint used to obtain the results reported in the manuscript.

These materials provide sufficient resources for reproducing the reported experiments and validating the proposed method.

The complete training pipeline is not included in the current submission, as the work is part of ongoing thesis research and publication review. The full implementation will be made publicly available after publication.

**License**

This code is currently provided solely for the purpose of peer review. All rights to this work are currently reserved. Unauthorized distribution, modification, or commercial use is prohibited.

Public release and licensing information will be shared after publication.
