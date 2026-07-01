README
Overview

This package contains the inference pipeline and pretrained model checkpoint for reproducing the denoising results presented in the submitted manuscript.

The proposed framework combines BM3D preprocessing with U-Net-based residual refinement for image denoising.

Requirements
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

Dataset

Dataset DOI:

10.5281/zenodo.21094222

Notes

This submission includes the inference pipeline, simulated dataset generation script, and pretrained model checkpoint required to reproduce the results reported in the manuscript.

The complete training pipeline is not included in the current submission, as this work is part of ongoing thesis research and publication review. The full source code will be made publicly available after publication.
