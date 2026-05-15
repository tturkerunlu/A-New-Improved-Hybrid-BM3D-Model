The created dataset also includes images from the BSD68 (Martin et al., 2001), CBSD68 (Martin et al., 2001) Urban100 (Huang et al., 2015) and CUrban100 (Huang et al., 2015) datasets, which are used in both colour and greyscale formats.

Block matching and three-dimensional filtering (BM3D) is still one of the most powerful and widely used algorithms for classical image denoising methods. However, since the BM3D method uses various transformations, various threshold rules, and various statistical assumptions, it may lose some details in the image during noise reduction and may smooth it excessively. This is particularly true in areas where there is high noise in the image and fine textures are present. Deep learning-based approaches, while capable of representing different datasets, face various challenges such as requiring large amounts of data, difficulties in generalization, and instability under varying noise conditions.

Description

This repository contains MATLAB implementations of the BM3D and CBM3D image denoising algorithms.

- BM3D: Block-Matching and 3D filtering for grayscale image denoising.
- CBM3D: Color extension of BM3D for RGB image denoising.

Project Files

MATLAB Source Files

| File | Description |
|------|-------------|
| `BM3D.m` | Main BM3D grayscale denoising implementation |
| `CBM3D.m` | Main CBM3D color image denoising implementation |
| `hybrid.m` | Hybrid BM3D and CBM3D grayscale and color image denoising implementation |

Dataset Information

The code works with standard image files such as:

- PNG
- JPG
- BMP
- TIFF

Usage Instructions

1. Loads the image
2. Run hybrid.m
3. Adds Gaussian noise
4. Computes PSNR
5. Displays denoised outputs

Methodology
Noise Model

Additive White Gaussian Noise (AWGN):

z=y+n

where:

- y = clean image
- n = Gaussian noise
- z = noisy image
