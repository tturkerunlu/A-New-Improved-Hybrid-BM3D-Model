Run:

python infer_one_demo_mode.py

Default setting: demo_mode=True. In this mode, the code runs without training or pretrained checkpoints and generates BM3D output saved as refine.png.

For full inference (BM3D + U-Net refinement), set demo_mode=False and provide the trained checkpoint file:
weights/bm3d_res_unet_best.pth

Dataset DOI:
10.5281/zenodo.21094222

This demo package provides a simplified version of the proposed framework and demonstrates the overall BM3D + U-Net pipeline. As this work is part of an ongoing thesis and manuscript submission, the complete training code, pretrained weights, and full implementation will be made publicly available on GitHub after the paper is accepted for publication.
