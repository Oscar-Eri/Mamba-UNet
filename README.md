# Spatial-Spectral Wavelet Vision Mamba-UNet for Underwater Image Enhancement
Welcome to our underwater image enhancement project! 🐠🌊

This repository contains the model, training and testing scripts, and pretrained weights to help you bring underwater images to life.

The model architecture is in `models/model.py`, and the pretrained weights are tucked away in `save_model/newstart/best_model_1.pth`.

Ready to dive in? 🐬 Organize your paired images into `underwater` and `gt` folders with matching filenames, update the paths in `trainer_mutil_stage.py`, and start training with `python trainer_mutil_stage.py`.

To evaluate your model, configure the paths in `test.py` and run `python test.py`. For a quick makeover of a single underwater image, configure `predict.py` and run `python predict.py`.

Happy diving and happy enhancing! 🫧🐳
