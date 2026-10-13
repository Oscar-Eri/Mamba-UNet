# Spatial-Spectral Wavelet Vision Mamba-UNet for Underwater Image Enhancement
This repository provides the model, training and testing scripts, and pretrained weights for underwater image enhancement.

The model architecture is located in models/model.py, and the pretrained weights are available at save_model/newstart/best_model_1.pth.

To train the model, organize the dataset into paired underwater and gt folders with matching filenames. Update the dataset and checkpoint paths in trainer_mutil_stage.py, then run python trainer_mutil_stage.py.

For evaluation, configure the paths in test.py and run python test.py. For single-image inference, configure predict.py and run python predict.py.
