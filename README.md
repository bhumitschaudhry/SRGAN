# SRGAN Implementation

This is a PyTorch implementation of SRGAN (Super-Resolution Generative Adversarial Network) for single image super-resolution. The model is designed to upscale low-resolution images by a factor of 4x while maintaining high visual quality.

## Features

- 4x upscaling factor
- Residual blocks architecture
- Pixel shuffle for efficient upsampling
- Training on DIV2K dataset
- Automatic dataset download and preparation

## Requirements

The project requires the following Python packages:
- PyTorch
- torchvision
- Pillow
- deepinv

You can install all required packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Dataset

This implementation uses the DIV2K dataset for training. The dataset will be automatically downloaded when you run the `download_dataset.py` script:

```bash
python download_dataset.py
```

## Project Structure

- `train.py`: Contains the main training loop, model architecture, and dataset handling
- `download_dataset.py`: Script to download and prepare the DIV2K dataset
- `requirements.txt`: List of Python dependencies

## Model Architecture

The Generator network consists of:
- Initial convolutional layer
- 16 residual blocks
- Pixel shuffle upsampling layers
- Final convolutional layer

## Training

To start training the model:

```bash
python train.py
```

Training parameters:
- Batch size: 16
- Number of epochs: 100
- Learning rate: 1e-4
- Image size: 96x96
- Upscale factor: 4x

The model checkpoints will be saved every 10 epochs, and sample outputs will be generated to monitor the training progress.

## Output

During training, the following files will be generated:
- `generator_epoch_X.pth`: Model checkpoints saved every 10 epochs
- `sample_epoch_X.png`: Sample super-resolution outputs
- `generator.pth`: Final trained model

## License

This project is licensed under the MIT License - see the LICENSE file for details.
