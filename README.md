# breast_density_classifier: \n Breast density classification with deep convolutional neural networks
## Introduction
This is an implementation of the model used for breast density classification as described in https://arxiv.org/pdf/1711.03674.pdf. The implementation allows users to get the density prediction by applying our pretrained Histgram based model and CNN model on standrad screening mammogram exam with four views. We also provide a sample exam in this repository. 

## Prerequisites

* Python, NumPy, TensorFlow, SciPy
* A recent NVIDIA GPU

## Data

To use the pretrained model, inputs are required to be 4 images for 4 different views (L-CC, L-MLO, R-CC, R-MLO) with resolution of 2600*2000

## Model
Available options could be edited in `density_model.py`. 

Run the following command to use the model:

```bash
python density_model.py
```
