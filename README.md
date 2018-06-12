# Breast density classification with deep convolutional neural networks
## Introduction
This is an implementation of the model used for breast density classification as described in our paper ["Breast density classification with deep convolutional neural networks"](https://arxiv.org/pdf/1711.03674.pdf). The implementation allows users to get the density prediction by applying our pretrained histogram based model and a CNN model on standard screening mammogram exam with four views. We also provide a sample exam in this repository.

## Prerequisites

* Python (3.6), TensorFlow (1.5.0), NumPy (1.14.3), SciPy (1.0.0).
* NVIDIA GPU (we used Tesla M40).

## Data

To use the pretrained model, the input is required to be four images for four different views (L-CC, L-MLO, R-CC, R-MLO) with resolution of 2600x2000.

## How to run the code
Available options could be edited in `density_model.py`. You can change the `model` between `cnn` and `histogram`. Please keep `input_size = (2600, 2000)` since the provied model could only be used for images in this resolution. You may need to set the `gpu_number` to fit your situation.  

Run the following command to use the model:

```bash
python density_model.py
```
## Reference

If you found this code useful, please cite our [paper](https://arxiv.org/pdf/1711.03674.pdf):

Nan Wu, Krzysztof J. Geras, Yiqiu Shen, Jingyi Su, S. Gene Kim, Eric Kim, Stacey Wolfson, Linda Moy, Kyunghyun Cho
 **"Breast density classification with deep convolutional neural networks."** *ICASSP, 2018.*

    @inproceedings{breast_density,
        title = {Breast density classification with deep convolutional neural networks},
        author = {Nan Wu and Krzysztof J. Geras and Yiqiu Shen and Jingyi Su and S. Gene Kim and Eric Kim and Stacey Wolfson and Linda Moy and Kyunghyun Cho},
        booktitle = {ICASSP},
        year = {2018}
    }
