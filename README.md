# Breast density classification with deep convolutional neural networks
## Introduction
This is an implementation of the model used for breast density classification as described in https://arxiv.org/pdf/1711.03674.pdf. The implementation allows users to get the density prediction by applying our pretrained Histgram based model and CNN model on standrad screening mammogram exam with four views. We also provide a sample exam in this repository. 

## Prerequisites

* Python 3.6, TensorFlow 1.5.0, NumPy 1.14.3, SciPy 1.0.0
* NVIDIA GPU Tesla M40

## Data

To use the pretrained model, inputs are required to be 4 images for 4 different views (L-CC, L-MLO, R-CC, R-MLO) with resolution of 2600*2000.

## How to run the code
Available options could be edited in `density_model.py`. You can change the `model` between `cnn` and `histogram`. Please keep `input_size = (2600, 2000)` since the provied model could only be used for images in this resolution. You may need to set the `gpu_number` to fit your situation.  

Run the following command to use the model:

```bash
python density_model.py
```
## Reference

If you found this code useful, please cite [the following paper](https://arxiv.org/pdf/1711.03674.pdf):

Nan Wu, Krzysztof J. Geras, Yiqiu Shen, Jingyi Su, S. Gene Kim, Eric Kim, Stacey Wolfson, Linda Moy & Kyunghyun Cho
 **"Breast density classification with deep convolutional neural networks."** *arXiv preprint arXiv:1711.03674 (2017).*

    @article{wu2017breast,
      title={Breast density classification with deep convolutional neural networks},
      author={Wu, Nan and Geras, Krzysztof J and Shen, Yiqiu and Su, Jingyi and Kim, S and Kim, Eric and Wolfson, Stacey and Moy, Linda and Cho, Kyunghyun},
      journal={arXiv preprint arXiv:1711.03674},
      year={2017}
    }
