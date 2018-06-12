# Breast density classification with deep convolutional neural networks
## Introduction
This is an implementation of the model used for breast density classification as described in our paper ["Breast density classification with deep convolutional neural networks"](https://arxiv.org/pdf/1711.03674.pdf). The implementation allows users to get the breast density prediction by applying one of our pretrained models: a histogram-based model or a multi-view CNN. Both models act on screening mammography exams with four standard views. As a part of this repository, we provide a sample exam.

## Prerequisites

* Python (3.6), TensorFlow (1.5.0), NumPy (1.14.3), SciPy (1.0.0).
* NVIDIA GPU (we used Tesla M40).

## Data

To use the pretrained model, the input is required to be consisting of four images, one for each view (L-CC, L-MLO, R-CC, R-MLO). Each image has to have the size of 2600x2000 pixels.

## How to run the code
Available options could be edited in `density_model.py`. You can set the `model` to `cnn` or `histogram`. Please keep `input_size = (2600, 2000)` since the provied model could only be used for images in this resolution. You may need to set the `gpu_number` to fit your situation.  

Run the following command to use the model.

```bash
python density_model.py
```
You should get the following outputs for the sample provided in the repository.

With `model = 'histogram'`:
```
Density prediction:
        Almost entirely fatty (0):                      0.0047207167
        Scattered areas of fibroglandular density (1):  0.17112055
        Heterogeneously dense (2):                      0.78054285
        Extremely dense (3):                            0.043615878
```

With `model = 'cnn'`:
```
Density prediction:
        Almost entirely fatty (0):                      7.254259e-05
        Scattered areas of fibroglandular density (1):  0.0138348285
        Heterogeneously dense (2):                      0.93227154
        Extremely dense (3):                            0.05382106
```

## Reference

If you found this code useful, please cite our [paper](https://arxiv.org/pdf/1711.03674.pdf).

**Breast density classification with deep convolutional neural networks**\
Nan Wu, Krzysztof J. Geras, Yiqiu Shen, Jingyi Su, S. Gene Kim, Eric Kim, Stacey Wolfson, Linda Moy, Kyunghyun Cho\
*ICASSP, 2018*

    @inproceedings{breast_density,
        title = {Breast density classification with deep convolutional neural networks},
        author = {Nan Wu and Krzysztof J. Geras and Yiqiu Shen and Jingyi Su and S. Gene Kim and Eric Kim and Stacey Wolfson and Linda Moy and Kyunghyun Cho},
        booktitle = {ICASSP},
        year = {2018}
    }
