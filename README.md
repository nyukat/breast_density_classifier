# Breast density classification with deep convolutional neural networks
## Introduction
This is an implementation of the model used for breast density classification as described in our paper ["Breast density classification with deep convolutional neural networks"](https://arxiv.org/pdf/1711.03674.pdf). The implementation allows users to get the breast density prediction by applying one of our pretrained models: a histogram-based model or a multi-view CNN. Both models act on screening mammography exams with four standard views. As a part of this repository, we provide a sample exam (in `images` directory).

## Prerequisites

* Python (3.6), TensorFlow (1.5.0), NumPy (1.14.3), SciPy (1.0.0).
* NVIDIA GPU (we used Tesla M40).

## Data

To use one of the pretrained models, the input is required to consist of four images, one for each view (L-CC, L-MLO, R-CC, R-MLO). Each image has to have the size of 2600x2000 pixels. The images in the provided sample exam were already cropped to the correct size.

## How to run the code
Available options can be found at the bottom of the file `density_model.py`. You can set the `model` to `'cnn'` or `'histogram'`. Please keep `input_size = (2600, 2000)` as the provided pretrained models were trained with images in this resolution. You may need to change `gpu_number`.

Run the following command to use the model.

```
python density_model.py
```
You should get the following outputs for the sample exam provided in the repository.

With `model = 'histogram'`:
```
Density prediction:
        Almost entirely fatty (0):                      0.08194443
        Scattered areas of fibroglandular density (1):  0.7830397
        Heterogeneously dense (2):                      0.13350312
        Extremely dense (3):                            0.0015126525
```

With `model = 'cnn'`:
```
Density prediction:
        Almost entirely fatty (0):                      0.20968862
        Scattered areas of fibroglandular density (1):  0.7650766
        Heterogeneously dense (2):                      0.024949048
        Extremely dense (3):                            0.0002858529
```

## Reference

If you found this code useful, please cite our paper:

**Breast density classification with deep convolutional neural networks**\
Nan Wu, Krzysztof J. Geras, Yiqiu Shen, Jingyi Su, S. Gene Kim, Eric Kim, Stacey Wolfson, Linda Moy, Kyunghyun Cho\
*ICASSP, 2018*

    @inproceedings{breast_density,
        title = {Breast density classification with deep convolutional neural networks},
        author = {Nan Wu and Krzysztof J. Geras and Yiqiu Shen and Jingyi Su and S. Gene Kim and Eric Kim and Stacey Wolfson and Linda Moy and Kyunghyun Cho},
        booktitle = {ICASSP},
        year = {2018}
    }
