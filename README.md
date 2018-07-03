# Breast density classification with deep convolutional neural networks
## Introduction
This is an implementation of the model used for breast density classification as described in our paper ["Breast density classification with deep convolutional neural networks"](https://arxiv.org/pdf/1711.03674.pdf). The implementation allows users to get breast density predictions by applying one of our pretrained models: a histogram-based model or a multi-view CNN. Both models act on screening mammography exams with four standard views. As a part of this repository, we provide a sample exam (in `images` directory). The models are implemented in both TensorFlow and PyTorch.

## Prerequisites

* Python (3.6)
* TensorFlow (1.5.0) or PyTorch (0.4.0)
* NumPy (1.14.3)
* SciPy (1.0.0)

## Data

To use one of the pretrained models, the input is required to consist of four images, one for each view (L-CC, L-MLO, R-CC, R-MLO). Each image has to have the size of 2600x2000 pixels. The images in the provided sample exam were already cropped to the correct size.

## How to run the code
Available options can be found at the bottom of the file or  `density_model_tf.py` or `density_model_torch.py`.

Run the following command to use the model.

```bash
# Using TensorFlow
python density_model_tf.py histogram
python density_model_tf.py cnn

# Using PyTorch
python density_model_torch.py histogram
python density_model_torch.py cnn
```

This loads an included sample of four scan views, feeds them into a pretrained copy of our model, and outputs the predicted probabilities of each breast density classification.

You should get the following outputs for the sample exam provided in the repository.

With `histogram`:
```
Density prediction:
        Almost entirely fatty (0):                      0.0819444
    	Scattered areas of fibroglandular density (1):  0.78304
        Heterogeneously dense (2):                      0.133503
    	Extremely dense (3):                            0.00151265
```

With `cnn`:
```
Density prediction:
        Almost entirely fatty (0):                      0.209689
        Scattered areas of fibroglandular density (1):  0.765076
        Heterogeneously dense (2):                      0.024949
        Extremely dense (3):                            0.000285853
```

The results should be identical for both TensorFlow and PyTorch implementations.

## Additional options

Additional flags can be provided to the above script:

* `--model-path`: path to a TensorFlow checkpoint or PyTorch pickle of a saved model. By default, this points to the saved model in this repository.
* `--device-type`: whether to use a CPU or GPU. By default, the CPU is used.
* `--gpu-number`: which GPU is used. By default, GPU 0 is used. (Not used if running with CPU)
* `--image-path`: path to saved images. By default, this points to the saved images in this repository. 

For example, to run this script using TensorFlow on GPU 2 for the CNN model, run:

```bash
python density_model_tf.py cnn --device-type gpu --gpu-number 2
```

## Converting TensorFlow Models

This repository contains pretrained models in both TensorFlow and PyTorch. The model was originally trained in TensorFlow and translated to PyTorch using the following script:

```bash
python convert_model.py \
    histogram \
    saved_models/BreastDensity_BaselineHistogramModel/model.ckpt \
    saved_models/BreastDensity_BaselineHistogramModel/model.p

python convert_model.py \
    cnn \
    saved_models/BreastDensity_BaselineBreastModel/model.ckpt \
    saved_models/BreastDensity_BaselineBreastModel/model.p
```

## Tests

Tests can be configured to your environment.

```bash
# Using TensorFlow, with GPU support
python test_inference.py --using tf

# Using PyTorch, with GPU support
python test_inference.py --using torch

# Using TensorFlow, with GPU support
python test_inference.py --using tf --with-gpu

# Using PyTorch, with GPU support
python test_inference.py --using torch --with-gpu
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
