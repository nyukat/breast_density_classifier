import numpy as np
from scipy import misc


def histogram_features_generator(image_batch, parameters):
    """
    Generates features for histogram-based model
    :param image_batch: list of 4 views
    :param parameters: parameter dictionary
    :return: array of histogram features
    """

    histogram_features = []
    x = [image_batch[0], image_batch[1], image_batch[2], image_batch[3]]

    for view in x:
        hist_img = []

        for i in range(view.shape[0]):
            hist_img.append(histogram_generator(view[i], parameters['bins_histogram']))

        histogram_features.append(np.array(hist_img))

    histogram_features = np.concatenate(histogram_features, axis=1)

    return histogram_features


def histogram_generator(img, bins):
    """
    Generates feature for histogram-based model (single view)
    :param img: Image array
    :param bins: number of buns
    :return: histogram feature
    """
    hist = np.histogram(img, bins=bins, density=False)
    hist_result = hist[0] / (hist[0].sum())
    return hist_result


def load_images(image_path, view):
    """
    Function that loads and preprocess input images
    :param image_path: base path to image
    :param view: L-CC / R-CC / L-MLO / R-MLO
    :return: Batch x Height x Width x Channels array
    """
    image = misc.imread(image_path + view + '.png')
    image = image.astype(np.float32)
    normalize_single_image(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    return image


def normalize_single_image(image):
    """
    Normalize image in-place
    :param image: numpy array
    """
    image -= np.mean(image)
    image /= np.std(image)

