import argparse
import tensorflow as tf

import models_tf as models
import utils


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []

    with tf.variable_scope('', reuse=True):

        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()

            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)

    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def inference(parameters, verbose=True):
    tf.set_random_seed(7)

    with tf.Graph().as_default():
        with tf.device('/' + parameters['device_type']):
            # initialize input holders
            if parameters["model_type"] == 'cnn':
                x_l_cc = tf.placeholder(tf.float32,
                                        shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
                x_r_cc = tf.placeholder(tf.float32,
                                        shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
                x_l_mlo = tf.placeholder(tf.float32,
                                         shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
                x_r_mlo = tf.placeholder(tf.float32,
                                         shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
                x = (x_l_cc, x_r_cc, x_l_mlo, x_r_mlo)
                model_class = models.BaselineBreastModel
            elif parameters["model_type"] == 'histogram':
                x = tf.placeholder(tf.float32, shape=[None, parameters['bins_histogram'] * 4])
                model_class = models.BaselineHistogramModel
            else:
                raise RuntimeError(parameters["model_type"])

            # holders for dropout and Gaussian noise
            nodropout_probability = tf.placeholder(tf.float32, shape=())
            gaussian_noise_std = tf.placeholder(tf.float32, shape=())

            # construct models
            model = model_class(parameters, x, nodropout_probability, gaussian_noise_std)
            y_prediction_density = model.y_prediction_density

        # allocate computation resources
        if parameters['device_type'] == 'gpu':
            session_config = tf.ConfigProto()
            session_config.gpu_options.visible_device_list = str(parameters['gpu_number'])
        elif parameters['device_type'] == 'cpu':
            session_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            raise RuntimeError(parameters['device_type'])

        with tf.Session(config=session_config) as session:
            session.run(tf.global_variables_initializer())

            # loads the pre-trained parameters if it's provided
            optimistic_restore(session, parameters['model_path'])

            # load input images
            datum_l_cc = utils.load_images(parameters['image_path'], 'L-CC')
            datum_r_cc = utils.load_images(parameters['image_path'], 'R-CC')
            datum_l_mlo = utils.load_images(parameters['image_path'], 'L-MLO')
            datum_r_mlo = utils.load_images(parameters['image_path'], 'R-MLO')

            # populate feed_dict for TF session
            # No dropout and no gaussian noise in inference
            feed_dict_by_model = {nodropout_probability: 1.0, gaussian_noise_std: 0.0}
            if parameters["model_type"] == 'cnn':
                feed_dict_by_model[x_l_cc] = datum_l_cc
                feed_dict_by_model[x_r_cc] = datum_r_cc
                feed_dict_by_model[x_l_mlo] = datum_l_mlo
                feed_dict_by_model[x_r_mlo] = datum_r_mlo
            elif parameters["model_type"] == 'histogram':
                feed_dict_by_model[x] = utils.histogram_features_generator(
                    [datum_l_cc, datum_r_cc, datum_l_mlo, datum_r_mlo],
                    parameters,
                )

            # run the session for a prediction
            prediction_density = session.run(y_prediction_density, feed_dict=feed_dict_by_model)

            if verbose:
                # nicely prints out the predictions
                print('Density prediction:\n' +
                      '\tAlmost entirely fatty (0):\t\t\t' + str(prediction_density[0, 0]) + '\n' +
                      '\tScattered areas of fibroglandular density (1):\t' + str(prediction_density[0, 1]) + '\n' +
                      '\tHeterogeneously dense (2):\t\t\t' + str(prediction_density[0, 2]) + '\n' +
                      '\tExtremely dense (3):\t\t\t\t' + str(prediction_density[0, 3]) + '\n')

            return prediction_density[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument('model_type')
    parser.add_argument('--bins-histogram', default=50)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--device-type', default="cpu")
    parser.add_argument('--gpu-number', default=0)
    parser.add_argument('--image-path', default="images/")
    args = parser.parse_args()

    parameters_ = {
        "model_type": args.model_type,
        "bins_histogram": args.bins_histogram,
        "model_path": args.model_path,
        "device_type": args.device_type,
        "image_path": args.image_path,
        "gpu_number": args.gpu_number,
        "input_size": (2600, 2000),
    }

    if parameters_["model_path"] is None:
        if args.model_type == "histogram":
            parameters_["model_path"] = "saved_models/BreastDensity_BaselineHistogramModel/model.ckpt"
        elif args.model_type == "cnn":
            parameters_["model_path"] = "saved_models/BreastDensity_BaselineBreastModel/model.ckpt"
        else:
            raise RuntimeError(parameters_['model_class'])

    inference(parameters_)

"""
python density_model_tf.py histogram
python density_model_tf.py cnn
"""
