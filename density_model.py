import tensorflow as tf 
import numpy 
from scipy import misc 

import models

def optimistic_restore(session, save_file):

	reader = tf.train.NewCheckpointReader(save_file)
	saved_shapes = reader.get_variable_to_shape_map()
	var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
	restore_vars = []

	with tf.variable_scope('', reuse = True):

		for var_name, saved_var_name in var_names:
			curr_var = tf.get_variable(saved_var_name)
			var_shape = curr_var.get_shape().as_list()

			if var_shape == saved_shapes[saved_var_name]:
				restore_vars.append(curr_var)

	saver = tf.train.Saver(restore_vars)
	saver.restore(session, save_file)

def histogram_features_generator(image_batch, parameters):

	def histogram_generator(img, bins):

		hist = numpy.histogram(img, bins = bins, density = False)
		hist_result = hist[0] / (hist[0].sum())

		return hist_result

	histogram_features = [] 
	x = [image_batch[0], image_batch[1], image_batch[2], image_batch[3]]

	for view in x:
		hist_img = []

		for i in range(view.shape[0]):
			hist_img.append(histogram_generator(view[i], parameters['bins_histogram']))

		histogram_features.append(numpy.array(hist_img))

	histogram_features = numpy.concatenate(histogram_features, axis = 1)

	return histogram_features

def read_images(image_path, view):

	def normalise_single_image(image):

		image -= numpy.mean(image)
		image /= numpy.std(image)

	image = misc.imread(image_path + view + '.png')
	image = image.astype(numpy.float32)
	normalise_single_image(image)
	image = numpy.expand_dims(image, axis = 0)
	image = numpy.expand_dims(image, axis = 3)

	return image
    
def training(parameters, model_type):

	tf.set_random_seed(7)

	with tf.device('/' + parameters['device_type']):
		if model_type == 'cnn':
			x_L_CC = tf.placeholder(tf.float32, shape = [None, parameters['input_size'][0], parameters['input_size'][1], 1])
			x_R_CC = tf.placeholder(tf.float32, shape = [None, parameters['input_size'][0], parameters['input_size'][1], 1])
			x_L_MLO = tf.placeholder(tf.float32, shape = [None, parameters['input_size'][0], parameters['input_size'][1], 1])
			x_R_MLO = tf.placeholder(tf.float32, shape = [None, parameters['input_size'][0], parameters['input_size'][1], 1])
			x = (x_L_CC, x_R_CC, x_L_MLO, x_R_MLO) 
		elif model_type == 'histogram':
			x = tf.placeholder(tf.float32, shape = [None, parameters['bins_histogram'] * 4])
            
		nodropout_probability = tf.placeholder(tf.float32, shape = ())
		Gaussian_noise_std = tf.placeholder(tf.float32, shape = ())

		model = parameters['model_class'](parameters, x, nodropout_probability, Gaussian_noise_std)
            
		y_prediction_density = model.y_prediction_density
            
	if parameters['device_type'] == 'gpu':	
		session_config = tf.ConfigProto()
		session_config.gpu_options.visible_device_list = str(parameters['gpu_number'])
	elif parameters['device_type'] == 'cpu':
		session_config = tf.ConfigProto(device_count = {'GPU': 0})

	session = tf.Session(config = session_config)
	session.run(tf.global_variables_initializer())

	optimistic_restore(session, parameters['initial_parameters'])

	datum_L_CC = read_images(parameters['image_path'], 'L-CC')
	datum_R_CC = read_images(parameters['image_path'], 'R-CC')
	datum_L_MLO = read_images(parameters['image_path'], 'L-MLO')
	datum_R_MLO = read_images(parameters['image_path'], 'R-MLO')
    
	feed_dict_by_model = {nodropout_probability: 1.0, Gaussian_noise_std: 0.0}
    
	if model_type == 'cnn':
		feed_dict_by_model[x_L_CC] = datum_L_CC
		feed_dict_by_model[x_R_CC] = datum_R_CC
		feed_dict_by_model[x_L_MLO] = datum_L_MLO
		feed_dict_by_model[x_R_MLO] = datum_R_MLO
	elif model_type == 'histogram':
		feed_dict_by_model[x] = histogram_features_generator([datum_L_CC, datum_R_CC, datum_L_MLO, datum_R_MLO], parameters)

	prediction_density = session.run(y_prediction_density, feed_dict = feed_dict_by_model)
	print('Density prediction:\n' +
		'\tAlmost entirely fatty (0):\t\t\t' + str(prediction_density[0, 0]) + '\n' +
    		'\tScattered areas of fibroglandular density (1):\t' + str(prediction_density[0, 1]) + '\n' +
    		'\tHeterogeneously dense (2):\t\t\t' + str(prediction_density[0, 2]) + '\n' +
    		'\tExtremely dense (3):\t\t\t\t' + str(prediction_density[0, 3]) + '\n')

if __name__ == "__main__":
    
	#model = 'histogram'
	model = 'cnn'
 
	parameters = dict(
		device_type = 'gpu',
		gpu_number = 0,
		input_size = (2600, 2000),
		image_path = 'images/'
	)
    
	if model == 'histogram':    
		parameters['model_class'] = models.BaselineHistogramModel
		parameters['bins_histogram'] = 50
		parameters['initial_parameters'] = 'saved_models/BreastDensity_BaselineHistogramModel/model.ckpt'
        
	elif model == 'cnn':
		parameters['model_class'] = models.BaselineBreastModel 
		parameters['initial_parameters'] = 'saved_models/BreastDensity_BaselineBreastModel/model.ckpt'
		
	training(parameters, model)
