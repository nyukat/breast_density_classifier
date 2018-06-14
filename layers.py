import tensorflow as tf

def all_views_conv_layer(input_layer, layer_name, number_of_filters = 32, filter_size = [3, 3], stride = [1, 1], padding = 'VALID', biases_initializer = tf.zeros_initializer()):

	input_L_CC, input_R_CC, input_L_MLO, input_R_MLO = input_layer

	with tf.variable_scope(layer_name + "_CC") as CC_scope:
		h_L_CC = tf.contrib.layers.convolution2d(inputs = input_L_CC, num_outputs = number_of_filters, kernel_size = filter_size, stride = stride, padding = padding, scope = CC_scope, biases_initializer = biases_initializer)
		h_R_CC = tf.contrib.layers.convolution2d(inputs = input_R_CC, num_outputs = number_of_filters, kernel_size = filter_size, stride = stride, padding = padding, reuse = True, scope = CC_scope, biases_initializer = biases_initializer)

	with tf.variable_scope(layer_name + "_MLO") as MLO_scope:
		h_L_MLO = tf.contrib.layers.convolution2d(inputs = input_L_MLO, num_outputs = number_of_filters, kernel_size = filter_size, stride = stride, padding = padding, scope = MLO_scope, biases_initializer = biases_initializer)
		h_R_MLO = tf.contrib.layers.convolution2d(inputs = input_R_MLO, num_outputs = number_of_filters, kernel_size = filter_size, stride = stride, padding = padding, reuse = True, scope = MLO_scope, biases_initializer = biases_initializer)

	h = (h_L_CC, h_R_CC, h_L_MLO, h_R_MLO)

	return h

def all_views_max_pool(input_layer, stride = [2, 2]):

	input_L_CC, input_R_CC, input_L_MLO, input_R_MLO = input_layer

	output_L_CC = tf.nn.max_pool(input_L_CC, ksize = [1, stride[0], stride[1], 1], strides = [1, stride[0], stride[1], 1], padding = 'SAME')
	output_R_CC = tf.nn.max_pool(input_R_CC, ksize = [1, stride[0], stride[1], 1], strides = [1, stride[0], stride[1], 1], padding = 'SAME')
	output_L_MLO = tf.nn.max_pool(input_L_MLO, ksize = [1, stride[0], stride[1], 1], strides = [1, stride[0], stride[1], 1], padding = 'SAME')
	output_R_MLO = tf.nn.max_pool(input_R_MLO, ksize = [1, stride[0], stride[1], 1], strides = [1, stride[0], stride[1], 1], padding = 'SAME')

	output = (output_L_CC, output_R_CC, output_L_MLO, output_R_MLO)

	return output

def all_views_global_avg_pool(input_layer):

	input_L_CC, input_R_CC, input_L_MLO, input_R_MLO = input_layer

	input_layer_shape = input_L_CC.get_shape()
	pooling_shape = [1, input_layer_shape[1], input_layer_shape[2], 1]

	output_L_CC = tf.nn.avg_pool(input_L_CC, ksize = pooling_shape, strides = pooling_shape, padding = 'SAME')
	output_R_CC = tf.nn.avg_pool(input_R_CC, ksize = pooling_shape, strides = pooling_shape, padding = 'SAME')
	output_L_MLO = tf.nn.avg_pool(input_L_MLO, ksize = pooling_shape, strides = pooling_shape, padding = 'SAME')
	output_R_MLO = tf.nn.avg_pool(input_R_MLO, ksize = pooling_shape, strides = pooling_shape, padding = 'SAME')

	output = (output_L_CC, output_R_CC, output_L_MLO, output_R_MLO)

	return output

def all_views_flattening_layer(input_layer):

	input_L_CC, input_R_CC, input_L_MLO, input_R_MLO = input_layer

	input_layer_shape = input_L_CC.get_shape()
	input_layer_size = int(input_layer_shape[1]) * int(input_layer_shape[2]) * int(input_layer_shape[3])

	h_L_CC_flat = tf.reshape(input_L_CC, [-1, input_layer_size])
	h_R_CC_flat = tf.reshape(input_R_CC, [-1, input_layer_size])
	h_L_MLO_flat = tf.reshape(input_L_MLO, [-1, input_layer_size])
	h_R_MLO_flat = tf.reshape(input_R_MLO, [-1, input_layer_size])

	h_flat = tf.concat(axis = 1, values = [h_L_CC_flat, h_R_CC_flat, h_L_MLO_flat, h_R_MLO_flat])

	return h_flat

def fc_layer(input_layer, number_of_units = 128, activation_fn = tf.nn.relu, reuse = None, scope = None):

	h = tf.contrib.layers.fully_connected(inputs = input_layer, num_outputs = number_of_units, activation_fn = activation_fn, reuse = reuse, scope = scope)

	return h

def softmax_layer(input_layer, number_of_outputs = 4):

	with tf.variable_scope('fully_connected_2') as fully_scope:

		y_prediction = tf.contrib.layers.fully_connected(inputs = input_layer, num_outputs = number_of_outputs, activation_fn = tf.nn.softmax, scope = fully_scope)

	return y_prediction

def dropout_layer(input_layer, nodropout_probability):
	
	output = tf.nn.dropout(input_layer, nodropout_probability)
	
	return output

def Gaussian_noise_layer(input_layer, std):
	
	noise = tf.random_normal(tf.shape(input_layer), mean = 0.0, stddev = std, dtype = tf.float32)
	
	output = tf.add_n([input_layer, noise])

	return output

def all_views_Gaussian_noise_layer(input_layer, std):
	
	input_L_CC, input_R_CC, input_L_MLO, input_R_MLO = input_layer
	
	output_L_CC = Gaussian_noise_layer(input_L_CC, std)
	output_R_CC = Gaussian_noise_layer(input_R_CC, std)
	output_L_MLO = Gaussian_noise_layer(input_L_MLO, std)
	output_R_MLO = Gaussian_noise_layer(input_R_MLO, std)

	output = (output_L_CC, output_R_CC, output_L_MLO, output_R_MLO)

	return output
