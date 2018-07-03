import argparse
import torch
import tensorflow as tf

import models_torch


def histogram_tf_to_torch(input_path, output_path):
    g = tf.Graph()
    histogram_model = models_torch.BaselineHistogramModel(num_bins=50)
    with tf.Session(graph=g) as sess:
        saver = tf.train.import_meta_graph(input_path + ".meta")
        saver.restore(sess, input_path)
        histogram_model.fc1.weight.data = torch.Tensor(sess.run(g.get_tensor_by_name("fully_connected/weights:0")).T)
        histogram_model.fc1.bias.data = torch.Tensor(sess.run(g.get_tensor_by_name("fully_connected/biases:0")))
        histogram_model.fc2.weight.data = torch.Tensor(sess.run(g.get_tensor_by_name("fully_connected_2/weights:0")).T)
        histogram_model.fc2.bias.data = torch.Tensor(sess.run(g.get_tensor_by_name("fully_connected_2/biases:0")))
        torch.save(histogram_model.state_dict(), output_path)


def cnn_tf_to_torch(input_path, output_path):
    g = tf.Graph()
    device = torch.device("cpu")
    bbmodel = models_torch.BaselineBreastModel(device, nodropout_probability=1.0)
    with tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.import_meta_graph(input_path + ".meta")
        saver.restore(sess, input_path)
        var_dict = {
            var.name: var
            for var in g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        }
        for conv_name, conv_layer in bbmodel.conv_layer_dict.items():
            for view in ["CC", "MLO"]:
                conv_layer.ops[view].weight.data = torch.Tensor(sess.run(
                    var_dict["{}_{}/weights:0".format(conv_name, view)]
                )).permute(3, 2, 0, 1)
                conv_layer.ops[view].bias.data = torch.Tensor(sess.run(
                    var_dict["{}_{}/biases:0".format(conv_name, view)]
                ))
        bbmodel.fc1.weight.data = torch.Tensor(sess.run(
            var_dict["fully_connected/weights:0"]
        ).T)
        bbmodel.fc1.bias.data = torch.Tensor(sess.run(
            var_dict["fully_connected/biases:0"]
        ))
        bbmodel.fc2.weight.data = torch.Tensor(sess.run(
            var_dict["fully_connected_2/weights:0"]
        ).T)
        bbmodel.fc2.bias.data = torch.Tensor(sess.run(
            var_dict["fully_connected_2/biases:0"]
        ))
        torch.save(bbmodel.state_dict(), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert from TensorFlow checkpoints to PyTorch pickles')
    parser.add_argument('model_type')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    if args.model_type == "histogram":
        histogram_tf_to_torch(args.input_path, args.output_path)
    elif args.model_type == "cnn":
        cnn_tf_to_torch(args.input_path, args.output_path)
    else:
        raise RuntimeError(args.model_type)
