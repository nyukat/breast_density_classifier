import argparse
import numpy as np


MODEL_PATH_DICT = {
    "cnn": {
        "tf": "saved_models/BreastDensity_BaselineBreastModel/model.ckpt",
        "torch": "saved_models/BreastDensity_BaselineBreastModel/model.p",
    },
    "histogram": {
        "tf": "saved_models/BreastDensity_BaselineHistogramModel/model.ckpt",
        "torch": "saved_models/BreastDensity_BaselineHistogramModel/model.p",
    },
}


def get_result(library, device_type, model_type):

    if library == "tf":
        import density_model_tf
        inference_func = density_model_tf.inference
    elif library == "torch":
        import density_model_torch
        inference_func = density_model_torch.inference
    else:
        raise RuntimeError(library)

    return inference_func({
        "model_type": model_type,
        "model_path": MODEL_PATH_DICT[model_type][library],
        "device_type": device_type,
        "gpu_number": 0,
        "image_path": "images/",
        "input_size": (2600, 2000),
        "bins_histogram": 50,
    }, verbose=False)


GOLDEN_RESULT = {
    "histogram": (0.0819444, 0.78304, 0.133503, 0.00151265),
    "cnn": (0.209689, 0.765076, 0.024949, 0.000285853),
}


# CPU-GOLDEN Consistency
def test_tf_golden_equal_cnn():
    assert np.allclose(get_result("tf", "cpu", "cnn"), GOLDEN_RESULT["cnn"])


def test_torch_golden_equal_cnn():
    assert np.allclose(get_result("torch", "cpu", "cnn"), GOLDEN_RESULT["cnn"])


def test_tf_golden_equal_histogram():
    assert np.allclose(get_result("tf", "cpu", "histogram"), GOLDEN_RESULT["histogram"])


def test_torch_golden_equal_histogram():
    assert np.allclose(get_result("torch", "cpu", "histogram"), GOLDEN_RESULT["histogram"])


# CPU-GPU Consistency
def test_tf_cpu_gpu_equal_cnn():
    assert np.allclose(get_result("tf", "cpu", "cnn"), get_result("tf", "gpu", "cnn"))


def test_torch_cpu_gpu_equal_cnn():
    assert np.allclose(get_result("torch", "cpu", "cnn"), get_result("torch", "gpu", "cnn"))


def test_tf_cpu_gpu_equal_histogram():
    assert np.allclose(get_result("tf", "cpu", "histogram"), get_result("tf", "gpu", "histogram"))


def test_torch_cpu_gpu_equal_histogram():
    assert np.allclose(get_result("torch", "cpu", "histogram"), get_result("torch", "gpu", "histogram"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Tests')
    parser.add_argument('--using')
    parser.add_argument('--with-gpu', action="store_true")
    args = parser.parse_args()

    test_list = []
    if args.using == "tf":
        test_list.append(test_tf_golden_equal_cnn)
        test_list.append(test_tf_golden_equal_histogram)
        if args.with_gpu:
            test_list.append(test_tf_cpu_gpu_equal_cnn)
            test_list.append(test_tf_cpu_gpu_equal_histogram)
    elif args.using == "torch":
        test_list.append(test_torch_golden_equal_cnn)
        test_list.append(test_torch_golden_equal_histogram)
        if args.with_gpu:
            test_list.append(test_torch_cpu_gpu_equal_cnn)
            test_list.append(test_torch_cpu_gpu_equal_histogram)
    else:
        raise RuntimeError("Provide --using 'tf' or 'torch'")

    for test_func in test_list:
        try:
            test_func()
            print("{}: PASSED".format(test_func.__name__))
        except Exception as e:
            print("{}: FAILED".format(test_func.__name__))
            raise

    print("All {} test(s) passed.".format(len(test_list)))
