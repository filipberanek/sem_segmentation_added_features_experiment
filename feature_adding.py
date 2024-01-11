import pathlib

from feature_add import image_feature_adder

if __name__ == "__main__":
    input_path_train = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/soiling_dataset/train/rgbImages")
    input_path_test = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/soiling_dataset/test/rgbImages")
    output_path_train = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/soiling_dataset/train")
    output_path_test = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/soiling_dataset/test")
    image_size = 480
    image_feature_adder(input_path_train, output_path_train, image_size)
    image_feature_adder(input_path_test, output_path_test, image_size)