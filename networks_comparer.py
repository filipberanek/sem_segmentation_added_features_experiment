import keras
import tensorflow as tf

import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm

from models.data_reader import CustomDataGen
from models.segnet import Segnet
from models.evaluator import EvaluateModel

print(f"List of available devices {tf.config.list_physical_devices()}")

TRAIN_DATA_PATH = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/soiling_dataset/train")
VAL_DATA_PATH = pathlib.Path("/home/fberanek/Desktop/datasets/segmentation/semantic/soiling_dataset/test")
OUTPUT_MODEL_PATH = pathlib.Path("./trained_models")
OUTPUT_STATS = pathlib.Path("./stats")
IMAGE_SIZE = 480 # MUST BE SAME AS USED FOR FEATURE_ADDER IMG SIZE AND MODEL SIZE MUST BE SAME
BATCH_SIZE = 4
N_OF_CLASSES = 4
EPOCHS = 200
DICT_OF_INPUT_FOLDERS = {
                          #"original_images":"rgbImagesOriginal",
                          #"with_added_canny_edges":"rgbImagesAddedCanny",
                          #"with_added_harris_edges":"rgbImagesAddedHarris", 
                          "with_added_sift":"rgbImagesAddedSift",
                          "with_added_sobel_edges":"rgbImagesAddedSobel"
                          }
LABELS_FOLDER = "gtLabels"
RGB_LABELS_FOLDER = "rgbLabels"
VAL_NUMBER_OF_FILES = 800
DEVICE_TO_USE = "/GPU:0"

if __name__ == "__main__":
    with tf.device(DEVICE_TO_USE):
        for dataset_key, dataset_value in tqdm(DICT_OF_INPUT_FOLDERS.items()):
            # Create output folders 
            model_output_folder = OUTPUT_MODEL_PATH / dataset_key
            model_output_folder.mkdir(exist_ok=True, parents=True)
            stats_output_folder = OUTPUT_STATS / dataset_key
            stats_output_folder.mkdir(exist_ok=True, parents=True)
            # Preprocess files for datasets
            train_images = sorted(list((TRAIN_DATA_PATH / dataset_value).rglob("*")))
            train_masks = sorted(list((TRAIN_DATA_PATH / LABELS_FOLDER).rglob("*")))
            val_images = sorted(list((VAL_DATA_PATH / dataset_value).rglob("*")))[:VAL_NUMBER_OF_FILES]
            val_masks = sorted(list((VAL_DATA_PATH / LABELS_FOLDER).rglob("*")))[:VAL_NUMBER_OF_FILES]
            test_images = sorted(list((VAL_DATA_PATH / dataset_value).rglob("*")))[VAL_NUMBER_OF_FILES:]
            test_masks = sorted(list((VAL_DATA_PATH / LABELS_FOLDER).rglob("*")))[VAL_NUMBER_OF_FILES:]
            df_train = pd.DataFrame.from_dict({"image_paths":train_images,"label_paths":train_masks})
            df_val = pd.DataFrame.from_dict({"image_paths":val_images,"label_paths":val_masks})
            df_test = pd.DataFrame.from_dict({"image_paths":test_images,"label_paths":test_masks})
            train_data_gen = CustomDataGen(df_train,BATCH_SIZE,(IMAGE_SIZE, IMAGE_SIZE),N_OF_CLASSES)
            val_data_gen = CustomDataGen(df_val,BATCH_SIZE,(IMAGE_SIZE, IMAGE_SIZE),N_OF_CLASSES)
            test_data_gen = CustomDataGen(df_test,1,(IMAGE_SIZE, IMAGE_SIZE),N_OF_CLASSES)
            # Get new and fresh instance of model
            N_INPUT_FEATURES = np.load(str(df_train.iloc[0]["image_paths"]), allow_pickle=True)["data"].shape[-1]
            model = Segnet(N_OF_CLASSES, IMAGE_SIZE, IMAGE_SIZE, N_INPUT_FEATURES).get_model()
            # Define attributes of training
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                                    factor=0.2, 
                                                    patience=5, 
                                                    min_lr=1e-13, 
                                                    min_delta = 1e-13, 
                                                    mode = "auto"), 
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                patience=10, 
                                                min_delta=1e-13, 
                                                restore_best_weights=True), 
                tf.keras.callbacks.ModelCheckpoint(filepath=str(model_output_folder / "./best_model.h5"), 
                                                monitor='val_accuracy', 
                                                mode='max', 
                                                save_best_only=True)
            ]
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=loss,
                metrics=["accuracy"])
            training_history = model.fit(
                train_data_gen,
                epochs=EPOCHS,
                validation_data=val_data_gen,
                callbacks=callbacks
            )
            pd.DataFrame.from_dict(training_history.history).to_csv(str(model_output_folder / "./traning_metrics.csv"))
            model = keras.models.load_model(str(model_output_folder / "./best_model.h5"))
            EvaluateModel(model, test_data_gen, N_OF_CLASSES, stats_output_folder).evalute()