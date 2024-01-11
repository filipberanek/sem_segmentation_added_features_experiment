import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import pathlib
from models.data_reader import CustomDataGen

class EvaluateModel:
    def __init__(self, model:keras.src.engine.functional.Functional, 
                 dataset:CustomDataGen, 
                 n_classes:int, 
                 output_path:pathlib.Path) -> None:
        self.model = model 
        self.dataset = dataset
        self.n_classes = n_classes
        self.output_path = output_path
        self.conf_matrx = {}
        self.prefix = "class_"
        for class_id in range(n_classes):
            self.conf_matrx[self.prefix+str(class_id)] ={
                "tp":0,
                "fp":0,
                "tn":0,
                "fn":0, 
                "total":0
            }
    
    def infer(self, image_tensor):
        predictions = self.model.predict(np.expand_dims((image_tensor), axis=0))
        predictions = np.squeeze(predictions)
        predictions = np.argmax(predictions, axis=2)
        return predictions

    def calculate_tp(self, prediction_mask, label_mask, class_id):
        return np.logical_and(prediction_mask==class_id, 
                       label_mask==class_id)

    def calculate_fp(self, prediction_mask, label_mask, class_id):
        return np.logical_and(prediction_mask==class_id, 
                       label_mask!=class_id)
    
    def calculate_tn(self, prediction_mask, label_mask, class_id):
        return np.logical_and(prediction_mask!=class_id, 
                       label_mask!=class_id)

    def calculate_fn(self, prediction_mask, label_mask, class_id):
        return np.logical_and(prediction_mask!=class_id, 
                       label_mask==class_id)

    def evalute(self):
        for image_id in range(self.dataset.__len__()):
            image_tensor, label_tensor = self.dataset.__getitem__(image_id)
            label_tensor = tf.squeeze(label_tensor)
            prediction_mask = self.infer(image_tensor=tf.squeeze(image_tensor))
            for seg_class in self.conf_matrx:
                class_id = int(seg_class.replace(self.prefix, ""))
                tp = self.calculate_tp(prediction_mask, label_tensor, class_id)
                tn = self.calculate_tn(prediction_mask, label_tensor, class_id)
                fp = self.calculate_fp(prediction_mask, label_tensor, class_id)
                fn = self.calculate_fn(prediction_mask, label_tensor, class_id)
                self.conf_matrx[seg_class]["tp"] +=np.sum(tp)
                self.conf_matrx[seg_class]["tn"] +=np.sum(tn)
                self.conf_matrx[seg_class]["fp"] +=np.sum(fp)
                self.conf_matrx[seg_class]["fn"] +=np.sum(fn)
                self.conf_matrx[seg_class]["total"] += len(prediction_mask.reshape(-1))
        pd.DataFrame(self.conf_matrx).to_csv(self.output_path / "stats_absolute_numbers.csv")
        df = pd.DataFrame(self.conf_matrx)
        (df / df.loc["total"]).to_csv(self.output_path / "stats_relative_numbers.csv")