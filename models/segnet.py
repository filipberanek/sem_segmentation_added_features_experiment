from keras_segmentation.models.segnet import resnet50_segnet
import tensorflow as tf
from keras import Model
from keras import layers
import numpy as np

class Segnet:
    def __init__(self, n_classes, image_height, image_width, n_input_features = 3) -> None:
        self.n_classes = n_classes
        self.image_height = image_height
        self.image_width = image_width
        self.n_input_features = n_input_features
    
    def get_model(self):
        backbone = resnet50_segnet(n_classes=self.n_classes, 
                                input_height=self.image_height,
                                input_width=self.image_width)
        cut_layer_idx = -3
        cut_layer_name = backbone.layers[cut_layer_idx].name
        model_cutted = Model(backbone.input, outputs=backbone.get_layer(cut_layer_name).output)
        model_config = model_cutted.get_config()
        model_config["layers"][0]["config"]["batch_input_shape"] = (None, 
                                                                    self.image_width, 
                                                                    self.image_height, 
                                                                    self.n_input_features)
        modified_model = tf.keras.Model.from_config(model_config)
        model_extended = layers.UpSampling2D((2,2),interpolation="bilinear")(modified_model.output)
        model_output = layers.Conv2D(self.n_classes, kernel_size=(1,1), activation="softmax", padding="same")(model_extended)
        model = Model(modified_model.input, model_output)
        print(model.summary())
        return model
    
    def infer(model, image_tensor):
        predictions = model.predict(np.expand_dims((image_tensor), axis=0))
        predictions = np.squeeze(predictions)
        predictions = np.argmax(predictions, axis=2)
        return predictions