import tensorflow as tf
import numpy as np
from PIL import Image

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, 
                 files_dataframe,
                 batch_size,
                 dim,
                 num_classes,
                 shuffle=True):
        
        self.df = files_dataframe.copy()
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.len_of_set = len(self.df)

    def __len__(self):
        return (self.len_of_set // self.batch_size) -1 
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_data(self, batches):
        n_channels = np.load(batches.iloc[0]["image_paths"])["data"].shape[-1]
        X = np.empty((self.batch_size, *self.dim, n_channels))
        y = np.empty((self.batch_size,*self.dim), dtype=int)
        for row_id,(df_id,files) in enumerate(batches.iterrows()):
            X[row_id,] = np.load(str(files["image_paths"]),allow_pickle=True)["data"]
            y[row_id] = np.array(Image.open(str(files["label_paths"])).resize((self.dim)))
        X = X/255
        y = tf.convert_to_tensor(y)
        #y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        X = tf.convert_to_tensor(X)
        return X, y
    
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    

    
    

