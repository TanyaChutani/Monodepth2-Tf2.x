import os
import numpy as np
import tensorflow as tf

class DataGenerator():
    def __init__(self, 
                 data_path, 
                 mode="train", 
                 batch_size=4, 
                 n_channels=3, 
                 shuffle=True, 
                 resize_dim=(640, 192)
        ):
        self.mode = mode
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.resize_dim = resize_dim  
        self.data_path = data_path
        self.data =  os.listdir(data_path)
        self.index = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        return (np.ceil(len(self.data) / self.batch_size))

    def __call__(self):
        for i in (self.index):
            x, y = self.load(os.path.join(self.data_path, 
                                          self.data[i]))
            yield x, y
