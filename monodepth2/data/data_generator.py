import os
import numpy as np
import tensorflow as tf

from monodepth2.utils.data_utils import image_mean
from monodepth2.utils.kitti_utils import generate_depth_map

class DataGenerator():
    def __init__(self, 
                 data_path, 
                 mode="train", 
                 batch_size=1, 
                 dim=(375, 1242), 
                 n_channels=3, 
                 shuffle=True, 
                 resize_dim=(192, 640)
        ):
        self.dim = dim
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
            x, y = self.load()
            yield x, y

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=self.n_channels)
        image.set_shape([None, None, 3])
        if self.mode == "train":
            image = tf.image.resize(image,size=self.resize_dim,
                                       method=tf.image.ResizeMethod.BILINEAR)
            image = tf.image.random_crop(image,size=(self.resize_dim[0],self.resize_dim[1],3))
        image = image/255.0
        image = tf.cast(image,tf.float32)
        return image
    
    def get_path(files):
        file_names = dict()
        side_conversion = 'image_02' if side == 'l' else 'image_03'
        
        with open(files, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line_split = line.split(' ')
            folder = line_split[0]
            frame_idx = int(line_split[1])
            side = line_split[2].rstrip() 
