import os
import numpy as np
import tensorflow as tf

from utils.data_utils import image_mean
from utils.kitti_utils import generate_depth_map


class DataGenerator:
    def __init__(
        self,
        data_path,
        batch_size=3,
        n_channels=3,
        shuffle=True,
        resize_dim=(192, 640),
        base_path="/home/ubuntu/dataset",
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.resize_dim = resize_dim
        self.base_path = base_path
        self.file_names = self.get_path()
        self.index = np.arange(len(self.file_names))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        return np.ceil(len(self.file_names) / self.batch_size)

    def __call__(self):
        for i in self.index:
            prev_image, target_image, next_image, depth_image = self.load(
                self.file_names[i]
            )
            yield prev_image, target_image, next_image, depth_image

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=self.n_channels)
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            image, size=self.resize_dim, method=tf.image.ResizeMethod.BILINEAR
        )
        image = tf.image.random_crop(
            image, size=(self.resize_dim[0], self.resize_dim[1], 3)
        )
        image = image / 255.0
        image = tf.cast(image, tf.float32)
        return image

    def get_path(self):
        side_conversion = {"l": "image_02", "r": "image_02"}

        with open(self.data_path, "r") as f:
            lines = f.readlines()
        final = []
        for line in lines:
            file_names = dict()
            line_split = line.split(" ")
            folder = line_split[0]
            frame_idx = int(line_split[1])
            side = line_split[2].rstrip()
            file_names["side"] = side
            file_names["prev_image_paths"] = os.path.join(
                self.base_path,
                folder,
                side_conversion[side],
                "data",
                "{:010d}.png".format((frame_idx - 1)),
            )
            file_names["target_image_paths"] = os.path.join(
                self.base_path,
                folder,
                side_conversion[side],
                "data",
                "{:010d}.png".format(frame_idx),
            )
            file_names["next_image_paths"] = os.path.join(
                self.base_path,
                folder,
                side_conversion[side],
                "data",
                "{:010d}.png".format((frame_idx + 1)),
            )
            file_names["calib_paths"] = os.path.join(
                self.base_path, os.path.dirname(folder)
            )
            file_names["velo_paths"] = os.path.join(
                self.base_path,
                folder,
                "velodyne_points",
                "data",
                "{:010d}.bin".format(frame_idx),
            )
            final.append(file_names)
        return final

    def get_depth(self, file_name):
        if not os.path.isfile(file_name["velo_paths"]):
            return tf.zeros((self.resize_dim[0], self.resize_dim[1]), dtype=tf.float32)
        cam = 2 if file_name["side"] == "l" else 3
        depth_image = generate_depth_map(
            file_name["calib_paths"], file_name["velo_paths"], cam
        )
        depth_image = tf.expand_dims(depth_image, axis=-1)

        depth_image = tf.image.resize(
            depth_image, size=self.resize_dim, method=tf.image.ResizeMethod.BILINEAR
        )
        depth_image = tf.cast(depth_image, tf.float32)
        return depth_image

    def load(self, file_name):
        prev_image = tf.io.read_file(file_name["prev_image_paths"])
        target_image = tf.io.read_file(file_name["target_image_paths"])
        next_image = tf.io.read_file(file_name["next_image_paths"])

        prev_image = self.preprocess_image(prev_image)
        target_image = self.preprocess_image(target_image)
        next_image = self.preprocess_image(next_image)

        img_mean = image_mean()

        prev_image -= img_mean
        target_image -= img_mean
        next_image -= img_mean

        depth_image = self.get_depth(file_name)

        return prev_image, target_image, next_image, depth_image
