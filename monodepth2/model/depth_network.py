import tensorflow as tf

from monodepth2.utils.encoder_utils import ResNet_Block
from monodepth2.utils.decoder_utils import Decoder_Block


class Depth_Estimation(tf.keras.models.Model):
    def __init__(self, n_blocks=2, downsample=True, filters=64):
        super(Depth_Estimation, self).__init__()
        self.n_blocks = n_blocks
        self.downsample = downsample
        self.filters = filters
        self.conv_layer = tf.keras.layers.Conv2D(
            self.filters, 7, strides=2, padding="same", use_bias=False
        )
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.relu_layer = tf.keras.layers.ReLU()
        self.maxpool_layer = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=2, padding="same"
        )
        self.conv_decoder_layer = tf.keras.layers.Conv2D(
            1, 3, padding="same", activation="sigmoid"
        )
        self.conv2_decoder_layer = tf.keras.layers.Conv2D(
            1, 3, padding="same", activation="sigmoid"
        )
        self.conv3_decoder_layer = tf.keras.layers.Conv2D(
            1, 3, padding="same", activation="sigmoid"
        )
        self.conv4_decoder_layer = tf.keras.layers.Conv2D(
            1, 3, padding="same", activation="sigmoid"
        )
        self.conv5_decoder_layer = tf.keras.layers.Conv2D(
            1, 3, padding="same", activation="sigmoid"
        )

    def make_encoder_block(self, filters, downsample):
        label = []
        for idx in range(self.n_blocks):
            label.append(
                ResNet_Block(filters, downsample)
            ) if idx == 0 else label.append(ResNet_Block(filters, False))
        return tf.keras.Sequential(label)

    def call(self, input_tensor, training=None):
        multi_scale = []

        x = self.conv_layer(input_tensor)
        x = self.bn_layer(x, training=training)
        x1 = self.relu_layer(x)
        x = self.maxpool_layer(x1)
        x2 = self.make_encoder_block(self.filters, False)(x)
        x3 = self.make_encoder_block(self.filters * 2, self.downsample)(x2)
        x4 = self.make_encoder_block(self.filters * 4, self.downsample)(x3)
        x5 = self.make_encoder_block(self.filters * 8, self.downsample)(x4)

        x = Decoder_Block(self.filters * 4, x4)(x5)
        multi_scale.append(self.conv_decoder_layer(x))
        x = Decoder_Block(self.filters * 2, x3)(x)
        multi_scale.append(self.conv2_decoder_layer(x))
        x = Decoder_Block(self.filters, x2)(x)
        multi_scale.append(self.conv3_decoder_layer(x))
        x = Decoder_Block(self.filters // 2, x1)(x)
        multi_scale.append(self.conv4_decoder_layer(x))
        x = Decoder_Block(self.filters // 4, None)(x)
        multi_scale.append(self.conv5_decoder_layer(x))
        return multi_scale
