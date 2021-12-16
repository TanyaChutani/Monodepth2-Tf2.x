import tensorflow as tf

from monodepth2.utils.decoder_utils import Decoder_Block
from monodepth2.model.resnet_encoder import ResNet_Encoder

class Decoder(tf.keras.models.Model):
    def __init__(self,
                filters=64
        ):
        super(Decoder, self).__init__()
        self.filters = filters
        self.conv_layer = tf.keras.layers.Conv2D(1, 
                                                 3, 
                                                 padding='same',
                                                 activation='sigmoid')
        self.conv2_layer = tf.keras.layers.Conv2D(1, 
                                                 3, 
                                                 padding='same',
                                                 activation='sigmoid')
        self.conv3_layer = tf.keras.layers.Conv2D(1, 
                                                 3, 
                                                 padding='same',
                                                 activation='sigmoid')
