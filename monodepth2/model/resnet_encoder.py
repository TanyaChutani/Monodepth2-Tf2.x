import tensorflow as tf

from monodepth2.utils.encoder_utils import ResNet_Block

class ResNet_Encoder(tf.keras.models.Model):
    def __init__(self,
                n_blocks=2,
                downsample=True):
        super(ResNet_Encoder, self).__init__()
        self.n_blocks = n_blocks
        self.downsample = downsample
        self.conv_layer = tf.keras.layers.Conv2D(64, 
                                                 7, 
                                                 strides=2,
                                                 padding='same',
                                                 use_bias=False)
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.relu_layer = tf.keras.layers.ReLU()
        self.maxpool_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                       strides=2,
                                                       padding="same")
        self.make_resnet_block = ResNet_Block(64, self.downsample)
        
    def make_encoder_block(self, filters, downsample):
        label = []
        for idx in range(self.n_blocks):
            label.append(ResNet_Block(filters, downsample)) if idx == 0 \
            else label.append(ResNet_Block(filters, False))
        return tf.keras.Sequential(label)
    
    def call(self,
             input_tensor,
             training=None):
        x = self.conv_layer(input_tensor)
        x = self.bn_layer(x, training=training) 
        x = self.relu_layer(x)
        x = self.maxpool_layer(x)
        x = self.make_encoder_block(64, False)(x)
        x = self.make_encoder_block(128, self.downsample)(x)
        x = self.make_encoder_block(256, self.downsample)(x)
        x = self.make_encoder_block(512, self.downsample)(x)
        return x
