import tensorflow as tf

from monodepth2.utils.encoder_utils import ResNet_Block

class PoseNet(tf.keras.models.Model):
    def __init__(self,
                n_blocks=2,
                downsample=True,
                filters=64):
        super(PoseNet, self).__init__()
        self.n_blocks = n_blocks
        self.downsample = downsample
        self.filters = filters
        self.conv_layer = tf.keras.layers.Conv2D(self.filters, 
                                                 7, 
                                                 strides=2,
                                                 padding='same',
                                                 use_bias=False)
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.relu_layer = tf.keras.layers.ReLU()
        self.maxpool_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                       strides=2,
                                                       padding="same")
    
        self.conv_posenet_layer = tf.keras.layers.Conv2D(self.filters * 4, 
                                                         1)
        self.conv2_posenet_layer = tf.keras.layers.Conv2D(self.filters * 4, 
                                                          3, 
                                                          padding='same')
        self.conv3_posenet_layer = tf.keras.layers.Conv2D(self.filters * 4, 
                                                          3, 
                                                 padding='same')
        self.conv4_posenet_layer = tf.keras.layers.Conv2D(6, 
                                                          1)

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
        x = self.make_encoder_block(self.filters, False)(x)
        x = self.make_encoder_block(self.filters * 2, self.downsample)(x)
        x = self.make_encoder_block(self.filters * 4, self.downsample)(x)
        x = self.make_encoder_block(self.filters * 8, self.downsample)(x)
        
        x = self.conv_posenet_layer(x)
        x = self.relu_layer(x)
        x = self.conv2_posenet_layer(x)
        x = self.relu_layer(x)
        x = self.conv3_posenet_layer(x)
        x = self.relu_layer(x)
        x = self.conv4_posenet_layer(x)
        x = tf.math.reduce_mean(x, axis = [1, 2])
        return x
