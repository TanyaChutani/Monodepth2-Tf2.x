import tensorflow as tf

class ResNet_Block(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 downsample):
        super(ResNet_Block,self).__init__()
        self.filters = filters
        self.downsample = downsample
        self.strides = 2 if self.downsample else 1
        self.conv_layer = tf.keras.layers.Conv2D(self.filters, 
                                                 3, 
                                                 strides=self.strides,
                                                 padding='same',
                                                 use_bias=False)
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.relu_layer = tf.keras.layers.ReLU()
        self.conv2_layer = tf.keras.layers.Conv2D(self.filters, 
                                                 3, 
                                                 padding='same',
                                                 strides=1,
                                                 use_bias=False)
        self.bn2_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.conv3_layer = tf.keras.layers.Conv2D(self.filters, 
                                                 1,
                                                 strides=self.strides,
                                                 padding='same',
                                                 use_bias=False)

    def call(self,
             input_tensor,
             training=None):
        x = self.conv_layer(input_tensor)
        x = self.bn_layer(x, training=training) 
        x = self.relu_layer(x)
        x = self.conv2_layer(x)
        x = self.bn2_layer(x, training=training) 
        if self.downsample:
            y = self.conv3_layer(input_tensor)
            x = x + y
        else:
            x = x + input_tensor
        x = self.relu_layer(x)
        return x
