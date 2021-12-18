import tensorflow as tf

class Decoder_Block(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 concat):
        super(Decoder_Block,self).__init__()
        self.filters = filters
        self.concat = concat
        self.conv_layer = tf.keras.layers.Conv2D(self.filters, 
                                                 3, 
                                                 padding='same',
                                                 use_bias=False)
        self.elu_layer = tf.keras.layers.ELU()
        self.tconv_layer = tf.keras.layers.UpSampling2D()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.conv2_layer = tf.keras.layers.Conv2D(self.filters, 
                                                 3, 
                                                 padding='same',
                                                 use_bias=False)
        
    def call(self,
             input_tensor,
             training=None):
        x = self.conv_layer(input_tensor)
        x = self.elu_layer(x)
        x = self.tconv_layer(x)
        if self.concat is not None:
            x = self.concat_layer([x, self.concat])
        x = self.conv2_layer(x)
        x = self.elu_layer(x)
        return x
