import tensorflow as tf


from utils.loss_utils import Reflection_Pad

class SSIM_Loss(tf.keras.losses.Loss):
    def __init__(self,
                 **kwargs):
        super(SSIM_Loss,self).__init__()
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
        self.reflection_pad = Reflection_Pad(pad=(1,1))
        
        self.avg_pool_layer1 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                                strides=(1, 1),
                                                                padding='valid')
        self.avg_pool_layer2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                                strides=(1, 1),
                                                                padding='valid')
        self.avg_pool_layer3 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                                strides=(1, 1),
                                                                padding='valid')
        self.avg_pool_layer4 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                                strides=(1, 1),
                                                                padding='valid')
        self.avg_pool_layer5 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                                strides=(1, 1),
                                                                padding='valid')
        self.avg_pool_layer6 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                                strides=(1, 1),
                                                                padding='valid')
        
    def __call__(self, predicted_image, output_image):
        pred = self.reflection_pad(predicted_image)
        op = self.reflection_pad(output_image)
        
        mean_pred = self.avg_pool_layer1(pred)
        mean_op = self.avg_pool_layer2(op)
        
        std_pred = self.avg_pool_layer3(pred ** 2) - mean_pred ** 2
        std_op = self.avg_pool_layer4(op ** 2) - mean_op ** 2
        std_pred_op = self.avg_pool_layer4(pred ** op) - mean_pred ** mean_op
        
        numerator = (2 * mean_pred * mean_op + self.C1) * (2 * std_pred_op + self.C2)
        denom = (mean_pred ** 2 + mean_op ** 2 + self.C1) * (std_pred + std_pred + self.C2)
        
        ssim = (1 - numerator/denom)
        return tf.clip_by_value(ssim, clip_value_min = 0, clip_value_max = 1)
        

class Photometric_Loss(tf.keras.losses.Loss):
    def __init__(self,
                 alpha= 0.85,
                 **kwargs):
        super(Photometric_Loss,self).__init__()
        self.alpha = alpha
        self.ssim_loss = SSIM_Loss()
        self.l1_loss = tf.keras.losses.MeanAbsoluteError()
        
    def __call__(self, predicted_image, output_image):
        l1_loss = self.l1_loss(predicted_image, output_image)
        ssim_loss = tf.reduce_mean(self.ssim_loss(predicted_image, output_image),\
                                   axis=3, keepdims=True)
        return (self.alpha * ssim_loss) + ((1 - self.alpha) * l1_loss)
