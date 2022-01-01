import tensorflow as tf

class Edge_Aware_Smooth(tf.keras.losses.Loss):
    def __init__(self,
                 **kwargs
                ):
        super(Edge_Aware_Smooth,self).__init__()
        
    def __call__(self,
                disparity, 
                image):
        mean_dispaarity = tf.reduce_mean(disparity,
                                         axis=[1, 2],
                                         keepdims=True)
        norm_disparity = disparity / (mean_dispaarity + 1e-7)
        
        grad_disp_x = tf.abs(norm_disparity[:, :, :-1, :] - norm_disparity[:, :, 1:, :])
        grad_disp_y = tf.abs(norm_disparity[:, :-1, :, :] - norm_disparity[:, 1:, :, :])

        grad_img_x = tf.reduce_mean(tf.abs(image[:, :, :-1, :] - image[:, :, 1:, :]),
                                    axis=-1,
                                    keepdims=True)
        grad_img_y = tf.reduce_mean(tf.abs(image[:, :-1, :, :] - image[:, 1:, :, :]),
                                    axis=-1,
                                    keepdims=True)

        grad_disp_x *= tf.exp(-grad_img_x) 
        grad_disp_y *= tf.exp(-grad_img_y)  

        loss_x = tf.reduce_mean(tf.squeeze(grad_disp_x, axis=3),
                                axis=[1, 2]) 
        loss_y = tf.reduce_mean(tf.squeeze(grad_disp_y, axis=3),
                                axis=[1, 2])

        return tf.reduce_mean(loss_x + loss_y)
