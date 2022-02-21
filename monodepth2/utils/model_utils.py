import tensorflow as tf

def disp_to_depth(disparity, max_depth = 100.0, min_depth=0.1):
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disparity
    depth = 1.0 / scaled_disp
    return depth

def rot_from_axisangle(vec):
    angle = tf.norm(vec, axis=1, keepdims=True) 
    axis = vec / (angle + 1e-7) 

    angle = tf.squeeze(angle, axis=1)  
    ca = tf.cos(angle) 
    sa = tf.sin(angle) 
    C = 1.0 - ca
    
    x = axis[:, 0]  
    y = axis[:, 1]  
    z = axis[:, 2] 

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    
    col0 = tf.stack([(x * xC + ca), (xyC + zs), (zxC - ys)], axis=1)
    col1 = tf.stack([(xyC - zs), ( y * yC + ca), (yzC + xs)], axis=1) 
    col2 = tf.stack([(zxC + ys), (yzC - xs), (z * zC + ca)], axis=1) 
    rot = tf.stack([col0, col1, col2], axis=2) 
    rot = tf.reshape(rot, [-1,4,4])
    return rot

def get_translation_matrix(batch_size, trans_vector):
    ones = tf.ones([batch_size,1,1], dtype=tf.float32)
    zeros = tf.zeros([batch_size,1,1], dtype=tf.float32)

    T = tf.concat([
        ones, zeros, zeros, trans_vector[:,:, :1],
        zeros, ones, zeros, trans_vector[:,:,1:2],
        zeros, zeros, ones, trans_vector[:,:, 2:3],
        zeros, zeros, zeros, ones

    ], axis=2)

    T = tf.reshape(T,[-1, 4, 4])
    return T

def transformation_from_parameters(raw_transformation, invert=False):
    axisangle = raw_transformation[:, :3]
    translation = raw_transformation[:, 3:]
    batch_size = tf.shape(translation)[0]
    
    rot = rot_from_axisangle(axisangle)
    translation = tf.reshape(translation, [batch_size, 1, 3])
    t = -tf.ones_like(translation)
    
    if invert:
        rot = tf.transpose(rot, perm=[0, 2, 1])
        translation = tf.math.multiply(t, translation)
    T = get_translation_matrix(batch_size, translation)

    if invert:
        M = tf.linalg.matmul(rot, T)
    else:
        M = tf.linalg.matmul(T, rot)

    return M
