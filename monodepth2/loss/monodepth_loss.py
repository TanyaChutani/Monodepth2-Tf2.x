import tensorflow as tf
import numpy as np

from utils.loss_utils import Depth_to_PointCloud, PointCloud_to_Pixel
from utils.model_utils import disp_to_depth
from loss.smooth import Edge_Aware_Smooth
from loss.photometric_error import Photometric_Loss


class Monodepth_Loss:
    def __init__(
        self,
        image_width=640,
        image_height=192,
        batch_size=3,
        smoothness_factor=1e-3,
        num_scales=5,
        K=1,
        **kwargs
    ):
        super(Monodepth_Loss, self).__init__()

        self.image_width = image_width
        self.image_height = image_height
        self.num_scales = num_scales
        self.batch_size = batch_size
        self.smoothness_factor = smoothness_factor
        self.edge_smooth_loss = Edge_Aware_Smooth()
        self.photometric_loss = Photometric_Loss()
        self.depth_to_point_layer = Depth_to_PointCloud(
            self.image_height, self.image_width, self.batch_size
        )
        self.K = np.array(
            [
                [0.58 * self.image_width, 0, 0.5 * self.image_width],
                [0, 1.92 * self.image_height, 0.5 * self.image_height],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.inverse_K = np.linalg.inv(self.K)
        self.pointclout_to_pixel_layer = PointCloud_to_Pixel(
            self.K, self.image_height, self.image_width, self.batch_size
        )

    def __call__(
        self,
        disps,
        pose_before_target,
        pose_after_target,
        prev_image,
        target_image,
        next_image,
    ):
        loss = tf.zeros((), tf.float32)
        for s in range(self.num_scales):

            scale_factor = 2.0 ** s
            h = self.image_height // scale_factor
            w = self.image_width // scale_factor
            disp = disps[s]
            disp_original = tf.image.resize(
                disp,
                size=(self.image_width, self.image_height),
                method=tf.image.ResizeMethod.BILINEAR,
            )
            depth = disp_to_depth(disp_original)

            camera_coords = self.depth_to_point_layer(depth, self.inverse_K)
            print("camera_coords", camera_coords)
            op_before_target = self.pointclout_to_pixel_layer(
                camera_coords, prev_image, pose_before_target
            )
            op_after_target = self.pointclout_to_pixel_layer(
                camera_coords, next_image, pose_after_target
            )

            pos_loss_before = self.photometric_loss(target_image, op_before_target)
            pos_loss_after = self.photometric_loss(target_image, op_after_target)

            identity_loss_before = self.photometric_loss(target_image, prev_image)
            identity_loss_after = self.photometric_loss(target_image, next_image)

            reprojection_losses = tf.stack(
                [
                    pos_loss_before,
                    pos_loss_after,
                    identity_loss_before,
                    identity_loss_after,
                ],
                axis=-1,
            )

            min_reprojection_loss = tf.reduce_min(reprojection_losses, axis=-1)

            loss += tf.reduce_mean(min_reprojection_loss)

            target_image_scaled = tf.image.resize(target_image, (h, w))
            smoothness_loss = self.edge_smooth_loss(disp, target_image_scaled)
            loss += smoothness_loss * self.smoothness_factor / scale_factor
            if scale_idx == 0:
                prev_image_op = op_before_target
                next_image_op = op_after_target

        loss /= float(num_scales)

        return loss, prev_image_op, next_image_op
