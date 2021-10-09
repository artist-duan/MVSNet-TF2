import tensorflow as tf

import sys
sys.path.append("../")
from models.module import *

class FeatureNet(tf.keras.Model):
    def __init__(self):
        super(FeatureNet,self).__init__()
        self.inplanes = 32

        self.net = tf.keras.Sequential()
        self.net.add(ConvBnReLU(3,   8, 3, 1, "SAME"))
        self.net.add(ConvBnReLU(8,   8, 3, 1, "SAME"))

        self.net.add(ConvBnReLU(8,  16, 5, 2, "SAME"))
        self.net.add(ConvBnReLU(16, 16, 3, 1, "SAME"))
        self.net.add(ConvBnReLU(16, 16, 3, 1, "SAME"))

        self.net.add(ConvBnReLU(16, 32, 5, 2, "SAME"))
        self.net.add(ConvBnReLU(32, 32, 3, 1, "SAME"))
        # self.net.add(ConvBnReLU(32, 32, 3, 1, "SAME"))
        self.net.add(tf.keras.layers.Conv2D(filters = 32,
                                           kernel_size = 3,
                                           strides = 1,
                                           padding = 'SAME',
                                           use_bias = False))

    def call(self, x):
        return self.net(x)

class FeatureNetGN(tf.keras.Model):
    def __init__(self):
        super(FeatureNetGN,self).__init__()
        self.inplanes = 32

        self.net = tf.keras.Sequential()
        self.net.add(ConvGnReLU(3,   8, 3, 1, "SAME"))
        self.net.add(ConvGnReLU(8,   8, 3, 1, "SAME"))

        self.net.add(ConvGnReLU(8,  16, 5, 2, "SAME"))
        self.net.add(ConvGnReLU(16, 16, 3, 1, "SAME"))
        self.net.add(ConvGnReLU(16, 16, 3, 1, "SAME"))

        self.net.add(ConvGnReLU(16, 32, 5, 2, "SAME"))
        self.net.add(ConvGnReLU(32, 32, 3, 1, "SAME"))
        # self.net.add(ConvGnReLU(32, 32, 3, 1, "SAME"))
        self.net.add(tf.keras.layers.Conv2D(filters = 32,
                                    kernel_size = 3,
                                    strides = 1,
                                    padding = 'SAME',
                                    use_bias = False))

    def call(self, x):
        return self.net(x)

class CostRegNet(tf.keras.Model):
    def __init__(self):
        super(CostRegNet,self).__init__()

        self.conv0 = ConvBnReLU3D(32,  8, 3, 1, "SAME")

        self.conv1 = ConvBnReLU3D(8,  16, 3, 2, "SAME")
        self.conv2 = ConvBnReLU3D(16, 16, 3, 1, "SAME")

        self.conv3 = ConvBnReLU3D(16, 32, 3, 2, "SAME")
        self.conv4 = ConvBnReLU3D(32, 32, 3, 1, "SAME")

        self.conv5 = ConvBnReLU3D(32, 64, 3, 2, "SAME")
        self.conv6 = ConvBnReLU3D(64, 64, 3, 1, "SAME")

        self.conv7 = tf.keras.Sequential()
        self.conv7.add(tf.keras.layers.Conv3DTranspose(filters = 32, kernel_size = 3, strides = 2, padding = "SAME", output_padding = 1, use_bias = False))
        self.conv7.add(tf.keras.layers.BatchNormalization())
        self.conv7.add(tf.keras.layers.ReLU())

        self.conv9 = tf.keras.Sequential()
        self.conv9.add(tf.keras.layers.Conv3DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = "SAME", output_padding = 1, use_bias = False))
        self.conv9.add(tf.keras.layers.BatchNormalization())
        self.conv9.add(tf.keras.layers.ReLU())

        self.conv11 = tf.keras.Sequential()
        self.conv11.add(tf.keras.layers.Conv3DTranspose(filters = 8, kernel_size = 3, strides = 2, padding = "SAME", output_padding = 1, use_bias = False))
        self.conv11.add(tf.keras.layers.BatchNormalization())
        self.conv11.add(tf.keras.layers.ReLU())

        self.prob = tf.keras.layers.Convolution3D(1, 3, 1, "SAME")

    def call(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        
        x = self.conv6(self.conv5(conv4))

        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)

        x = self.prob(x)
        return x

class RefineNet(tf.keras.Model):
    def __init__(self):
        super(RefineNet,self).__init__()
        self.net = tf.keras.Sequential()

        self.net.add(ConvBnReLU(4,  32, 3, 1, "SAME"))
        self.net.add(ConvBnReLU(32, 32, 3, 1, "SAME"))
        self.net.add(ConvBnReLU(32, 32, 3, 1, "SAME"))
        # self.net.add(ConvBnReLU(32,  1, 3, 1, "SAME"))
        self.net.add(tf.keras.layers.Conv2D(filters = 1,
                                            kernel_size = 3,
                                            strides = 1,
                                            padding = "SAME",
                                            use_bias = False))
        
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, img, depth_init):
        # shape = tf.shape(depth_init).numpy()
        shape = depth_init.shape.as_list()
        img = tf.image.resize(img,[shape[1],shape[2]])
        depth_init = tf.expand_dims(depth_init, -1)
        concat = self.concat([img, depth_init])

        depth_residual = self.net(concat)
        # 允许为负，所以没有ReLU
        depth_refine = depth_residual + depth_init
        return depth_refine

class MVSNet(tf.keras.Model):
    def __init__(self, refine = True, training = True, norm = "bn", regular = "3DCNN"):
        super(MVSNet,self).__init__()

        self.training  = training
        self.is_refine = refine
        self.norm = norm
        self.regular = regular
        
        if self.norm == "bn":
            self.feature = FeatureNet()
        else:
            self.feature = FeatureNetGN()

        if self.regular == "3DCNN":
            self.cost_regularization = CostRegNet()
        else:
            pass
    
        if self.is_refine:
            self.refine = RefineNet()

    def call(self, imgs, proj_matrices, depth_values):
        """
        imgs: [B, oN, oH, oW, 3]
        proj_matrices: [B, N, 4, 4]
        depth_values: [B, num_depth]
        """
        # shape = tf.shape(imgs).numpy()
        shape = imgs.shape.as_list()
        batch, num_views, img_height, img_width = shape[0], shape[1], shape[2], shape[3]
        # shape = tf.shape(depth_values).numpy()
        shape = depth_values.shape.as_list()
        num_depth = shape[1]

        imgs = tf.split(imgs, num_views, axis = 1)
        proj_matrices = tf.split(proj_matrices, num_views, axis = 1)
        # N x [B, 1, H, W, C] -> N x [B, H, W, C]
        imgs = [tf.squeeze(img, 1) for img in imgs]
        # N x [B, 1, 4, 4] -> N x [B, 4, 4]
        proj_matrices = [tf.squeeze(proj_matrice, 1) for proj_matrice in proj_matrices]
        ''' Feature Extraction '''
        # N x [B, H, W, C]
        features = [self.feature(img) for img in imgs]
        # [B, H, W, C], (N - 1) x [B, H, W, C]
        ref_feature, src_features = features[0], features[1:]

        # [B, 4, 4], (N - 1) x [B, 4, 4]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        ''' Differentiable Homography '''
        # [B, H, W, C] -> [B, 1, H, W, C] -> [B, num_depth, H, W, C] 
        # ref_volume = tf.tile(tf.expand_dims(ref_feature, axis = 1), [1, num_depth, 1, 1, 1])
        # volume_sum = ref_volume
        # volume_sq_sum = ref_volume ** 2
        # del ref_volume
        volume_sum = tf.tile(tf.expand_dims(ref_feature, axis = 1), [1, num_depth, 1, 1, 1])
        volume_sq_sum = volume_sum ** 2
        
        for src_feat, src_proj in zip(src_features, src_projs):
            # src_feat-[B, H, W, C], src_proj-[B, 4, 4]
            warp_volume = HomographyWarping(src_feat, src_proj, ref_proj, depth_values)
            if self.training: 
                volume_sum = volume_sum + warp_volume
                volume_sq_sum = volume_sq_sum + warp_volume ** 2
            else:
                volume_sum += warp_volume
                volume_sq_sum += tf.pow(warp_volume, 2)
            # del warp_volume
        # [B, num_depth, H, W, C] 
        volume_variance = (volume_sq_sum / num_views) - ((volume_sum / num_views)**2)

        ''' Cost Volume Regularization '''
        # [B, num_depth, H, W, 1]
        cost_reg = self.cost_regularization(volume_variance)
        # [B, num_depth, H, W]
        cost_reg = tf.squeeze(cost_reg, -1)

        prob_volume = tf.keras.activations.softmax(cost_reg, axis = 1)
        # [B, H, W]
        depth = depth_regression(prob_volume, depth_values)

        # photometric confidence
        # [B, num_depth, H, W] -> [B, num_depth, H, W, 1] -> [B, 1+num_depth+2, H, W, 1] -> [B, num_depth, H, W, 1] -> [B, num_depth, H, W]
        prob_volume_sum4 = tf.pad(tf.expand_dims(prob_volume, axis = -1), paddings =[[0,0],[1,2],[0,0],[0,0],[0,0]] ) 
        prob_volume_sum4 = tf.keras.layers.AveragePooling3D(pool_size = (4, 1, 1), strides = 1, padding = 'valid')(prob_volume_sum4)
        prob_volume_sum4 = 4.0 * tf.squeeze(prob_volume_sum4, axis = -1)
        # [B, H, W]
        depth_index = depth_regression(prob_volume, tf.tile(tf.expand_dims(tf.range(num_depth, dtype = tf.float32), axis = 0), [batch, 1]))
        depth_index = tf.cast(depth_index, tf.int32)
        photometric_confidence = get_propability_map(prob_volume_sum4, depth_index)

        if self.is_refine:
            # [B, H, W, C]
            ref_img = imgs[0]
            refine_depth = tf.squeeze(self.refine(ref_img, depth),-1)
            return depth, refine_depth, photometric_confidence
        else:
            return depth, photometric_confidence

def mvsnet_loss(depth_est, depth_refine, depth_gt, mask, sigma = 1):
    """
    depth_est: [B, H, W]
    depth_refine: [B, H, W]
    depth_gt: [B, H, W]
    mask: [B, H, W]
    """
    mask = mask > 0.5
    sigma_2 = sigma ** 2
    
    depth_est = tf.boolean_mask(depth_est, mask)
    depth_refine = tf.boolean_mask(depth_refine, mask)
    depth_gt = tf.boolean_mask(depth_gt, mask)

    depth_diff = depth_est - depth_gt
    depth_refine_diff = depth_refine - depth_gt

    abs_depth_diff = tf.abs(depth_diff)
    abs_depth_refine_diff = tf.abs(depth_refine_diff)

    smoothL1_sign = tf.stop_gradient( tf.cast( tf.less( abs_depth_diff, 1.0 / sigma_2 ), tf.float32 ) )
    refine_smoothL1_sign = tf.stop_gradient( tf.cast( tf.less( abs_depth_refine_diff, 1.0 / sigma_2 ), tf.float32 ) )
    
    loss_depth = tf.pow(depth_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_depth_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss_depth = tf.reduce_mean(loss_depth)

    loss_depth_refine = tf.pow(depth_refine_diff, 2) * (sigma_2 / 2.) * refine_smoothL1_sign + (abs_depth_refine_diff - (0.5 / sigma_2)) * (1. - refine_smoothL1_sign)
    loss_depth_refine = tf.reduce_mean(loss_depth_refine)

    loss = loss_depth + loss_depth_refine

    return loss

def Thres_metrics(depth_est, depth_gt, mask, thres):
    mask = mask > 0.5
    depth_est = tf.boolean_mask(depth_est, mask)
    depth_gt = tf.boolean_mask(depth_gt, mask)
    errors = tf.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return tf.reduce_mean(tf.cast(err_mask, tf.float32))

def AbsDepthError_metrics(depth_est, depth_gt, mask):
    mask = mask > 0.5
    depth_est = tf.boolean_mask(depth_est, mask)
    depth_gt = tf.boolean_mask(depth_gt, mask)
    return tf.reduce_mean(tf.abs(depth_est - depth_gt))

def non_zero_mean_absolute_diff(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch """
    # shape = tf.shape(y_pred).numpy()
    shape = y_pred.shape.as_list()
    interval = tf.reshape(interval, [shape[0],])
    mask_true = tf.cast(tf.math.not_equal(y_true, 0.0), dtype = tf.float32)
    denom = tf.reduce_sum(mask_true, axis=[1, 2]) + 1e-7
    masked_abs_error = tf.abs(mask_true * (y_true - y_pred))
    masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2])
    masked_mae = tf.reduce_sum((masked_mae / interval) / denom)
    return masked_mae

def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    # shape = tf.shape(y_pred).numpy()
    shape = y_pred.shape.as_list()
    mask_true = tf.cast(tf.math.not_equal(y_true, 0.0), dtype='float32')
    denom = tf.reduce_sum(mask_true) + 1e-7
    interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1]), [1, shape[1], shape[2]])
    abs_diff_image = tf.abs(y_true - y_pred) / interval_image
    less_one_image = mask_true * tf.cast(tf.math.less_equal(abs_diff_image, 1.0), dtype = tf.float32)    
    return tf.reduce_sum(less_one_image) / denom

def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    # shape = tf.shape(y_pred).numpy()
    shape = y_pred.shape.as_list()
    mask_true = tf.cast(tf.math.not_equal(y_true, 0.0), dtype='float32')
    denom = tf.reduce_sum(mask_true) + 1e-7
    interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1]), [1, shape[1], shape[2]])
    abs_diff_image = tf.abs(y_true - y_pred) / interval_image
    less_three_image = mask_true * tf.cast(tf.math.less_equal(abs_diff_image, 3.0), dtype = tf.float32)    
    return tf.reduce_sum(less_three_image) / denom

def mvsnet_regression_loss(depth_pred, depth, depth_interval):
    """
    depth_pred:[B,H,W]
    depth:[B,H,W]
    depth_interval:[B,]
    """
    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(depth, depth_pred, depth_interval)
    less_one_accuracy = less_one_percentage(depth, depth_pred, depth_interval)
    less_three_accuracy = less_three_percentage(depth, depth_pred, depth_interval)
    return masked_mae, less_one_accuracy, less_three_accuracy


if __name__ == "__main__":
    print(f"tf version:{tf.__version__}")
    print(f"tf gpu:{tf.test.is_gpu_available()}")

    # mvsnet = MVSNet(refine = True, training = True)

    # for weight in mvsnet.weights:
    #     print(weight.name, weight.shape)

    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #     assert tf.config.experimental.get_memory_growth(physical_devices[0])
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     pass

    # import numpy as np

    # # B, N, H, W, C = 1, 3, 1184, 1600, 3
    # B, N, H, W, C = 1, 3, 640, 512, 3
    # num_depth = 192

    # imgs = np.random.random((B, N, H, W, C)).astype(np.float32)
    # depth_value = (100.0 + 1000.0 * np.random.random((1, num_depth))).astype(np.float32)
    # depth_values = np.concatenate([depth_value for _ in range(B)], axis = 0)
    # imgs = tf.constant(imgs)
    # projs = np.zeros([B, N, 4, 4])
    # for i in range(B):
    #     for j in range(N):
    #         theta_x = 2 * np.pi * np.random.random()
    #         theta_y = 2 * np.pi * np.random.random()
    #         theta_z = 2 * np.pi * np.random.random()

    #         Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]], dtype = np.float32)
    #         Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]], dtype = np.float32)
    #         Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]], dtype = np.float32)
    #         R = Rx.dot(Ry.dot(Rz))
            
    #         projs[i, j, 3, 3] = 1
    #         projs[i, j, :3, :3] = R
    #         projs[i, j, 3, 0] = 100 * np.random.random() 
    #         projs[i, j, 3, 1] = 100 * np.random.random() 
    #         projs[i, j, 3, 2] = 100 * np.random.random() 

    # projs = projs.astype(np.float32)
    # outputs = mvsnet(imgs, projs, depth_values)
    # print(outputs['depth'].shape, outputs['refine_depth'].shape, outputs['photometric_confidence'].shape)
    
    # with tf.GradientTape() as g:
    #     g.watch(imgs)
    
    #     dy_dx = g.gradient(outputs['depth'], imgs)
    #     print(dy_dx)
    
    # feat_ = FeatureNet()
    # # feat_.build(input_shape=(None,1184, 1600, 3))
    # image = tf.zeros((2,1184,1600,3))
    # s = tf.shape(image).numpy()
    # print(s[0],s[1],s[2],s[3])
    # feat = feat_(image)
    # # [1,296,400,32]
    # print(feat.shape)

    # prob_ = CostRegNet()
    # # prob_.build(input_shape=(None,296,400,256,32))
    # feat = tf.zeros((1,296,400,256,32))
    # prob = prob_(feat)
    # # [8,296,400,32]
    # print(prob.shape)

    # for weight in feat_.weights:
    #     print(weight.name, weight.shape)

    # for weight in prob_.weights:
    #     print(weight.name, weight.shape)


    # import torch
    # import numpy as np

    # b,d,h,w = 2,256,128,128
    # x = np.random.random((b, d, h, w)).astype(np.float32)
    # index = np.random.randint(0,d,size=(b,h,w)).astype(np.int32)
    
    # r_tf = get_propability_map(x, index)
    # print(r_tf.numpy())

    # x = torch.from_numpy(x)
    # index = torch.from_numpy(index).long()
    # r_pt = torch.gather(x, 1, index.unsqueeze(1)).squeeze(1)
    # print(r_pt.numpy())
    # print('-'*100)
    # print(r_pt.numpy()-r_tf.numpy())
    # print((r_pt.numpy()-r_tf.numpy()).sum())

    ''' loss test '''
    # import numpy as np
 
    # depth_est = (2 * np.random.random((2,128,128))).astype(np.float32)
    # depth_gt = (2 * np.random.random((2,128,128))).astype(np.float32)
    # mask = np.random.randint(0,2,(2,128,128))

    # tf_loss =  mvsnet_loss(depth_est, depth_gt, mask, sigma = 1)
    # print(tf_loss)

    # import torch
    # import torch.nn.functional as F
    # mask = mask>0.5
    # pt_loss = F.smooth_l1_loss(torch.from_numpy(depth_est[mask]), torch.from_numpy(depth_gt[mask]), size_average=True)
    # print(pt_loss)


    # tf_loss =  mvsnet_loss(depth_est, depth_gt, mask, sigma = 1):
    # """
    # depth_est: [B, H, W]
    # depth_gt: [B, H, W]
    # mask: [B, H, W]