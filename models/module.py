import tensorflow as tf
import tensorflow_addons as tfa


class ConvBnReLU(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, pad = "SAME"):
        super(ConvBnReLU,self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters = out_channels,
                                           kernel_size = kernel_size,
                                           strides = stride,
                                           padding = pad,
                                           use_bias = False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        return self.relu( self.bn( self.conv(x) ) )

class ConvGnReLU(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, pad = "SAME"):
        super(ConvGnReLU,self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters = out_channels,
                                           kernel_size = kernel_size,
                                           strides = stride,
                                           padding = pad,
                                           use_bias = False)
        self.gn = tfa.layers.GroupNormalization(groups = out_channels//8)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        return self.relu( self.gn( self.conv(x) ) )

class ConvBn(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, pad = "SAME"):
        super(ConvBn,self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters = out_channels,
                                           kernel_size = kernel_size,
                                           strides = stride,
                                           padding = pad,
                                           use_bias = False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        return self.bn( self.conv(x) )

class ConvGn(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, pad = "SAME"):
        super(ConvGn,self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters = out_channels,
                                           kernel_size = kernel_size,
                                           strides = stride,
                                           padding = pad,
                                           use_bias = False)
        self.gn = tfa.layers.GroupNormalization(groups = out_channels//8)

    def call(self, x):
        return self.gn( self.conv(x) )


class ConvBnReLU3D(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, pad = "SAME"):
        super(ConvBnReLU3D,self).__init__()

        self.conv = tf.keras.layers.Conv3D(filters = out_channels,
                                           kernel_size = kernel_size,
                                           strides = stride,
                                           padding = pad,
                                           use_bias = False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        return self.relu( self.bn( self.conv(x) ) )

class ConvBn3D(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, pad = "SAME"):
        super(ConvBn3D,self).__init__()

        self.conv = tf.keras.layers.Conv3D(filters = out_channels,
                                           kernel_size = kernel_size,
                                           strides = stride,
                                           padding = pad,
                                           use_bias = False)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        return self.bn( self.conv(x) )

def HomographyWarping(src_feat, src_proj, ref_proj, depth_values):
    """
    src_feat [B, H, W, C]
    src_proj [B, 4, 4]
    ref_ptoj [B, 4, 4]
    depth_values [B, num_depth]
    """
    # shape = tf.shape(src_feat).numpy()
    shape = src_feat.shape.as_list()
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    # shape = tf.shape(depth_values).numpy()
    shape = depth_values.shape.as_list()
    num_depth = shape[1]

    # |R2inv(R1) t2-R2inv(R1)t1|
    # |   0             1      |
    proj = tf.matmul(src_proj, tf.linalg.inv(ref_proj))
    # [B, 3, 3], [B, 3, 1]
    rot, trans = proj[:, :3, :3], proj[:, :3, 3:4]
    # 注意这里和pytorch生成的是反的
    # y, x = tf.meshgrid(tf.range(0, height, dtype = tf.float32), tf.range(0, width, dtype = tf.float32))
    x, y = tf.meshgrid(tf.range(0, width, dtype = tf.float32), tf.range(0, height, dtype = tf.float32))
    x, y = tf.reshape(x, (-1,)), tf.reshape(y, (-1,))
    # [3, HW]
    xyz = tf.stack((x, y, tf.ones_like(x)))
    # [B, 3, HW]
    xyz = tf.tile(tf.expand_dims(xyz, axis = 0), [batch, 1, 1])
    # [B, 3, HW]
    # print(tf.shape(rot).numpy(), tf.shape(xyz).numpy())
    rot_xyz = tf.matmul(rot, xyz)
    # [B, 3, num_depth, HW]
    rot_depth_xyz = tf.tile(tf.expand_dims(rot_xyz, axis = 2), [1, 1, num_depth, 1]) * tf.reshape(depth_values, [batch, 1, num_depth, 1])
    
    # [B, 3, num_depth, HW]
    proj_xyz = rot_depth_xyz + tf.reshape(trans, [batch, 3, 1, 1])
    # [B, 2, num_depth, HW]
    proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
    proj_xy = tf.transpose(proj_xy, [0, 2, 3, 1]) 
    # [B, num_depth, HW]
    # proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
    # proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
    # # [B, num_depth, HW, 2]
    # proj_xy = tf.stack((proj_x_normalized, proj_y_normalized), axis = 3)
    grid = tf.reshape(proj_xy, [batch, num_depth, height, width, 2])
    # [B, num_depth, H, W, C]
    warped_src_feat =  tfa.image.resampler(src_feat, grid)

    return warped_src_feat

def depth_regression(p, depth_values):
    """
    p:[B, D, H, W]
    depth_values:[B, D]
    """
    # shape = tf.shape(depth_values).numpy()
    shape = depth_values.shape.as_list()
    B, D = shape[0], shape[1]
    depth_values = tf.reshape(depth_values, [B, D, 1, 1])
    depth = p * depth_values
    depth = tf.reduce_sum(depth, axis = 1)

    return depth

def get_propability_map(prob_volume, index):
    """
    prob_volume:[B, D, H, W]
    index:[B, H, W]
    """
    # shape = tf.shape(prob_volume).numpy()
    shape = prob_volume.shape.as_list()
    B, D, H, W = shape[0], shape[1], shape[2], shape[3]
    # Bs, Hs, Ws = tf.meshgrid(tf.range(B), tf.range(H), tf.range(W))
    Hs, Bs, Ws = tf.meshgrid(tf.range(H), tf.range(B), tf.range(W))

    Bs = tf.reshape(Bs, [-1,])
    Hs = tf.reshape(Hs, [-1,])
    Ws = tf.reshape(Ws, [-1,])
    BHW = tf.stack([Bs, Hs, Ws], axis = 1)
    Ds = tf.gather_nd(index, BHW)
    BHWD = tf.stack([Bs, Ds, Hs, Ws], axis = 1)
    prob_map = tf.gather_nd(prob_volume, BHWD)
    prob_map = tf.reshape(prob_map, [B, H, W])

    return prob_map

if __name__ == "__main__":
    print(f"tf version:{tf.__version__}")
    print(f"tf gpu:{tf.test.is_gpu_available()}")

    
    import numpy as np
    import torch
    import torch.nn.functional as F

    b,num_depth,h,w = 2,10,128,256
    prob = np.random.random((b,num_depth, h, w)).astype(np.float32)
    
    prob_volume = tf.keras.activations.softmax(tf.constant(prob), axis = 1)
    prob_volume_sum4 = tf.pad(tf.expand_dims(prob_volume, axis = -1), paddings =[[0,0],[1,2],[0,0],[0,0],[0,0]] ) 
    prob_volume_sum4 = tf.keras.layers.AveragePooling3D(pool_size = (4, 1, 1), strides = 1, padding = 'valid')(prob_volume_sum4)
    prob_volume_sum4 = 4.0 * tf.squeeze(prob_volume_sum4, axis = -1)
    # [B, H, W]
    depth_index = depth_regression(prob_volume, tf.tile(tf.expand_dims(tf.range(num_depth, dtype = tf.float32), axis = 0), [b, 1]))
    depth_index = tf.cast(depth_index, tf.int32)
    photometric_confidence = get_propability_map(prob_volume_sum4, depth_index)


    def depth_regression_pt(p, depth_values):
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
        depth = torch.sum(p * depth_values, 1)
        return depth

    prob_volume = torch.from_numpy(prob)
    prob_volume = F.softmax(prob_volume, dim=1)
    prob_volume_sum4_pt = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
    depth_index_pt = depth_regression_pt(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
    photometric_confidence_pt = torch.gather(prob_volume_sum4_pt, 1, depth_index_pt.unsqueeze(1)).squeeze(1)

    print(np.abs(prob_volume_sum4.numpy() - prob_volume_sum4_pt.numpy()).sum())
    print((depth_index.numpy() - depth_index_pt.numpy()).sum())
    print(np.abs(photometric_confidence.numpy() - photometric_confidence_pt.numpy()).sum())
    



    # cbr = ConvBnReLU(3,3)
    # cbr.build(input_shape=(None,224,224,3))
    # for weight in cbr.weights:
    #     print(weight.name)

    # import numpy as np
    # b, h, w, c = 2, 32, 32, 16
    # num_depth = 100
    # p = np.random.random((b,num_depth, h, w))
    # depth = np.random.random((b,num_depth))
    # depth_ = depth_regression(p,depth)

    # import numpy as np

    # b, h, w, c = 2, 16, 16, 32
    # num_depth = 16

    # src_feat = np.random.random((b, h, w, c)).astype(np.float32)
    # depth_value = (100.0 + 1000.0 * np.random.random((1, num_depth))).astype(np.float32)
    # depth_values = np.concatenate([depth_value for _ in range(b)], axis = 0)

    # src_proj = np.zeros([b, 4, 4])
    # for i in range(b):
    #     theta_x = 2 * np.pi * np.random.random()
    #     theta_y = 2 * np.pi * np.random.random()
    #     theta_z = 2 * np.pi * np.random.random()

    #     Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]], dtype = np.float32)
    #     Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]], dtype = np.float32)
    #     Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]], dtype = np.float32)
    #     R = Rx.dot(Ry.dot(Rz))
        
    #     src_proj[i, 3, 3] = 1
    #     src_proj[i, :3, :3] = R
    #     src_proj[i, 3, 0] = 100 * np.random.random() 
    #     src_proj[i, 3, 1] = 100 * np.random.random() 
    #     src_proj[i, 3, 2] = 100 * np.random.random() 
    
    # ref_proj = np.zeros([b, 4, 4])
    # for i in range(b):
    #     theta_x = 2 * np.pi * np.random.random()
    #     theta_y = 2 * np.pi * np.random.random()
    #     theta_z = 2 * np.pi * np.random.random()

    #     Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]], dtype = np.float32)
    #     Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]], dtype = np.float32)
    #     Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]], dtype = np.float32)
    #     R = Rx.dot(Ry.dot(Rz))
        
    #     ref_proj[i, 3, 3] = 1
    #     ref_proj[i, :3, :3] = R
    #     ref_proj[i, 3, 0] = 100 * np.random.random() 
    #     ref_proj[i, 3, 1] = 100 * np.random.random() 
    #     ref_proj[i, 3, 2] = 100 * np.random.random()
    
    # src_feat = src_feat.astype(np.float32)
    # src_proj = src_proj.astype(np.float32)
    # ref_proj = ref_proj.astype(np.float32)
    # depth_values = depth_values.astype(np.float32)
    # warp = HomographyWarping(src_feat, src_proj, ref_proj, depth_values)
    # print(warp.shape)

    # import torch
    # import torch.nn.functional as F
    # def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    #     # src_fea: [B, C, H, W]
    #     # src_proj: [B, 4, 4]
    #     # ref_proj: [B, 4, 4]
    #     # depth_values: [B, Ndepth]
    #     # out: [B, C, Ndepth, H, W]
    #     batch, channels = src_fea.shape[0], src_fea.shape[1]
    #     num_depth = depth_values.shape[1]
    #     height, width = src_fea.shape[2], src_fea.shape[3]

    #     with torch.no_grad():
    #         # duan
    #         # |R2inv(R1) t2-R2inv(R1)t1|
    #         # |   0             1      |
    #         proj = torch.matmul(src_proj, torch.inverse(ref_proj))
    #         rot = proj[:, :3, :3]  # [B,3,3]
    #         trans = proj[:, :3, 3:4]  # [B,3,1]

    #         y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
    #                             torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
    #         y, x = y.contiguous(), x.contiguous()
    #         y, x = y.view(height * width), x.view(height * width)
    #         xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    #         xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    #         rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    #         rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
    #                                                                                             1)  # [B, 3, Ndepth, H*W]
    #         proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
    #         proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
    #         proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
    #         proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
    #         proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
    #         grid = proj_xy

    #     warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
    #                                 padding_mode='zeros')
    #     warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    #     return warped_src_fea

    # src_feat_pt, src_proj_pt, ref_proj_pt, depth_values_pt = torch.from_numpy(np.transpose(src_feat,[0,3,1,2])), torch.from_numpy(src_proj), torch.from_numpy(ref_proj), torch.from_numpy(depth_values)
    # warp_pt = homo_warping(src_feat_pt, src_proj_pt, ref_proj_pt, depth_values_pt)
    # warp_pt = warp_pt.numpy().transpose([0,2,3,4,1])

    # # 存在 10^(-2) 的误差，可以接受？？？？？
    # # print(warp_pt, warp_pt.sum())
    # # print(warp.numpy(), warp.numpy().sum())

    # for i in range(b):
    #     for j in range(num_depth):
    #         for k in range(h):
    #             for l in range(w):
    #                 print(warp_pt[i,j,k,l], warp.numpy()[i,j,k,l])
    # print(np.abs(warp_pt - warp.numpy()).sum())


