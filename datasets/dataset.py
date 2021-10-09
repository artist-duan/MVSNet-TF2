import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

import sys
sys.path.append('../')
from datasets.data_io import *

class MVSDatasets():
    def __init__(self, batch_size, data_path, mode = "train", n_views = 3, n_depths = 192, interval_scale = 1.06, **kwargs):
        self.batch_size = batch_size
        self.data_path = data_path
        self.mode = mode
        self.n_views = n_views
        self.n_depths = n_depths
        self.interval_scale = interval_scale

        assert self.mode in ["train", "eval", "val"]
        self.training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                             45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                             74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                             101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                             121, 122, 123, 124, 125, 126, 127, 128]
        self.validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
        self.eval_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]

        self.metas = self.build_list()
        # self.metas = self.metas[:111]

        self.dataset = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        self.dataset = self.dataset.shuffle(10 * self.batch_size, reshuffle_each_iteration=True)
        self.dataset = self.dataset.batch(self.batch_size)
        # self.dataset = self.dataset.repeat()

    def build_list(self):
        metas = []
        
        if self.mode == "train":
            scans = self.training_set
        elif self.mode == "val":
            scans = self.validation_set
        else:
            scans = self.eval_set

        for scan in scans:
            with open(os.path.join(self.data_path, "Cameras/pair.txt"), "r") as f:
                num_viewpoints = int(f.readline())
                for view_idx in range(num_viewpoints):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light condition 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print(f"dataset {self.mode} metas:{len(metas)}")
        return metas

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        # np_img = np.array(img, dtype=np.float32) / 255.
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img - np.mean(np_img)) / np.std(np_img)
        return np_img

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def get_item(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        views_ids = [ref_view] + src_views[:self.n_views - 1]

        imgs, mask, depth, depth_values, proj_matrices, intervals = [], None, None, None, [], []

        for i, vid in enumerate(views_ids):
            img_filename = os.path.join(self.data_path,'Rectified/scan{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename = os.path.join(self.data_path, 'Depths/scan{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.data_path, 'Depths/scan{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.data_path, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            imgs.append(self.read_img(img_filename))
            
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            # ref image
            if i == 0:
                depth_values = np.arange(depth_min, depth_interval * self.n_depths + depth_min, depth_interval, dtype = np.float32)
                # 前景、背景 mask
                # mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)
                intervals.append(depth_interval)

                depth_start = depth_min
                depth_end = depth_interval * self.n_depths + depth_min
                _, depth = cv2.threshold(depth, depth_start, 100000, cv2.THRESH_TOZERO)
                _, depth = cv2.threshold(depth, depth_end, 100000, cv2.THRESH_TOZERO_INV)
                
        
        imgs = np.stack(imgs)
        proj_matrices = np.stack(proj_matrices)
        intervals = np.stack(intervals)


        return imgs, proj_matrices, depth, depth_values, intervals

    def generator(self):
        for i in range(len(self.metas)):
            yield self.get_item(i)



if __name__ == "__main__":
    dataset = MVSDatasets(2,"../../Datasets/mvs_training/dtu", "train", 3, 192)
    print(len(dataset.metas))

    # imgs, proj_matrices, depth, depth_values, mask = dataset.get_item(0)
    # print(imgs.shape, proj_matrices.shape, depth.shape, depth_values.shape, mask.shape)
    # print(mask[0,0],depth[0,0])

    import cv2
    # cv2.imshow('img', imgs[0])
    # cv2.imshow('depth', depth.astype(np.uint8))
    # cv2.imshow('mask', depth.astype(np.uint8))
    # cv2.waitKey(0)

    img1, img2 = None, None
    for i in range(2):
        for j, data in enumerate(dataset.dataset):
            imgs, proj_matrices, depth, depth_values, intervals = data
            print(i, j, imgs.shape, proj_matrices.shape, depth.shape, depth_values.shape, intervals)
            if i==0 and j==0:
                img1 = imgs[0][0]
            if i==1 and j==0:
                img2 = imgs[0][0]
            
            break
        # break
    cv2.imshow('1', img1.numpy())
    cv2.imshow('2', img2.numpy())
    cv2.waitKey(0)