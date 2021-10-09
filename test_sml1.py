import os
import sys
import cv2
import time
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

from models.mvsnet import *
from utils.preprocess import *

"""
Test
"""
def parse_args():
    parser = argparse.ArgumentParser(description="A Tensorflow2.3 Implementation of MVSNet")

    parser.add_argument("--normalization", default = "gn", help = "select batch or group normalization", type = str)

    parser.add_argument("--path", default = "./mvs_training/test_data/scan9", help = "test data path", type = str)
    parser.add_argument("--mode", default = "train", help = "train or test", choices = ["train", "test", "profilw"], type = str)
    parser.add_argument("--model", default = "mvsnet", help = "select model", type = str)
    parser.add_argument("--weights_path", default = "./checkpoints/mvsnet-dtu/weights/model-16.h5", help = "checkpoints path", type = str)

    parser.add_argument("--num_depth", default = 256, type = int, help = "The number of depth values")
    parser.add_argument("--num_view", default = 5, type = int, help = "The number of src views")
    parser.add_argument("--max_w", default = 1600, type = int, help = "Max width in training")
    parser.add_argument("--max_h", default = 1200, type = int, help = "Max height in training")
    parser.add_argument("--interval_scale", default = 0.8, type = float, help = "Depth interval")
    parser.add_argument("--adaptive_scaling", action = "store_true", help = "Let image size to fit the network, including 'scaling', 'cropping'")
    parser.add_argument("--base_image_size", default = 8, help = "Base image size")
    parser.add_argument("--sample_scale", default = 0.25, help = "Downsample scale for building cost volume (W and H)")
    args = parser.parse_args()
    return args

def gen_pipeline_mvs_list(folder, num_views = 5):
    """ mvs input path list """
    image_folder = os.path.join(folder, 'images')
    cam_folder = os.path.join(folder, 'cams')
    cluster_list_path = os.path.join(folder, 'pair.txt')
    cluster_list = open(cluster_list_path).read().split()

    # for each dataset
    mvs_list = []
    pos = 1
    for i in range(int(cluster_list[0])):
        paths = []

        # ref image
        ref_index = int(cluster_list[pos])
        pos += 1
        ref_image_path = os.path.join(image_folder, ('%08d.jpg' % ref_index))
        ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
        paths.append(ref_image_path)
        paths.append(ref_cam_path)

        # view images
        all_view_num = int(cluster_list[pos])
        pos += 1
        check_view_num = min(num_views - 1, all_view_num)
        for view in range(check_view_num):
            view_index = int(cluster_list[pos + 2 * view])
            view_image_path = os.path.join(image_folder, ('%08d.jpg' % view_index))
            view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
            paths.append(view_image_path)
            paths.append(view_cam_path)
        pos += 2 * all_view_num
        # depth path
        mvs_list.append(paths)
    return mvs_list

class DataGenerator():
    def __init__(self, data_list, args):
        self.args = args
        self.data_list = data_list

        self.dataset = tf.data.Dataset.from_generator(self.item, output_types=(tf.float32, tf.float32, tf.float32, tf.float32 ,tf.int32, tf.float32, tf.float32, tf.float32, tf.float32))
        # self.dataset = self.dataset.shuffle(10 * self.batch_size, reshuffle_each_iteration=True)
        self.dataset = self.dataset.batch(1)
        self.dataset = self.dataset.prefetch(buffer_size=1)

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype = np.float32) / 255.
        # np_img = np.array(img, dtype = np.float32)
        return np_img

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype = np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype = np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.args.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def item(self):
        for i, data in enumerate(self.data_list):
            # data = self.data_list[idx]
            # data:[img1,cam1, img2,cam2, ....]
            image_index = int(os.path.splitext(os.path.basename(data[0]))[0])
            selected_view_num = int(len(data) / 2)

            imgs, depth_values, proj_matrices = [], None, []
            Intrinsics, Extrinsics = [], []
            for view in range(min(self.args.num_view, selected_view_num)):

                img_filename = data[view * 2]
                proj_filename = data[view * 2 + 1]

                imgs.append(self.read_img(img_filename))
                intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_filename)
                Intrinsics.append(intrinsics)
                Extrinsics.append(extrinsics)

                # ref image
                if view == 0:
                    depth_values = np.arange(depth_min, depth_interval * self.args.num_depth + depth_min, depth_interval, dtype = np.float32)

            if selected_view_num < self.args.num_view:
                for _ in range(selected_view_num, self.args.num_view):
                    imgs.append(imgs[0])
                    Intrinsics.append(Intrinsics[0])
                    Extrinsics.append(Extrinsics[0])

            resize_scale = 1
            if self.args.adaptive_scaling:
                h_scale, w_scale = 0, 0
                for view in range(self.args.num_view):
                    height_scale = float(self.args.max_h) / imgs[view].shape[0]
                    width_scale = float(self.args.max_w) / imgs[view].shape[1]
                    if height_scale > h_scale:
                        h_scale = height_scale
                    if width_scale > w_scale:
                        w_scale = width_scale
                if h_scale > 1 or w_scale > 1:
                    print("max_h and max_w should < W and H.")
                    exit(-1)
                    
                resize_scale = h_scale
                if w_scale > resize_scale:
                    resize_scale = w_scale
                    
            scaled_input_images, scaled_input_Intrinsics = scale_mvs_input(imgs, Intrinsics, scale = resize_scale)
            croped_images, croped_Intrinsics = crop_mvs_input(scaled_input_images, scaled_input_Intrinsics, self.args)
            scale_Intrinsics = scale_mvs_camera(croped_Intrinsics, self.args.sample_scale)
            
            scale_images = [scale_image(img, self.args.sample_scale) for img in scaled_input_images]

            for extrinsic, intrinsic in zip(Extrinsics, scale_Intrinsics):
                proj_mat = extrinsic.copy()
                proj_mat[:3, :4] = np.matmul(intrinsic, proj_mat[:3, :4])
                proj_matrices.append(proj_mat)

            # scaled_input_images = [(img-np.mean(img))/np.std(img) for img in scaled_input_images]
            scaled_input_images = np.stack(scaled_input_images)
            scale_images = np.stack(scale_images)
            proj_matrices = np.stack(proj_matrices)
            scale_Intrinsics = np.stack(scale_Intrinsics)
            Extrinsics = np.stack(Extrinsics)

            yield scaled_input_images, scale_images, proj_matrices, depth_values, image_index, scale_Intrinsics, Extrinsics, depth_min, depth_interval

def test():
    args = parse_args()
    # mvsnet = MVSNet(refine = True, training = True)
    mvsnet = MVSNet(refine = True, training = True, norm = args.normalization, regular = '3DCNN')

    mvs_list = gen_pipeline_mvs_list(args.path, num_views = args.num_view)
    dataset = DataGenerator(mvs_list, args)
    
    output_path = os.path.join(args.path, "outputs")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(os.path.join(output_path, 'depths_mvsnet')):
        os.makedirs(os.path.join(output_path, 'depths_mvsnet'))
    if not os.path.exists(os.path.join(output_path, 'images')):
        os.makedirs(os.path.join(output_path, 'images'))
    if not os.path.exists(os.path.join(output_path, 'cams')):
        os.makedirs(os.path.join(output_path, 'cams'))

    count, times = 0, 0.0
    for i, data in enumerate(dataset.dataset):
        bar((i+1)/len(dataset.data_list), args.path.split('/')[-1] )
        # (1, N, H, W, 3) (1, N, H/4, W/4, 3) (1, 5, 4, 4) (1, 256) index 
        imgs, scale_imgs, projs, depth_values, img_index, intrinsics, extrinsics, depth_min, depth_interval = data
        if i == 0:
            mvsnet(imgs, projs, depth_values)
            mvsnet.load_weights(args.weights_path)

        # (1, H/4, W/4) (1, H/4, W/4) (1, H/4, W/4)
        start = time.time()
        depth, refine_depth, photometric_confidence = mvsnet(imgs, projs, depth_values)
        end = time.time()
        times += end - start
        count += 1

        depth = np.squeeze(depth.numpy())
        refine_depth = np.squeeze(refine_depth.numpy())
        photometric_confidence = np.squeeze(photometric_confidence.numpy())
        scale_img = np.squeeze(np.squeeze(scale_imgs.numpy())[0])
        img_index = np.squeeze(img_index.numpy()) 
        intrinsics = np.squeeze(intrinsics.numpy()[0][0])
        extrinsics = np.squeeze(extrinsics.numpy()[0][0])

        depth_path = os.path.join(output_path, 'depths_mvsnet', '%08d_init.pfm' % img_index)
        refine_depth_path = os.path.join(output_path, 'depths_mvsnet', '%08d_refine.pfm' % img_index)
        photometric_confidence_path = os.path.join(output_path, 'depths_mvsnet', '%08d_prob.pfm' % img_index)
        scale_img_path = os.path.join(output_path, 'images','%08d.jpg' % img_index)
        cam_path = os.path.join(output_path, 'cams', '%08d.txt' % img_index)

        # save output
        write_pfm(depth_path, depth)
        write_pfm(refine_depth_path, refine_depth)
        write_pfm(photometric_confidence_path, photometric_confidence)
        # cv2.imwrite(scale_img_path, (scale_img * 255).astype(np.uint8)[:,:,::-1])
        cv2.imwrite(scale_img_path, (scale_img*255).astype(np.uint8)[:,:,::-1])
        # cv2.imshow('img', (scale_img * 255).astype(np.uint8)[:,:,::-1])
        # cv2.waitKey(0)
        # break

        # extrinsics
        cam = np.zeros((2, 4, 4), dtype = np.float32)
        cam[0] = extrinsics
        cam[1][:3,:3] = intrinsics
        cam[1][3][0] = depth_min
        cam[1][3][1] = depth_interval
        cam[1][3][2] = args.num_depth
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        write_cam(cam_path, cam)
    # Average time for 49 images: 0.4558243313614203s
    print(f"Average time for {count} images: {times / count}s")

if __name__ == "__main__":
    print(f"tf version:{tf.__version__}")
    print(f"tf gpu:{tf.test.is_gpu_available()}")

    test()

    # rlaunch --cpu=2 --gpu=1 --memory=10240 -- python test.py --max_w 1152 --max_h 864 --num_depth 192 --interval_scale 1.06
