import os
import cv2
import sys
import time
import numpy as np
from PIL import Image
import tensorflow as tf

from utils.config import *
from models.mvsnet import *
from datasets.dtu_yao import *

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.numpy().astype(np.uint8)
    # x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = cv2.applyColorMap(x, cmap)
    x_ = tf.convert_to_tensor(np.expand_dims(x_, axis=0))
    return x_

def visualize_prob(prob, cmap=cv2.COLORMAP_BONE):
    """
    prob: (H, W) 0~1
    """
    x = 255 * prob.numpy().astype(np.uint8)
    # x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = cv2.applyColorMap(x, cmap)
    x_ = tf.convert_to_tensor(np.expand_dims(x_, axis=0))
    return x_


def train():
    args = parse_args()
    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    ''' Datasets '''
    train_dataset = MVSDatasets(args.batch_size, args.train_path, args.train_list, "train", args.num_view, args.num_depth, args.interval_scale)
    test_dataset = MVSDatasets(args.batch_size, args.test_path, args.test_list, "test", args.num_view, args.num_depth, args.interval_scale)
    
    len_epoch = len(train_dataset.metas) / args.batch_size
    milestones = [int(float(epoch_idx) * len(train_dataset.metas) / args.batch_size) for epoch_idx in args.lr_epochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lr_epochs.split(':')[1])
    learning_rates = [args.lr * (lr_gamma**i) for i in range(len(milestones)+1)]
    # learning_rates = [args.lr * args.batch_size * (lr_gamma**i) for i in range(len(milestones)+1)]
    
    train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset.dataset)
    test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset.dataset)

    ''' MVSNet '''
    with mirrored_strategy.scope():
        mvsnet = MVSNet(refine = True, training = True, norm = args.normalization, regular = args.regularization)
    
    ''' Load trained files '''
    weight_dir = os.path.join(args.log_dir, f"multi-{args.model}-{args.dataset}-{args.regularization}-{args.normalization}", "weights")
    start_epoch = 0
    trained_weight = None
    if args.resume:
        file_list = os.listdir(weight_dir)
        for f in file_list:
            if f.endswith(".h5"):
                e = int(f.split(".")[0].split('-')[1])
                if e > start_epoch:
                    start_epoch = e
                    trained_weight = os.path.join(weight_dir, f)
        # if trained_weight is not None:
            # imgs, proj_matrices, depth_values
            # mvsnet(imgs, proj_matrices, depth_values)
            # mvsnet.build(input_shape=([args.batch_size, args.num_view, 512, 640, 3],[args.batch_size, args.num_view, 4, 4],[args.batch_size, args.num_depth]))
            # print(f"load weight from {trained_weight}")
            # mvsnet.load_weights(trained_weight)

    ''' Optimizer '''
    with mirrored_strategy.scope():
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(milestones, learning_rates)
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_fn, beta_1 = 0.9, beta_2 = 0.999)

    ''' Summary '''
    summary_dir = os.path.join(args.log_dir, f"multi-{args.model}-{args.dataset}-{args.regularization}-{args.normalization}", "summarys")
    writer = tf.summary.create_file_writer(summary_dir)

    def compute_loss(depth_est, depth_refine, depth, mask):
        loss = mvsnet_loss(depth_est, depth_refine, depth, mask)
        abs_depth, abs_refine_depth = AbsDepthError_metrics(depth_est, depth, mask), AbsDepthError_metrics(depth_refine, depth, mask)
        less1, less1_refine = Thres_metrics(depth_est, depth, mask, 1.0), Thres_metrics(depth_refine, depth, mask, 1.0)
        less3, less3_refine = Thres_metrics(depth_est, depth, mask, 3.0), Thres_metrics(depth_refine, depth, mask, 3.0)

        loss = tf.reshape(loss,[-1])
        abs_depth = tf.reshape(abs_depth,[-1])
        abs_refine_depth = tf.reshape(abs_refine_depth,[-1])
        less1 = tf.reshape(less1,[-1])
        less1_refine = tf.reshape(less1_refine,[-1])
        less3 = tf.reshape(less3,[-1])
        less3_refine = tf.reshape(less3_refine,[-1])

        loss = tf.nn.compute_average_loss(loss, global_batch_size=args.batch_size)
        abs_depth = tf.nn.compute_average_loss(abs_depth, global_batch_size=args.batch_size)
        abs_refine_depth = tf.nn.compute_average_loss(abs_refine_depth, global_batch_size=args.batch_size)
        less1 = tf.nn.compute_average_loss(less1, global_batch_size=args.batch_size)
        less1_refine = tf.nn.compute_average_loss(less1_refine, global_batch_size=args.batch_size)
        less3 = tf.nn.compute_average_loss(less3, global_batch_size=args.batch_size)
        less3_refine = tf.nn.compute_average_loss(less3_refine, global_batch_size=args.batch_size)

        return loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine

    def train_step(imgs, proj_matrices, depth, depth_values, mask):
        with tf.GradientTape() as tape:
            depth_est, depth_refine, photometric_confidence = mvsnet(imgs, proj_matrices, depth_values)
            loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine = compute_loss(depth_est, depth_refine, depth, mask)

        grads = tape.gradient(loss, mvsnet.trainable_variables)
        optimizer.apply_gradients(zip(grads, mvsnet.trainable_variables))
        
        return loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine, depth_est, photometric_confidence

    @tf.function
    def distributed_train_step(imgs, proj_matrices, depth, depth_values, mask):
        per_replica_losses = mirrored_strategy.run(train_step, args=(imgs, proj_matrices, depth, depth_values, mask))
        loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine, depth_est, photometric_confidence = per_replica_losses
        
        loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        abs_depth = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, abs_depth, axis=None)
        abs_refine_depth = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, abs_refine_depth, axis=None)
        less1 = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, less1, axis=None)
        less1_refine = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, less1_refine, axis=None)
        less3 = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, less3, axis=None)
        less3_refine = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, less3_refine, axis=None)
        return loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine, depth_est, photometric_confidence

    def test_step(imgs, proj_matrices, depth, depth_values, mask):
        depth_est, depth_refine, photometric_confidence = mvsnet(imgs, proj_matrices, depth_values)
        loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine = compute_loss(depth_est, depth_refine, depth, mask)
        return loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine, depth_est, photometric_confidence

    @tf.function
    def distributed_test_step(imgs, proj_matrices, depth, depth_values, mask):
        per_replica_losses = mirrored_strategy.run(test_step, args=(imgs, proj_matrices, depth, depth_values, mask))
        loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine, depth_est, photometric_confidence = per_replica_losses
        
        loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        abs_depth = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, abs_depth, axis=None)
        abs_refine_depth = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, abs_refine_depth, axis=None)
        less1 = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, less1, axis=None)
        less1_refine = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, less1_refine, axis=None)
        less3 = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, less3, axis=None)
        less3_refine = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, less3_refine, axis=None)
        return loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine, depth_est, photometric_confidence

    '''  train '''
    index = 0
    if trained_weight is not None:
        index = int(start_epoch * len_epoch + 1)

    for i in range(start_epoch, args.epochs):
        if trained_weight is not None:
            for b, data in enumerate(train_dataset):
                imgs, proj_matrices, depth, depth_values, mask = data
                loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine, depth_est, photometric_confidence = distributed_test_step(imgs, proj_matrices, depth, depth_values, mask)
                break
            print(f"load weight from {trained_weight}")
            mvsnet.load_weights(trained_weight)
        
        for b, data in enumerate(train_dataset):
           
            imgs, proj_matrices, depth, depth_values, mask = data

            loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine, depth_est, photometric_confidence = distributed_train_step(imgs, proj_matrices, depth, depth_values, mask)
            ''' Print and Summary '''
            if b % args.summary_interval == 0:
                with writer.as_default():
                    tf.summary.scalar("Train/loss", loss.numpy(), step = index )
                    tf.summary.scalar("Train/abs_error", abs_depth.numpy(), step = index )
                    tf.summary.scalar("Train/abs_refine_error", abs_refine_depth.numpy(), step = index )
                    tf.summary.scalar("Train/less1", less1.numpy(), step = index )
                    tf.summary.scalar("Train/less1_refine", less1_refine.numpy(), step = index )
                    tf.summary.scalar("Train/less3", less3.numpy(), step = index )
                    tf.summary.scalar("Train/less3_refine", less3_refine.numpy(), step = index )
                    tf.summary.scalar("Train/lr", learning_rate_fn( index ).numpy(), step = index )
                    
                    show_index = np.random.randint(args.batch_size)
                    d_ = visualize_depth(depth.values[show_index][0])
                    de_ = visualize_depth(depth_est.values[show_index][0])
                    p_ = visualize_prob(photometric_confidence.values[show_index][0])
                    tf.summary.image("Train/imgs", imgs.values[show_index][:, 0], step = index )
                    tf.summary.image("Train/d_gt", d_, step = index )
                    tf.summary.image("Train/d", de_, step = index )
                    tf.summary.image("Train/prob", p_, step = index )
                    writer.flush()
                print(f"Train epoch:{i + 1}/{args.epochs}, batch:{b}, loss:{loss.numpy()}, lr:{learning_rate_fn( index ).numpy()}")

            index += 1

        ''' Save '''
        mvsnet.save_weights(os.path.join(weight_dir, f"model-{i + 1}.h5"))
        
        '''  Val '''
        test_loss = 0
        test_abs, test_abs_ref = 0, 0 
        test_less1, test_less1_ref = 0, 0
        test_less3, test_less3_ref = 0, 0
        count = 0
        for b, data in enumerate(test_dataset):
            imgs, proj_matrices, depth, depth_values, mask = data
            loss, abs_depth, abs_refine_depth, less1, less1_refine, less3, less3_refine, depth_est, photometric_confidence = distributed_test_step(imgs, proj_matrices, depth, depth_values, mask)
            if b % args.summary_interval == 0:
                print(f"Test epoch:{i + 1}/{args.epochs}, batch:{b}, loss:{loss.numpy()}")

            test_loss += loss.numpy()
            test_abs += abs_depth.numpy()
            test_abs_ref += abs_refine_depth.numpy()
            test_less1 += less1.numpy()
            test_less1_ref += less1_refine.numpy()
            test_less3 += less3.numpy()
            test_less3_ref += less3_refine.numpy()
            count += 1

        with writer.as_default():
            tf.summary.scalar("Val/loss", test_loss / count, step = i )
            tf.summary.scalar("Val/abs_error", test_abs / count, step = i )
            tf.summary.scalar("Val/abs_refine_error", test_abs_ref / count, step = i )
            tf.summary.scalar("Val/less1", test_less1 / count, step = i )
            tf.summary.scalar("Val/less1_refine", test_less1_ref / count, step = i )
            tf.summary.scalar("Val/less3", test_less3 / count, step = i )
            tf.summary.scalar("Val/less3_refine", test_less3_ref / count, step = i )

            show_index = np.random.randint(args.batch_size)
            d_ = visualize_depth(depth.values[show_index][0])
            de_ = visualize_depth(depth_est.values[show_index][0])
            p_ = visualize_prob(photometric_confidence.values[show_index][0])
            tf.summary.image("Val/imgs", imgs.values[show_index][:, 0], step = i )
            tf.summary.image("Val/d_gt", d_, step = i )
            tf.summary.image("Val/d", de_, step = i )
            tf.summary.image("Val/prob", p_, step = i )
            writer.flush()
    
if __name__ == "__main__":
    print(f"tf version:{tf.__version__}")
    print(f"tf gpu:{tf.test.is_gpu_available()}")

    train()