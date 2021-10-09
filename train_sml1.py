import os
import sys
import time
import numpy as np
import tensorflow as tf

from utils.config import *
from models.mvsnet import *
from datasets.dtu_yao import *

def train():
    args = parse_args()
    
    ''' Datasets '''
    train_dataset = MVSDatasets(args.batch_size, args.train_path, args.train_list, "train", args.num_view, args.num_depth, args.interval_scale)
    test_dataset = MVSDatasets(args.batch_size, args.test_path, args.test_list, "test", args.num_view, args.num_depth, args.interval_scale)

    ''' MVSNet '''
    mvsnet = MVSNet(refine = True, training = True, norm = args.normalization, regular = args.regularization)
    
    ''' Load trained files '''
    weight_dir = os.path.join(args.log_dir, f"{args.model}-{args.dataset}-{args.regularization}-{args.normalization}", "weights")
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
        
        if trained_weight is not None:
            imgs, proj_matrices, depth, depth_values, mask = train_dataset.dataset[0]
            mvsnet(imgs, proj_matrices, depth_values)
            mvsnet.load_weights(trained_weight)

    ''' Optimizer '''
    milestones = [int(float(epoch_idx) * len(train_dataset.metas) / args.batch_size) for epoch_idx in args.lr_epochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lr_epochs.split(':')[1])
    learning_rates = [args.lr * (lr_gamma**i) for i in range(len(milestones)+1)]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(milestones, learning_rates)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_fn, beta_1 = 0.9, beta_2 = 0.999)

    ''' Summary '''
    summary_dir = os.path.join(args.log_dir, f"{args.model}-{args.dataset}-{args.regularization}-{args.normalization}", "summarys")
    writer = tf.summary.create_file_writer(summary_dir)

    '''  train '''
    index = 0
    for i in range(start_epoch, args.epochs):
        for b, data in enumerate(train_dataset.dataset):
            
            imgs, proj_matrices, depth, depth_values, mask = data
            
            with tf.GradientTape() as tape:
                depth_est, depth_refine, photometric_confidence = mvsnet(imgs, proj_matrices, depth_values)
                loss = mvsnet_loss(depth_est, depth_refine, depth, mask)
                
                abs_depth, abs_refine_depth = AbsDepthError_metrics(depth_est, depth, mask), AbsDepthError_metrics(depth_refine, depth, mask)
                less1, less1_refine = Thres_metrics(depth_est, depth, mask, 1.0), Thres_metrics(depth_refine, depth, mask, 1.0)
                less3, less3_refine = Thres_metrics(depth_est, depth, mask, 3.0), Thres_metrics(depth_refine, depth, mask, 3.0)

            grads = tape.gradient(loss, mvsnet.trainable_variables)
            optimizer.apply_gradients(zip(grads, mvsnet.trainable_variables))

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
                    tf.summary.image("Train/imgs", imgs.numpy()[:, 0], step = index )
                    tf.summary.image("Train/prob", np.expand_dims(photometric_confidence.numpy(), -1) * 255, step = index )

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
        for b, data in enumerate(test_dataset.dataset):
            imgs, proj_matrices, depth, depth_values, mask = data
            depth_est, depth_refine, photometric_confidence = mvsnet(imgs, proj_matrices, depth_values)
            loss = mvsnet_loss(depth_est, depth_refine, depth, mask)
            abs_depth, abs_refine_depth = AbsDepthError_metrics(depth_est, depth, mask), AbsDepthError_metrics(depth_refine, depth, mask)
            less1, less1_refine = Thres_metrics(depth_est, depth, mask, 1.0), Thres_metrics(depth_refine, depth, mask, 1.0)
            less3, less3_refine = Thres_metrics(depth_est, depth, mask, 3.0), Thres_metrics(depth_refine, depth, mask, 3.0)         
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
            tf.summary.image("Val/imgs", imgs.numpy()[:, 0], step = i )
            tf.summary.image("Val/prob", np.expand_dims(photometric_confidence.numpy(), -1) * 255, step = i )
            writer.flush()
    
if __name__ == "__main__":
    print(f"tf version:{tf.__version__}")
    print(f"tf gpu:{tf.test.is_gpu_available()}")

    train()