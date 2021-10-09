# A Tensorflow2.x Implementation of MVSNet

## train
python train.py --normalization gn --regularization 3DCNN --nun_depth 128

## train with smooth-l1-loss
python train_sml1.py --normalization gn --regularization 3DCNN --batch_size 1 --epochs 16

python multi_train_sml1.py --normalization gn --regularization 3DCNN --batch_size 8 --epochs 16 --

## inference
python test.py --path PATH_TO_TEST_DATA --max_w 1152 --max_h 864 --num_depth 192 --interval_scale 1.06 --weights ./checkpoints/mvsnet-dtu-3DCNN-gn/weights/model-3.h5	

## fusion
python depthfusion.py --dense_folder ./mvs_training/test_data/scan9/outputs --prob_threshold 0.1

