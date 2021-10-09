import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="A Tensorflow2.3 Implementation of MVSNet")

    parser.add_argument("--multigpu", action = "store_true", help = "Continue to train the model")

    parser.add_argument("--mode", default = "train", help = "train or test", choices = ["train", "test", "profilw"], type = str)
    parser.add_argument("--model", default = "mvsnet", help = "select model", type = str)
    
    parser.add_argument("--normalization", default = "gn", help = "select batch or group normalization", type = str)
    parser.add_argument("--regularization", default = "3DCNN", help = "select regularization method", type = str)

    parser.add_argument("--dataset", default = "dtu", help = "select dataset", type = str)
    parser.add_argument("--train_path", default = "./mvs_training/dtu", help = "train data path")
    parser.add_argument("--train_list", default = "./mvs_training/dtu/train.txt", help = "train list")
    parser.add_argument("--test_path", default = "./mvs_training/dtu", help= "test data path")
    parser.add_argument("--test_list", default = "./mvs_training/dtu/test.txt", help = "test list")
    
    parser.add_argument("--epochs", default = 6, type = int, help = "Number of epochs to train")
    parser.add_argument("--lr", default = 0.001, type = float, help = "Learning rate")
    parser.add_argument("--stepvalue", default = 10000, type = int, help = "Step interval to decay learning rate.")
    parser.add_argument("--gamma", default = 0.9, type = float, help = "Learning rate decay rate.")
    parser.add_argument("--lr_epochs", default = "10,12,14:2", type = str, help = "Epoch ids to downscale lr and the downscale rate")
    parser.add_argument("--wd", default = 0.0, type = float, help = "Weight decay")

    parser.add_argument("--batch_size", default = 1, type = int, help = "Batch size")
    parser.add_argument("--num_depth", default = 192, type = int, help = "The number of depth values")
    parser.add_argument("--num_view", default = 3, type = int, help = "The number of src views")
    parser.add_argument("--max_w", default = 640, type = int, help = "Max width in training")
    parser.add_argument("--max_h", default = 512, type = int, help = "Max height in training")
    parser.add_argument("--interval_scale", default = 1.06, type = float, help = "Depth interval")

    parser.add_argument("--log_dir", default = './checkpoints', help = "The path to save checkpoint/logs")
    parser.add_argument("--resume", action = "store_true", help = "Continue to train the model")

    parser.add_argument("--summary_interval", default = 20, type = int, help = "Print and summary interval")
    parser.add_argument("--save_interval", default = 1, type = int, help = "Save checkpoint interval")
    parser.add_argument("--seed", default = 1, type = int, metavar = "S", help = "Random seed")

    # check
    args = parser.parse_args()
    if args.resume:
        assert args.mode == "train"
    if args.test_path is None:
        args.test_path = args.train_path

    summary_dir = os.path.join(args.log_dir, f"{args.model}-{args.dataset}-{args.regularization}-{args.normalization}", "summarys")
    save_dir = os.path.join(args.log_dir, f"{args.model}-{args.dataset}-{args.regularization}-{args.normalization}", "weights")
    
    if args.multigpu:
        summary_dir = os.path.join(args.log_dir, f"multi-{args.model}-{args.dataset}-{args.regularization}-{args.normalization}", "summarys")
        save_dir = os.path.join(args.log_dir, f"multi-{args.model}-{args.dataset}-{args.regularization}-{args.normalization}", "weights")

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    return args
    
if __name__ == "__main__":
    args = parse_args()
    print(args)