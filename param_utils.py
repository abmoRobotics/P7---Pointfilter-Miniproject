import argparse

"""Contains utils for arg parsing & misc"""

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_arguments():
    parser = argparse.ArgumentParser()
    # naming / file handling
    parser.add_argument('--name', type=str, default='pcdenoising', help='training run name')
    parser.add_argument('--network_model_dir', type=str, default='./Summary/Models/Train', help='output folder (trained models)')
    parser.add_argument('--trainset', type=str, default='./Dataset/Train', help='training set file name')
    parser.add_argument('--testset', type=str, default='./Dataset/Test', help='testing set file name')
    parser.add_argument('--save_dir', type=str, default='./Dataset/Results', help='')
    parser.add_argument('--summary_dir', type=str, default='./Summary/Models/Train/logs', help='')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--manualSeed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--patch_per_shape', type=int, default=8000, help='')
    parser.add_argument('--patch_radius', type=float, default=0.05, help='')

    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--model_interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')

    # others parameters
    parser.add_argument('--resume', type=str, default='', help='refine model at this path')
    parser.add_argument('--support_multiple', type=float, default=4.0, help='the multiple of support radius')
    parser.add_argument('--support_angle', type=int, default=15, help='')
    parser.add_argument('--gt_normal_mode', type=str, default='nearest', help='')
    parser.add_argument('--repulsion_alpha', type=float, default='0.97', help='')
    parser.add_argument('--use_cuda', type=bool, default='True', help='Setting this false will disable the use of cuda.')

    # evaluation parameters
    parser.add_argument('--eval_dir', type=str, default='./Summary/pre_train_model', help='')
    parser.add_argument('--eval_iter_nums', type=int, default=10, help='')

    return parser.parse_args()
