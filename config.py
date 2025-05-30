import argparse
parser = argparse.ArgumentParser()
# train/val
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument("--task", type=str, default='RGBD', help='which task')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.3, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='epoch number of decay rate')
parser.add_argument('--c', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--cc', type=str, default=None, help='train from checkpoints')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints') 
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--lr_train_root', type=str, default='D:\\Workplace\\USOD\\USOD10K\\TR', help='the train images root')
parser.add_argument('--lr_val_root', type=str, default='D:\\Workplace\\USOD\\USOD10K\\TE', help='the val images root')
parser.add_argument('--save_path', type=str, default='D:\\Workplace\\USOD\\pth\\SPDE\\', help='the path to save models and logs')
# test(predict)
parser.add_argument('--testsize', type=int, default=224, help='testing size')
# parser.add_argument('--test_path',type=str,defaultc='',help='test dataset path')
parser.add_argument('--ori_scale', type=int, default=224, help='testing size')
parser.add_argument('--test_path',type=str,default='D:\\Workplace\\USOD\\',help='test dataset path')
# parser.add_argument('--test_path',type=str,default='',help='test dataset path')
opt = parser.parse_args()