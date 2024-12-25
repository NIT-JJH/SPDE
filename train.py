import math
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import make_grid
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from config import opt
from torch.cuda import amp
import random
# build the model
# from Model.LSNet import LSNet
# from Model.CATNet.CatNet import CatNet
# from Model.PICR_Net.model.build_model import PICR_Net
from Model.SPDE.build_model import PICR_Net
# from Model.Popnet.model import PopNet
# from Model.TC_USOD.USOD_Net import ImageDepthNet
from Loss.loss import smooth_normal_loss, similarity_loss
import argparse
# from afl import AdaptiveFocalLossSigmoid
# from Loss.ohemCELoss import OhemCELoss
from Loss.pytorch_ssim import SSIM
# parser = argparse.ArgumentParser()
# parser.add_argument('--pretrained_model', type=str, default='/home/jjh/Workplace/USOD/Model/TC_USOD/pretrained_model/80.7_T2T_ViT_t_14.pth.tar', help = 'pretrain_model_path')
# args = parser.parse_args()

# model.load_pre()
# if (opt.load is not None):
#     model.load_state_dict(torch.load(opt.load))
#     print('load model from ', opt.load)

# set loss function
# def SPloss(pred, epoch=1, config=None):
#     mul = torch.exp((1-(epoch-1) / opt.epoch))
#     loss_map = torch.abs(pred - 0.5)
#     loss_map = torch.pow(loss_map, mul)
#     loss = pow(0.5, mul) - loss_map.mean()
#     return loss
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner 

class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, pred, label, epoch):
        sigmoid_x = pred.sigmoid()
        mul = 2 ** (1.0 - (epoch * 1.0) / (opt.epoch * 1.0))
        # cos = 1 / 2 * (1 + math.cos(math.pi * ((epoch * 1.0) / (opt.epoch * 1.0))))
        # mul = 2 ** cos
        # mul = 2
        loss_map = 0.5 - (sigmoid_x-0.5).abs().pow(mul)
        # weights = torch.abs(sigmoid_x - label)
        # loss_map = (0.5 - (sigmoid_x-0.5).abs().pow(mul)) / (1 -  weights.pow(mul))
        loss = loss_map.mean()

        return loss

class IOUBCE_loss1(nn.Module):
    def __init__(self):
        super(IOUBCE_loss1, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, taeget_scale):
            bce = self.nll_lose(inputs, targets)
            pred = torch.sigmoid(inputs)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b

def D(mu, L, pred):
    # mul = 2 ** (1.0 - (epoch * 1.0) / (opt.epoch * 1.0))
    # return torch.exp((L-mu) ** 2 / (((pred-0.5) ** 2).mean() + 1e-8))
    mul =  0.5 - (pred-0.5).pow(2)
    return mul.mean()

class IOUBCE_loss3(nn.Module):
    def __init__(self):
        super(IOUBCE_loss3, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.size()
        loss = []
        
        for inputs, targets in zip(input_scale, taeget_scale):
            bce = self.nll_lose(inputs, targets)
            pred = torch.sigmoid(inputs)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1) 
            d1 = D(0, bce, pred)
            loss.append((1 + d1)*(1- IOU + bce))
        total_loss = sum(loss)
        return total_loss / b 

class IOUBCE_loss2(nn.Module):
    def __init__(self, Greater=False):
        super(IOUBCE_loss2, self).__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        # self.thresh = thresh
        self.Greater = Greater

    def forward(self, input_scale, target_scale):
        b, _, _, _ = input_scale.size()
        loss = []
        thresh_list = []
        # thresh = F.binary_cross_entropy_with_logits(input_scale.clone().detach(), target_scale.clone().detach(), reduction='none')
        for inputs, targets in zip(input_scale, target_scale):
            bce = self.nll_loss(inputs, targets) # []
            pred = torch.sigmoid(inputs)
            # Reshape tensors for correct dimensionality
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1 - IOU + bce)
        # loss_tensor = torch.tensor(loss, device='cuda')
        thresh = sum(loss) / b
        if self.Greater:
            loss_hard  = list(filter(lambda x: x > thresh, loss))
        else: 
            loss_hard = list(filter(lambda x: x < thresh, loss))
 
        if len(loss_hard) > 0:
            total_loss = sum(loss_hard) / len(loss_hard)
        else:
            total_loss = torch.tensor(0.0, device=loss.device)
                                                                              
        return total_loss

class IOUBCEWithoutLogits_loss(nn.Module): 
    def __init__(self):
        super(IOUBCEWithoutLogits_loss, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, target_scale):
        b,c,h,w = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, target_scale):

            bce = self.nll_lose(inputs,targets)

            inter = (inputs * targets).sum(dim=(1, 2))
            union = (inputs + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b

### dice_loss
class DICELOSS(nn.Module):
    def __init__(self):
        super(DICELOSS, self).__init__()
    def forward(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

# BBA
def tesnor_bound(img, ksize):
    ''' 
    :param img: tensor, B*C*H*W
    :param ksize: tensor, ksize * ksize
    :param 2patches: tensor, B * C * H * W * ksize * ksize
    :return: tensor, (inflation - corrosion), B * C * H * W
    '''
    B, C, H, W = img.shape
    pad = int((ksize - 1) // 2)
    img_pad = F.pad(img, pad=[pad, pad, pad, pad], mode='constant',value = 0)
    # unfold in the second and third dimensions
    patches = img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    corrosion, _ = torch.min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = torch.max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    return inflation - corrosion

def compute_attention(pred, mask):
    pred = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    return bce

# train function
# def train(train_loader, model, optimizer, epoch, save_path, step, prev_losses1, prev_losses2, prev_losses3, prev_losses4, prev_losses5, prev_losses6):
def train(train_loader, model, optimizer, epoch, save_path, step):
    model.train()
    loss_all = 0
    epoch_step = 0
    total_step = len(train_loader)
    IOUBCE = IOUBCE_loss1().cuda()
    # Focal = Focal_loss().cuda()
    SP = SPLoss().cuda()
    IOUBCE2 = IOUBCE_loss2().cuda()
    IOUBCE3 = IOUBCE_loss3().cuda()
    # ssim_loss = SSIM(window_size=7, size_average=True).cuda()
    # dice = DICELOSS().cuda()
    IOUBCEWithoutLogits = IOUBCEWithoutLogits_loss().cuda()
    # # afl = AdaptiveFocalLossSigmoid().cuda()
    # if (epoch % 3) == 1:
    # # # if (epoch % 5) == 1 or (epoch % 5) ==2:
    #     prev_losses6.clear()
    #     prev_losses5.clear()
    #     prev_losses4.clear()
    #     prev_losses3.clear()
    #     prev_losses2.clear()
    try:
        for i, (images, gts, dep) in enumerate(tqdm(train_loader), start=1):
            
            optimizer.zero_grad()
            images = images.cuda()
            dep = dep.cuda()
            gts = gts.cuda()
            # if opt.task == 'RGBD':
            #     tis = torch.cat((dep, dep, dep), dim=1)
            gts2 = F.interpolate(gts, (112, 112))
            gts3 = F.interpolate(gts, (56, 56))
            gts4 = F.interpolate(gts, (28, 28))
            gts5 = F.interpolate(gts, (14, 14))
            gts6 = F.interpolate(gts, (7, 7))
            # bound = tesnor_bound(gts, 3).cuda()
            # bound2 = F.interpolate(bound, (112, 112))
            # bound3 = F.interpolate(bound, (56, 56))
            smap, sides, depth_pop = model(images, dep)
            # predict_e = tesnor_bound(torch.sigmoid(e), 3)
            # smap, sides = model(images, dep)
#             if epoch == 1:
#                 loss1 = CE(out[0], gts)
#                 prev_losses6.append(loss1.item())
#                 # print('prev_losses1', len(prev_losses1))
#                 # print('prev_losses2', len(prev_losses6))
#                 loss2 = CE(out[1][0], gts6)
#                 prev_losses2.append(loss1.item())

#                 loss3 = CE(out[1][1], gts5)
#                 prev_losses3.append(loss3.item())

#                 loss4 = CE(out[1][2], gts4)
#                 prev_losses4.append(loss4.item())

#                 loss5 = CE(out[1][3], gts3)
#                 prev_losses5.append(loss5.item())
#             else:
#                 # print('prev_losses6', len(prev_losses6))
#                 CE1 = torch.nn.BCEWithLogitsLoss(weight=sample_weight(out[0], gts, epoch, prev_losses6)[0]).cuda()
#                 loss1 = CE1(out[0], gts)
#                 prev_losses6.append(CE(out[0], gts).item())

#                 CE2 = torch.nn.BCEWithLogitsLoss(weight=sample_weight(out[1][0], gts6, epoch, prev_losses2)[0]).cuda()
#                 loss2 = CE2(out[1][0], gts6)
#                 prev_losses2.append(CE(out[1][0], gts6).item())

#                 CE3 = torch.nn.BCEWithLogitsLoss(weight = sample_weight(out[1][1], gts5, epoch, prev_losses3)[0]).cuda()
#                 loss3 = CE3(out[1][1], gts5)
#                 prev_losses3.append(CE(out[1][1], gts5).item())

#                 CE4 = torch.nn.BCEWithLogitsLoss(weight = sample_weight(out[1][2], gts4, epoch, prev_losses4)[0]).cuda()
#                 loss4 = CE4(out[1][2], gts4)
#                 prev_losses4.append(CE(out[1][2], gts4).item())

#                 CE5 = torch.nn.BCEWithLogitsLoss(weight = sample_weight(out[1][3], gts3, epoch, prev_losses5)[0]).cuda()
#                 loss5 = CE5(out[1][3], gts3)
#                 prev_losses5.append(CE(out[1][3], gts3).item())
            """DSU"""
            if (epoch % 5) == 1 or (epoch % 5) == 3:
            # if (epoch % 3) == 1 :
                loss1 = IOUBCE2(smap, gts)
                loss2 = IOUBCE2(sides[0], gts6)
                loss3 = IOUBCE2(sides[1], gts5)
                loss4 = IOUBCE2(sides[2], gts4)
                loss5 = IOUBCE2(sides[3], gts3)
            elif (epoch % 5) == 2 or (epoch % 5) == 4:
            # # if (epoch % 5) == 2 or (epoch % 5) == 4:
            # else:
                loss1 = IOUBCE3(smap, gts)
                # loss2 = IOUBCE3(sides, gts)
                loss2 = IOUBCE3(sides[0], gts6)
 
                loss3 = IOUBCE3(sides[1], gts5)
                
                loss4 = IOUBCE3(sides[2], gts4)

                loss5 = IOUBCE3(sides[3], gts3)
            else:
                loss1 = IOUBCE(smap, gts)
                loss2 = IOUBCE(sides[0], gts6)
                loss3 = IOUBCE(sides[1], gts5)
                loss4 = IOUBCE(sides[2], gts4)
                loss5 = IOUBCE(sides[3], gts3) 
            """"""
            # loss1 = IOUBCE(smap, gts)
            # loss2 = IOUBCE(sides[0], gts6)
            # loss3 = IOUBCE(sides[1], gts5)
            # loss4 = IOUBCE(sides[2], gts4)
            # loss5 = IOUBCE(sides[3], gts3)
            pre_depth = (depth_pop - depth_pop.min()) / (depth_pop.max() - depth_pop.min() + 1e-8)
            # print(pre_depth.shape)
            loss_sal_dep = smooth_normal_loss(pre_depth * gts)
            loss_nosal_dep = smooth_normal_loss(pre_depth * (torch.ones(gts.shape).cuda() - gts))
            loss_sal_nosal_dep = similarity_loss(pre_depth * gts, pre_depth * (torch.ones(gts.shape).cuda() - gts))
            # loss_e = IOUBCEWithoutLogits(predict_e, bound)
            loss_SP = SP(smap, gts, epoch)
           
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss_sal_dep + loss_nosal_dep + loss_sal_nosal_dep + loss_SP

            # loss = IOUBCE(out, gts)
            loss.backward()
            optimizer.step()
            step = step + 1
            epoch_step = epoch_step + 1
            loss_all = loss.item() + loss_all
            if i % 10 == 0 or i == total_step or i == 1:
                # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_sod: {:.4f},'
                #       'loss_bound: {:.4f},loss_trans: {:.4f}'.
                #       format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item(),
                #              loss_sod.item(),loss_bound.item(), loss_trans.item()))
                # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f},'
                
                #       format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item(),
                #              ))
                # logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_sod: {:.4f},'
                #               'loss_bound: {:.4f},loss_trans: {:.4f} '.
                #              format(epoch, opt.epoch, i, total_step, loss.item(),
                #                     loss_sod.item(),loss_bound.item(), loss_trans.item()))
                writer.add_scalar('Loss', loss, global_step=step)
                # grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/Ground_truth', grid_image, step)
                # grid_image = make_grid(bound[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/bound', grid_image, step)

                # grid_image = make_grid(body[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/body', grid_image, step)
                res = smap[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('OUT/out', torch.tensor(res), step, dataformats='HW')
                # res = predict_bound0[0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('OUT/bound', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        # return step, prev_losses1, prev_losses2, prev_losses3, prev_losses4, prev_losses5, prev_losses6
        return step
        # if (epoch) % 5 == 0:
        #     torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

# test function
def test(test_loader, model, epoch, save_path, best_mae, best_epoch):
    CE = IOUBCE_loss1().cuda()
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, ti, name = test_loader.load_data()
            gt = gt.cuda()
            image = image.cuda()
            ti = ti.cuda()
            # if opt.task == 'RGBD':
                # ti = torch.cat((ti, ti, ti), dim=1)

            res = model(image, ti)[0]
            loss = CE(res, gt)
            res = torch.sigmoid(res)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = torch.sum(torch.abs(res - gt)) * 1.0 / (torch.numel(gt))
            # print(mae_train)
            mae_sum = mae_train.item() + mae_sum
        # print(test_loader.size)
        mae = mae_sum / test_loader.size
        # print(test_loader.size)
        writer.add_scalar('MAE', torch.as_tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:   
            best_mae = mae  
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
        return best_mae, best_epoch

if __name__ == '__main__':
    # set the device for training
    # cudnn.benchmark = True
    # cudnn.enabled = True
    # 设置随机种子
    seed=42
    setup_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device('cuda')
    print('USE GPU:', opt.gpu_id)

    # bulid model
    model = PICR_Net()
    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    # model_dep =  PopNet()
    # model_dep.cuda()
    # params_depth = model_dep.parameters()
    # optimizer_depth = torch.optim.Adam(params_depth, opt.lr)
    # optimizer = torch.optim.SGD(params, opt.lr, momentum=0.9, weight_decay=5e-4)
    # model1 = PICR_Net()
    # model1.cuda()
    # model1.load_state_dict(torch.load(r'D:\Workplace\USOD\pth\PICRNet\Net_epoch_best.pth'))
    # for parameter in model1.parameters():
    #     parameter.requires_grad = False
    # set the path
    train_dataset_path = opt.lr_train_root

    val_dataset_path = opt.lr_val_root

    save_path = opt.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)                                                                                   

    # load data
    print('load data...')
    if opt.task =='RGBT':
        from rgbt_dataset import get_loader, test_dataset
        image_root = train_dataset_path  + '/RGB/'
        ti_root = train_dataset_path  + '/T/'
        gt_root = train_dataset_path  + '/GT/'
        val_image_root = val_dataset_path + '/RGB/'
        val_ti_root = val_dataset_path + '/T/'
        val_gt_root = val_dataset_path + '/GT/'
    elif opt.task == 'RGBD':
        from rgbd_dataset import get_loader, test_dataset
        image_root = train_dataset_path + '\\RGB\\'
        ti_root = train_dataset_path + '\\depth\\'
        gt_root = train_dataset_path + '\\GT\\'
        val_image_root = val_dataset_path + '\\RGB\\'
        val_ti_root = val_dataset_path + '\\depth\\'
        val_gt_root = val_dataset_path + '\\GT/'
    else:
        raise ValueError(f"Unknown task type {opt.task}")
    
    train_loader = get_loader(image_root, gt_root, ti_root, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=4)
    test_loader = test_dataset(val_image_root, val_gt_root, val_ti_root, opt.trainsize)
    logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Model:")
    # logging.info(model)
    logging.info(save_path + "Train")
    logging.info("Config")
    logging.info(
        'epoch:{}; lr:{}; batchsize:{}; trainsize:{}; clip:{}; decay_rate:{}; load:{};save_path:{}; decay_epoch:{}'.format(
            opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
            opt.decay_epoch))
    writer = SummaryWriter(save_path + 'summary', flush_secs = 30)
    best_mae = 1
    best_epoch = 0
    Sacler = amp.GradScaler(enabled=True)
    print("Start train...")
    step = 0
    prev_losses1 = []
    prev_losses2 = []
    prev_losses3 = []
    prev_losses4 = []
    prev_losses5 = []
    prev_losses6 = []
    # train_lossArr = []
    # val_lossArr = []
    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # step, prev_losses1, prev_losses2, prev_losses3, prev_losses4, prev_losses5, prev_losses6 = train(train_loader, model, optimizer, epoch, save_path, step, prev_losses1, prev_losses2, prev_losses3, prev_losses4, prev_losses5, prev_losses6)
        step = train(train_loader, model, optimizer, epoch, save_path, step)
        best_mae, best_epoch = test(test_loader, model, epoch, save_path, best_mae, best_epoch)