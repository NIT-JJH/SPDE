import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from Model.CATNet.CatNet import CatNet
from Model.PICR_Net.model.backbone import PICR_Net
from Model.LSNet import LSNet
from config import opt



dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

#load the model
model = CatNet()

#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load(opt.save_path + 'Net_epoch_best.pth'))
model.cuda()
model.eval()


#test
test_mae = []
if opt.task =='RGBT':
    from rgbt_dataset import test_dataset
    test_datasets = ['VT800','VT1000','VT5000']
elif opt.task == 'RGBD':
    from rgbd_dataset import test_dataset
    test_datasets = ['USOD10K','USOD']
    # test_datasets = ['USOD']
else:
    raise ValueError(f"Unknown task type {opt.task}")

for dataset in test_datasets:
    mae_sum  = 0
    save_path = opt.save_path + dataset + '\\'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if opt.task == 'RGBT':
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        ti_root=dataset_path + dataset +'/T/'
    elif opt.task == 'RGBD':
        image_root = dataset_path + dataset + '\\TE\\RGB\\'
        gt_root = dataset_path + dataset + '\\TE\\GT\\'
        ti_root = dataset_path + dataset + '\\TE\\depth\\'
    else:
        raise ValueError(f"Unknown task type {opt.task}")
    test_loader = test_dataset(image_root, gt_root, ti_root, opt.testsize)
    print(len(test_loader))
    for i in range(test_loader.size):
        image, gt, ti, name  = test_loader.load_data()
        gt = gt.cuda()
        image = image.cuda()
        ti = ti.cuda()
        # if opt.task == 'RGBD':
        #     ti = torch.cat((ti, ti, ti),dim=1)
        res = model(image, ti)[0]
        # res = F.interpolate(res, (480, 640), mode='bilinear', align_corners=True)
        predict = torch.sigmoid(res)
        # depth_image = torch.sigmoid(depth)
        # gt = F.interpolate(gt, (480, 640), mode='bilinear', align_corners=True)
        predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
        # depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()+ 1e-8)
        mae = torch.sum(torch.abs(predict - gt)) / torch.numel(gt)
        # mae = torch.abs(predict - gt).mean()
        mae_sum = mae.item() + mae_sum
        # predict = F.interpolate(predict, (480, 640), mode='bilinear', align_corners=True)
        predict = predict.data.cpu().numpy().squeeze()
        # depth_image = depth_image.data.cpu().numpy().squeeze()
        # print(predict.shape)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name, predict*255)
        # cv2.imwrite(save_path+'dpeth'+name, depth_image*255)
    test_mae.append(mae_sum / test_loader.size)
print('Test Done!', 'MAE', test_mae)