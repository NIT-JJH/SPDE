import torch
import torch.nn as nn
import torch.nn.functional as F

class OhemCELoss(nn.Module):
    """
    算法本质
    Ohem本质: 核心思想是取所有损失大于阈值的像素点参与计算, 但是最少也要保证取n_min个
    
    """
    def __init__(self, thresh, Greater=True):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        self.criteria = nn.BCEWithLogitsLoss(reduction='none')
        self.Greater = Greater

    def forward(self, logits, labels):

        # 1. 计算n_min（最少算多少个像素点）
        # n_min的大小， 一个batch的n张h*w的label图的所有的像素点的16分之一
        n_min = labels.numel() // 16
        
        # 2. 交叉熵预测得到loss之后，打平成一维的
        loss = self.criteria(logits, labels).view(-1)
        # print(loss.shape)
        # 3. 所有loss中大于or小于阈值的，这边叫做loss hard， 这些点才参与损失计算
        # 注意：这里是优化了pytorch中Ohem排序的，不然排序太耗时间了
        if self.Greater:
            loss_hard = loss[loss > self.thresh]
        else:
            loss_hard = loss[loss < self.thresh]
        # 4. 如果总数小于了n_min,那么肯定要保证n_min个
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        
        # 5. 如果参与的像素点的个数大于了n_min个，那么这些点都参与计算
        loss_hard_mean = torch.mean(loss_hard)

        # 6. 返回损失的均值
        return loss_hard_mean

# AST
def loss_weight(input, target):

    _, c, w, h = target.size()
    loss_w = F.binary_cross_entropy_with_logits(input.clone().detach(), target.clone().detach(), reduction='none')

    loss_sample_tensor = loss_w.data.clone().detach().mean(dim=1).mean(dim=1).mean(dim=1)
    loss_sample_weight = torch.softmax(1 - loss_sample_tensor, dim=0)

    weight = loss_sample_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, c, w, h)

    return weight, loss_sample_weight

# Self_paced
def sample_weight(input, target, epoch, prev_losses):
    _, c, w, h = target.size()
    loss_w = F.binary_cross_entropy_with_logits(input.clone().detach(), target.clone().detach(), reduction='none')

    loss_sample_tensor = torch.tensor(loss_w.data.clone().detach()).mean(dim=1).mean(dim=1).mean(dim=1).to('cuda')
    
    # 获取上一轮训练的样本损失值
    prev_losses.sort()
    # print('prev_loss', prev_losses)
    # 根据当前训练轮次和批次大小计算阈值索引
    threshold_index = int(len(prev_losses)/2) + epoch 

    if threshold_index < len(prev_losses):
        # 获取阈值索引处的损失值
        threshold = prev_losses[threshold_index]
    else:
        # 如果阈值索引超出损失值张量的大小，则将阈值设置为损失值张量的最大值
        threshold = prev_losses[-1]
    # print(threshold)
    threshold_sample_tensor = torch.tensor(threshold).to('cuda')
    # 根据损失值设置样本权重
    loss_sample_weight = torch.where(loss_sample_tensor > threshold_sample_tensor, torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())

    weight = loss_sample_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, c, w, h)
    # print(weight.shape)
    return weight, loss_sample_weight

# Self_paced
def sample_weight2(input, target, epoch, prev_losses):
    _, c, w, h = target.size()
    loss_w = F.binary_cross_entropy_with_logits(input.clone().detach(), target.clone().detach(), reduction='none')
    print(len(loss_w))
    # 获取上一轮训练的样本损失值
    prev_losses.sort(reverse=True)
    
    # 根据当前训练轮次和批次大小计算阈值索引
    threshold_index = epoch 

    if threshold_index < len(prev_losses):
        # 获取阈值索引处的损失值
        threshold = prev_losses[threshold_index]
    else:
        # 如果阈值索引超出损失值张量的大小，则将阈值设置为损失值张量的最大值
        threshold = prev_losses[-1]
        
    # 根据损失值设置样本权重
    if threshold > loss_w.item():
        loss_sample_weight = 1
    else:
        loss_sample_weight = 0
    
    return loss_sample_weight

"""
self-paced 
"""
class Self_paced_loss(nn.Module):
    def __init__(self):
        super(Self_paced_loss, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale, prev_loss, epoch):
        b,_,_,_ = input_scale.size()
        loss = []
        prev_loss.sort()
        threshold_index = len(prev_loss)
        for inputs, targets in zip(input_scale, taeget_scale):
            bce = self.nll_lose(inputs, targets)
            pred = torch.sigmoid(inputs)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b
    
class Focal_loss(nn.Module):
    def __init__(self):
        super(Focal_loss, self).__init__()
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = 0.25
        self.gamma = 2.0
    
    def forward(self, input, target):
        pt = torch.where(target == 1, input, 1 - input)
        loss = self.BCE_loss(input, target)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * loss
        return focal_loss.mean()

