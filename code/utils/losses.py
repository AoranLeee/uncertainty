import torch
from torch.nn import functional as F
import numpy as np

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

def softmax_ce_loss(input_logits, target_logits, threshold=0.6):
    """Takes softmax on both sides and returns conf-cross entropy loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    #计算ce loss,学生，教师
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    weight = target_softmax.max(1)[0] #计算每个像素的最大概率值
    mask = (weight >= threshold)

    ce_loss = F.cross_entropy(input_logits, torch.argmax(target_softmax, dim=1), reduction="none")
    ce_loss = ce_loss * weight

    return ce_loss[mask].mean()

def softmax_uac_loss(output_logits_list, ema_logits,uncertainty_list):
    """
    3D一致性损失函数 l_crc 实现
    
    参数：
        p (Tensor): 参考预测logits，形状 (B, C, H, W, D)
        pr_list (list of Tensor): 扰动预测logits列表，每个元素形状 (B, C, H, W, D)
        uncertainty_list (list of Tensor): 不确定性Ur列表，每个元素形状 (B, H, W, D)
        
    返回：
        loss (Tensor): 标量损失值
    """
    total_ce_term = 0.0
    total_ur_term = 0.0
    
    # 将参考预测转换为概率分布
    p_prob = F.softmax(ema_logits, dim=1)  # (B, C, H, W, D)
    
    for pr, ur in zip(output_logits_list, uncertainty_list):
        # 交叉熵: -sum(p * log_pr) over class维度
        ce_r = F.cross_entropy(pr, torch.argmax(p_prob, dim=1), reduction="none")
        ce_r=ce_r.unsqueeze(1)#(B, C, H, W, D)
        
        # 分母: exp(ur) + eps，防止除零
        exp_ur = torch.exp(ur).clamp(min=1e-8)  # (B, C, H, W, D)
        
        # --- 计算 CE / exp(Ur) ---
        ce_term = torch.sum((ce_r / exp_ur).clamp(min=1e-8))  # scalar
        total_ce_term += ce_term
        
        # --- 累加 Ur 项 ---
        ur_term = torch.sum(ur)  # scalar
        total_ur_term += ur_term
    
    total_loss = total_ce_term + total_ur_term
    total_loss = total_loss/(2*torch.numel(ce_r)*len(output_logits_list))
    return total_loss, total_ce_term, total_ur_term