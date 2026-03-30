import torch
from torch.nn import functional as F
import numpy as np

def entropy_uncertainty(ema_output_logits,output_logits=None):
    #通过熵值预测不确定度,传入模型输出的logits,这里传入的是两个辅助教师的均值
    preds = torch.zeros([ema_output_logits.shape[0], 2, 112, 112, 80]).cuda()
    preds = ema_output_logits
    preds = F.softmax(preds, dim=1)
    uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)

    return uncertainty

def uac_uncertainty(ema_output_logits,output_logits_list):
    #通过KL散度预测不确定度,传入模型输出的logits
    #output_logits:强处理数据预测结果，ema_output_logits：不处理数据预测结果

    # p_bar 是 pr 和 p 的均值
    p = F.softmax(ema_output_logits, dim=1)
    uncertainty = []
    for pr in output_logits_list:
        pr = F.softmax(pr, dim=1)
        p_bar = (pr + p) / 2.0  # 示例：动态计算 p_bar
        
        # 确保数值稳定性（避免除以零或对数负数）
        eps = 1e-8
        term1 = torch.log((p_bar + eps) / (pr + eps))
        term2 = torch.log((p_bar + eps) / (p + eps))
        
        Ur = p_bar * (term1 + term2)  # 形状 (B, C, H, W, D)
        #Ur = Ur.sum(dim=1)  # 沿类别维度求和 → (B, H, W, D)
        #uncertainty.append(Ur.mean(dim=1))
        uncertainty.append(Ur)
    return uncertainty