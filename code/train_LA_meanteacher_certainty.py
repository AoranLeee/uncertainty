import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses, uncertainty_calculate
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler,ColorJitter3D,GaussianBlur3D
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UAMT', help='model_name') # model_name
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')#是否使用确定性训练
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')# 指数移动平均衰减率
parser.add_argument('--consistency_type', type=str,  default="uac", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')# 无监督损失的权重
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')# 无监督权重的增长周期
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False 
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)

def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def params_to_mb(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 ** 2)
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

def format_duration_h_m(seconds):
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    return f"{hours} h {mins} mins"

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    ##0到40的迭代次数中，权重从0.0006增长到0.1，40步之后一直维持0.1
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    ema_aux_model = create_model(ema=True)#增加辅助教师

    kernel_size = int(random.random() * 4.95)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(), #随机翻转
                          RandomCrop(patch_size),#随机裁剪成112*112*80
                          ToTensor(),#转化为tensor
                          ]),
                        strong_transform = transforms.Compose([
                          ColorJitter3D(brightness=0.5, contrast=0.5),  # 3D 颜色抖动
                          GaussianBlur3D(kernel_size, sigma=(0.1, 2.0))  # 3D 高斯模糊
                        ])
                        )
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    with open(train_data_path + '/../test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [train_data_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)

    trainloader = DataLoader(db_train,batch_sampler=batch_sampler,num_workers=0,pin_memory=True,worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()
    ema_aux_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    elif args.consistency_type == 'ce':
        consistency_criterion = losses.softmax_ce_loss
    elif args.consistency_type == 'uac':
        consistency_criterion = losses.softmax_uac_loss
    else:
        assert False, args.consistency_type

    uncertainty_evaluate = uncertainty_calculate.uac_uncertainty

    tb_log_dir = os.path.join(snapshot_path, 'log', time.strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(tb_log_dir)
    logging.info("TensorBoard log dir: %s", tb_log_dir)
    logging.info("{} itertations per epoch".format(len(trainloader)))

    #用于验证集
    iter_num = 0
    count_iter = 150
    best_dice = 0.0
    best_iter = 0
    best_jc = 0.0
    best_hd95 = 0.0
    best_asd = 0.0

    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    printed_memory = False
    student_param_mb = params_to_mb(model, trainable_only=False)
    ema_param_mb = params_to_mb(ema_model, trainable_only=False)
    ema_aux_param_mb = params_to_mb(ema_aux_model, trainable_only=False)
    total_param_mb = student_param_mb + ema_param_mb + ema_aux_param_mb
    logging.info(
        'Model Total Params(MB) - student: %.2f, ema: %.2f, ema_aux: %.2f, total: %.2f',
        student_param_mb, ema_param_mb, ema_aux_param_mb, total_param_mb
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()
    train_start_time = time.time()

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        #i_batch是当前批次的索引，sampled_batch是当前批次的数据，包括图像和标签。
        for i_batch, sampled_batch in enumerate(trainloader):
            weak_images = sampled_batch['weak_image']  # 弱增强的图像
            weak_labels = sampled_batch['weak_label']  # 弱增强的标签
            strong_images = sampled_batch['strong_image']  # 强增强的图像
            
            weak_images,weak_labels= weak_images.cuda(), weak_labels.cuda()
            strong_images= strong_images.cuda()

            noise = torch.clamp(torch.randn_like(strong_images) * 0.1, -0.2, 0.2)#生成高斯噪声
            noise1 = torch.clamp(torch.randn_like(strong_images) * 0.2, -0.2, 0.2)#生成高斯噪声
            outputs = model(strong_images + noise1) #学生模型输出

            with torch.no_grad():
                ema_output = ema_model(weak_images) #教师模型输出
                ema_aux_output = ema_aux_model(weak_images + noise)

            # calculate the label loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], weak_labels[:labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], weak_labels[:labeled_bs] == 1)

            T=0.1
            # 数值稳定性：对概率进行裁剪，防止0或1导致NaN
            p_clamped = torch.clamp(ema_output, min=1e-8, max=1.0 - 1e-8)
            # 计算分子和分母的幂次项
            inv_T = 1.0 / T
            p_power = torch.pow(p_clamped, inv_T)
            one_minus_p_power = torch.pow(1.0 - p_clamped, inv_T)
            # 计算锐化后的概率,consistency_dist无监督损失
            sharpened_ema_output = p_power / (p_power + one_minus_p_power + 1e-8)
            uncertainty = uncertainty_evaluate(ema_output,[ema_aux_output,outputs])
            consistency_dist,total_ce_term,total_ur_term = consistency_criterion([ema_aux_output,outputs], sharpened_ema_output, uncertainty)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_loss = consistency_weight * consistency_dist
            loss = 0.5*(loss_seg+loss_seg_dice) + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available() and not printed_memory:
                torch.cuda.synchronize()
                peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                logging.info('GPU Peak Memory (after first batch): %.2f GB', peak_alloc_gb)
                printed_memory = True
            if(epoch_num % 2==0):
                update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            else: 
                update_ema_variables(model, ema_aux_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            if iter_num % count_iter == 0:
                model.eval()
                with torch.no_grad():
                    avg_metric = test_all_case(
                        model,
                        image_list,
                        num_classes=num_classes,
                        patch_size=(112, 112, 80),
                        stride_xy=18,
                        stride_z=4,
                        save_result=False,
                        verbose=False
                    )
                dice, jc, hd95, asd = avg_metric
                logging.info(
                    'iteration %d : val_dice: %.4f val_jc: %.4f val_hd95: %.4f val_asd: %.4f'
                    % (iter_num, dice, jc, hd95, asd)
                )
                writer.add_scalar('val/dice', dice, iter_num)
                if dice > best_dice:
                    best_dice = dice
                    best_iter = iter_num
                    best_jc = jc
                    best_hd95 = hd95
                    best_asd = asd
                    save_path = os.path.join(
                        snapshot_path,
                        f"dice{dice:.4f}_iter{iter_num}.pth"
                    )
                    torch.save(model.state_dict(), save_path)
                    logging.info("save best model to {}".format(save_path))
                model.train()

            writer.add_scalar('uncertainty/mean', uncertainty[0][0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0][0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0][0].min(), iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
            writer.add_scalar('train/total_ce_term', total_ce_term, iter_num)
            writer.add_scalar('train/total_ur_term', total_ur_term, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight))
            if iter_num % 50 == 0:
                image = weak_labels[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = weak_images[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = uncertainty[0][0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/uncertainty', grid_image, iter_num)

                #####
                image = weak_images[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = weak_labels[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    train_time_str = format_duration_h_m(time.time() - train_start_time)
    logging.info(
        'best val result : iter %d : dice: %.4f jc: %.4f hd95: %.4f asd: %.4f',
        best_iter, best_dice, best_jc, best_hd95, best_asd
    )
    logging.info('training time: %s', train_time_str)
    writer.close()
