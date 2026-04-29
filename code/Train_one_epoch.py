# -*- coding: utf-8 -*-
import torch.optim
import os
import time
from loss import  WeightedDiceBCE, iou_on_batch
import numpy as np
import torch.nn as nn
import cv2
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
def create_centroid_mask_batch(binary_images):

    batch_size, H, W = binary_images.shape
    centroid_masks = torch.zeros_like(binary_images)  
    bbox_masks = torch.zeros_like(binary_images)

    for b in range(batch_size):
        binary_image = binary_images[b].cpu().detach().numpy()  
        _, binary_mask = cv2.threshold(binary_image, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask=binary_mask.astype(np.uint8)
    
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

        for i in range(1, num_labels):  
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            x, y, w, h, _ = stats[i]
            bbox_masks[b, y:y + h, x:x + w] = 1.0
            if 0 <= cx < W and 0 <= cy < H:  
                centroid_masks[b, cy, cx] = 1

    return centroid_masks.to(binary_images.device),bbox_masks.to(binary_images.device)  
class RegionCountLoss_box(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

    def forward(self, pred, target):
        center_mask,box_mask=create_centroid_mask_batch(target)
        loss_box=self.criterion(pred,1-(1-target)*(1-box_mask))

        return loss_box
class RegionCountLoss_point(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

    def forward(self, pred, target):
        center_mask,box_mask=create_centroid_mask_batch(target)
        loss_box=self.criterion(pred,center_mask)
        return loss_box
    
box_f=RegionCountLoss_box()
point_f=RegionCountLoss_point()
cre_loss=nn.CrossEntropyLoss()
def print_summary(epoch, i, nb_batch, loss,  
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger, data_time_ave):
    '''
        mode = Train or Test
    '''
    current_datetime = time.localtime()
    summary = '   [' + str(mode) + '] {}-{}-{} {}:{}:{} Epoch: [{}][{}/{}]  '.format(current_datetime.tm_year,
        current_datetime.tm_mon,
        current_datetime.tm_mday,
        current_datetime.tm_hour,
        current_datetime.tm_min,
        current_datetime.tm_sec,
        epoch, i, nb_batch)
    string = ''
    string += 'Back_Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += '|| IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.6f}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    string += '(AvgDataTime {:.2f})   '.format(data_time_ave)
    summary += string
    logger.info(summary)

#          Train One Epoch
#=================================================================================
##################################################################################
downsample2 = nn.AvgPool2d(kernel_size=2, stride=2)
downsample4 = nn.AvgPool2d(kernel_size=4, stride=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_one_epoch(config, loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    df=pd.read_excel(config.train_text)
    logging_mode = 'Train' if model.training else 'Val'
    end = time.time()
    time_sum, loss_sum, data_time_sum = 0, 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    dices = []
    dataiter = iter(loader)
    steps = len(loader)
    for i in range(1, steps+1):

        time0 = time.time()
        sampled_batch, name_list = next(dataiter)
        data_time = time.time() - time0


        images, masks, text = sampled_batch["image"], sampled_batch["label"], sampled_batch["text_f"]
        images, masks, text = images.cuda(), masks.cuda(), text.cuda()
        optimizer.zero_grad(set_to_none=True)
        loss_x5,preds,num_pre,x_up3,x_down,feature_map=model(images, text, mode='train')

        if model.training:
            if config.task_name == 'MoNuSeg' :
                num_list=[]
                for ii in name_list:
                    name=ii
                    left2=df[df['Image'] == name]['Description'].values[0]
                    if 'few' in left2:
                        num_list.append(0.0)
                    elif 'moderate' in left2:
                        num_list.append(1.0)
                    else:
                        num_list.append(2.0)
                    x_num = torch.tensor(num_list, dtype=torch.float32, device=device)
            
            elif config.task_name == 'Kvasir':
                num_list=[]
                for ii in name_list:
                    name=ii
                    left2=df[df['Image'] == name]['Description'].values[0]
                    if 'One' in left2:
                        num_list.append(0.0)
                    elif 'Two' in left2:
                        num_list.append(1.0)
                    else:
                        num_list.append(2.0)
                    x_num = torch.tensor(num_list, dtype=torch.float32, device=device)
            elif config.task_name == 'MosMedplus':
                num_list=[]
                for ii in name_list:
                    name=ii
                    left2=df[df['Image'] == name]['Description'].values[0]
                    if 'Unilateral' in left2:
                        num_list.append(0.0)
                    elif 'Bilateral' in left2:
                        num_list.append(1.0)
                    else:
                        num_list.append(2.0)
                    x_num = torch.tensor(num_list, dtype=torch.float32, device=device)
            elif config.task_name == 'Covid19':

                num_list=[]
                for ii in name_list:
                    name='mask_'+ii
                    left2=df[df['Image'] == name]['Description'].values[0]
                    if 'Unilateral' in left2:
                        num_list.append(0.0)
                    elif 'Bilateral' in left2:
                        num_list.append(1.0)
                    x_num = torch.tensor(num_list, dtype=torch.float32, device=device)

        
            mask_down2=downsample2(masks.float())
            mask_down4=downsample4(masks.float())

            loss_g=cre_loss(num_pre,x_num.long())     
            loss_pos=box_f(x_down.squeeze(dim=1),mask_down2)  
            loss_num=point_f(x_up3.squeeze(dim=1),mask_down4) 


            criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
            loss_criterion = criterion(preds, masks.float())  # Loss

            out_loss = loss_criterion+loss_x5+0.1*loss_g+0.3*loss_pos+0.3*loss_num
            
            out_loss.backward()
            optimizer.step()
        else:
            out_loss=0

        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks, preds)

        batch_time = time.time() - end
        if (epoch + 1) % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)

        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        dice_sum += len(images) * train_dice
        data_time_sum += data_time

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, 
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger, 
                          data_time_ave=data_time_sum/config.print_frequency)
            data_time_sum = 0
        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)
        if config.lr == 'poly' and lr_scheduler is not None:
            lr_scheduler.step()

        torch.cuda.empty_cache()

    if lr_scheduler is not None and config.lr != 'poly':
        lr_scheduler.step()

    return average_loss, train_dice_avg
