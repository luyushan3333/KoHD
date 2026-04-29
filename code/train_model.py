# -*- coding: utf-8 -*-

import argparse
import os
import sys
import math
from sklearn.metrics import roc_auc_score, jaccard_score
from dataset import ValGenerator, RandomGenerator, ImageToImage2D,read_text
from transformers import AutoTokenizer, AutoModel
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', '-c', default='Config_covid19', metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file')
parser.add_argument('--gpu', '-g', default='0', metavar='cuda',
                    type=str,
                    help='device id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.cfg_path == "Config_monuseg":
    import config.Config_monuseg as config
elif args.cfg_path == "Config_MosMedPlus":
    import config.Config_MosMedPlus as config
elif args.cfg_path == "Config_kvasir":
    import config.Config_kvasir as config
else:
    import config.Config_covid1919 as config

import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch.backends import cudnn
from my_model import UNet
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, print_summary
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE
from thop import profile


model_name="emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


data=pd.read_excel(config.train_text)
texts=data['Description']

if config.task_name=='MosMedplus' or config.task_name=='Covid19':
    infection_pattern = r"(Bilateral|Unilateral) pulmonary infection"
    area_num_pattern = r"(one|two|three|four|five|six|seven|eight|nine) infected"
    area_pattern = r"((?:upper|middle|lower|all)(?:\s(?:upper|middle|lower|all))*)\s(left|right)"

    entities = set()
    edges = []
    G = nx.Graph()
    for text in texts:

        infection_match = re.search(infection_pattern, text)
        if infection_match:
            infection_type = infection_match.group()
            G.add_node(infection_type, type="infection")
            entities.add(infection_type)
        else:
            print('#######', text)

        area_num_match = re.search(area_num_pattern, text)
        if area_num_match:
            area_num = area_num_match.group()
            if area_num == 'one infected':
                area_num = area_num + ' area'
            else:
                area_num = area_num + ' areas'
            entities.add(area_num)
            edges.append((infection_type, area_num))
        else:
            print('!!!!!!!!!!', text)
        area_num_node = f"{area_num}"
        G.add_node(area_num_node, type="infection_count")

        area_matches = re.findall(area_pattern, text)
        lung_areas = [" ".join(match).strip() for match in area_matches]  # ["all left lung", "middle right lung"]

        for area in lung_areas:
            if area:
                G.add_node(area, type="lung_region")
                entities.add(area)
                G.add_edge(area_num_node, area)
                G.add_edge(infection_type, area)
                edges.append((infection_type, area_num))
                edges.append((infection_type, area))
                edges.append((area_num, area))
else:
    entities = set()
    edges = []
    G = nx.Graph()
    for text in texts:
        text = text[:-1]
        sentences = text.split(",")
        num = sentences[0]
        G.add_node(num, type="number") 
        entities.add(num)

        size = sentences[1]
        entities.add(size)
        edges.append((num, size))
        G.add_node(size, type="size")

        shape = sentences[2]
        entities.add(shape)
        G.add_node(shape, type="shape")
        edges.append((num, shape))
        edges.append((size, shape))    


entity_list = list(entities)
entity_to_index = {entity: i for i, entity in enumerate(entity_list)}

num_entities = len(entity_list)
adjacency_matrix = torch.zeros((num_entities, num_entities))
edge_freq = defaultdict(int)
for src, tgt in edges:
    i, j = entity_to_index[src], entity_to_index[tgt]
    adjacency_matrix[i, j] = 1  
    adjacency_matrix[j, i] = 1  
    edge_freq[(src, tgt)] += 1
    edge_freq[(tgt, src)] += 1

edge_index = []
edge_weight = []
for (src, tgt), freq in edge_freq.items():
    edge_index.append([entity_to_index[src], entity_to_index[tgt]]) 
    edge_weight.append(freq)  


def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()  


feature_matrix = torch.stack([encode_text(entity) for entity in entity_list])
adj= adjacency_matrix.nonzero().t()

edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # (2, num_edges)
edge_weight = torch.tensor(edge_weight, dtype=torch.float)  # (num_edges,)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feature_matrix=feature_matrix.to(device)
edge_index=edge_index.to(device)
edge_weight=edge_weight.to(device)
model=UNet(feature_matrix, edge_index, edge_weight)
model = model.to(device)


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr
def show_image_with_dice(predict_save, labs):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    return dice_pred, iou_pred

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '' + \
                   'latest_model.pth.tar'
    logger.info('\t Saving to {}'.format(filename))
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    print(config.task_name)
    if config.task_name == 'MoNuSeg' or config.task_name == 'MosMedplus' or config.task_name == 'Kvasir':
        train_text = read_text(config.train_text)
        val_text = read_text(config.val_text)
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
    elif config.task_name == 'Covid19':
        train_text = read_text(config.train_text)
        val_text = read_text(config.val_text)

        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=16,
                              pin_memory=True)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=16,
                            pin_memory=True)
                          
    lr = config.learning_rate
    logger.info(model_type)

    model=UNet(feature_matrix, edge_index, edge_weight)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.lr == 'cosineLR':
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    elif config.lr == 'exp':
        lambda1 = lambda epoch: max(0.99**epoch, 0.1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    elif config.lr == 'cosine':
        warm_up_steps = 0
        warm_up_with_cosine_lr = lambda step: step / warm_up_steps if step <= warm_up_steps and warm_up_steps!=0 else 0.5 * (math.cos((step - warm_up_steps) /(config.epochs - warm_up_steps) * math.pi) + 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif config.lr == 'poly':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(train_loader) * config.epochs)) ** 0.99)

    print(config.lr)
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    epoch = 0

    if config.resume:
        checkpoint = torch.load(config.resume_path, map_location='cpu')
        # print(type(checkpoint), type(checkpoint['model']), checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model = model.cuda()

    if config.resume:
        checkpoint = torch.load(config.resume_path, map_location='cpu')
        logger.info('resume path: {}'.format(config.resume_path))
        print(model.load_state_dict(checkpoint['state_dict']))
        
    if torch.cuda.device_count() > 1:
        logger.info("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    if config.resume:
        print(optimizer.load_state_dict(checkpoint['optimizer']))
        print(lr_scheduler.load_state_dict(checkpoint['lr_scheduler']))
        epoch = checkpoint['epoch']
        print("resume optimizer and lr scheduler successfuly")
    else:
        epoch = -999

    max_dice = 0.0
    for epoch in range(max(0, epoch+1), config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch}]")

        train_one_epoch(config, train_loader_tqdm, model, criterion, optimizer, writer, epoch, None, model_type, logger)  # sup

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            
            val_loss, val_dice = train_one_epoch(config, val_loader, model, criterion,optimizer, writer, epoch, lr_scheduler, model_type, logger)


        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 0:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        save_checkpoint({'epoch': epoch,
                        'best_model': False,
                        'model': model_type,
                        'state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        'optimizer': optimizer.state_dict()}, config.model_path)

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':

    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(tensorboard=True)