# -*- coding: utf-8 -*-
import os
import torch
import time


## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
use_cuda = torch.cuda.is_available()

seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)


lr='cosine'
n_channels = 3
n_labels = 1  
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 5000
early_stopping_patience = 100


task_name = 'MoNuSeg' 

#learning_rate = 2e-3 
learning_rate = 1e-3 
batch_size = 4


optimizer = "Adam"
weight_decay = 1e-5

resume = False
train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'

train_text='./datasets/' + task_name + '/Train_text.xlsx'
val_text='./datasets/' + task_name + '/Val_text.xlsx'
test_text='./datasets/' + task_name + '/Test_text.xlsx'
test_text_sp='./datasets/monu_test_split_all.xlsx'

session_name = 'session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name +'/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'

result_path='./result/'+task_name +'/'
xlsx_path='./result/'+task_name +'/'+'test.xlsx'