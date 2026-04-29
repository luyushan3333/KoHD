import random
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Callable
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import os
import torch.nn as nn
import numpy as np
import re

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images
def pad_to_length(arrays, target_length):
    #padded_arrays = []
    if arrays[-1]==102:
        arrays.pop()
    if arrays[-1]==119:
        arrays.pop()
    for i in range(target_length-len(arrays)):
        arrays.append(arrays[i])
    return arrays

infection_pattern = r"(No|Bilateral|Unilateral) pulmonary infection"  # 感染类型
area_num_pattern = r"(zero|one|two|three|four|five|six|seven|eight|nine) infected"  # 感染区域数量
area_pattern = r"((?:upper|middle|lower|all)(?:\s(?:upper|middle|lower|all))*)\s(left|right)"


model_name="emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224,) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.data_name=task_name
       
        if self.data_name== "MoNuSeg" or self.data_name == 'MosMedplus' or self.data_name == 'Kvasir':
            self.output_path = os.path.join(dataset_path, 'labelcol')
            self.input_path = os.path.join(dataset_path, 'img')

        else:
            self.input_path = os.path.join(dataset_path, 'Images')# Covid19
            self.output_path = os.path.join(dataset_path, 'Ground-truths')# Covid19
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))


    def __len__(self):
        return len(os.listdir(self.input_path))

    def encode_text(self,text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def __getitem__(self, idx):
        
        if self.data_name == "MoNuSeg" :
            image_filename = self.images_list[idx]  # MoNuSeg
            mask_filename = image_filename[: -3] + "png"  # MoNuSeg
            text = self.rowtext[mask_filename]
            entities =[]
            text = text[:-1]
            sentences = text.split(",")
            for i in sentences:
                entities.append(i)

            feature_matrix2 = torch.stack([self.encode_text(entity) for entity in entities])
            padded_matrix = torch.cat([feature_matrix2, torch.zeros(3-len(feature_matrix2), 768)], dim=0)

        elif self.data_name == 'Kvasir':
            image_filename = self.images_list[idx] 
            mask_filename = image_filename
            text = self.rowtext[mask_filename]
            entities =[]
            text = text[:-1]
            sentences = text.split(",")
            for i in sentences:
                entities.append(i)

            feature_matrix2 = torch.stack([self.encode_text(entity) for entity in entities])
            padded_matrix = torch.cat([feature_matrix2, torch.zeros(3-len(feature_matrix2), 768)], dim=0)


        elif self.data_name == 'MosMedplus':
            image_filename = self.images_list[idx] 
            mask_filename = image_filename
            text = self.rowtext[mask_filename]
            text = str(text)
            infection_match = re.search(infection_pattern, text)
            area_num_match = re.search(area_num_pattern, text)
            area_matches = re.findall(area_pattern, text)
            entities = []
            lung_areas = [" ".join(match).strip() for match in area_matches]
            if infection_match is not None:
                infection_type = infection_match.group()
                entities.append(infection_type)
            if area_num_match is not None:
                area_num = area_num_match.group()
                if area_num=='one infected':
                    area_num=area_num+' area'
                else:
                    area_num=area_num+' areas'
                entities.append(area_num)

            if len(area_matches) > 0:
                for area in lung_areas:

                   entities.append(area)

            entity_list = list(entities)

            feature_matrix2 = torch.stack([self.encode_text(entity) for entity in entities])
            padded_matrix = torch.cat([feature_matrix2, torch.zeros(4-len(feature_matrix2), 768)], dim=0)
            
        else:
            mask_filename = self.mask_list[idx]  # Covid19
            image_filename = mask_filename.replace('mask_', '')  # Covid19
            text = self.rowtext[mask_filename]

            text = str(text)

            infection_match = re.search(infection_pattern, text)
            area_num_match = re.search(area_num_pattern, text)
            area_matches = re.findall(area_pattern, text)
            entities = []

            lung_areas = [" ".join(match).strip() for match in area_matches]
            if infection_match is not None:
                infection_type = infection_match.group()
                entities.append(infection_type)
            if area_num_match is not None:
                area_num = area_num_match.group()
                if area_num=='one infected':
                    area_num=area_num+' area'
                else:
                    area_num=area_num+' areas'
                entities.append(area_num)


            if len(area_matches) > 0:
                for area in lung_areas:                   
                    entities.append(area)
            entity_list = list(entities)


            feature_matrix2 = torch.stack([self.encode_text(entity) for entity in entity_list])        
            padded_matrix = torch.cat([feature_matrix2, torch.zeros(4-len(feature_matrix2), 768)], dim=0)


        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        sample = {'image': image, 'label': mask,'text_f':padded_matrix}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()
class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label,c = sample['image'], sample['label'],sample['text_f']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3) 
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)

        sample = {'image': image, 'label': label,'text_f':c}
        return sample

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label,c = sample['image'], sample['label'],sample['text_f']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3) 
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)

        sample = {'image': image, 'label': label, 'text_f':c }
        return sample

import pandas as pd
from torch.utils.data import Dataset, DataLoader
def read_text(filename):
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:  # Gets the index of the row number and traverses it
        text[df.Image[i]] = df.Description[i]
    return text  # return dict (key: values)


