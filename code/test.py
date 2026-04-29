
import argparse
import os
from PIL import Image
import torch.nn.functional as F
import cv2
from dataset import ValGenerator, RandomGenerator, ImageToImage2D,read_text
from transformers import AutoTokenizer, AutoModel
import networkx as nx
import pandas as pd
import re
from collections import defaultdict
import config.Config_covid19 as config
from sklearn.metrics import roc_auc_score, jaccard_score

from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from my_model import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

parser = argparse.ArgumentParser(description='Test model')
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
    import config.Config_covid19 as config



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



red_color = (255, 0, 0)     # red
blue_color = (0, 0, 255)  # blue
green_color = (0, 255, 0)   # green
size = (224, 224)


path_all=config.result_path
if not os.path.exists(path_all):
    os.makedirs(path_all)


def show_image_with_dice(predict_save, labs):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    return dice_pred, iou_pred


def pred_mix(ground_truth, prediction_mask, original_image):
    gt = (ground_truth > 0).astype(np.uint8)
    pred = (prediction_mask > 0).astype(np.uint8)

    overlay = original_image.copy()

    # False Negative: GT=1, Pred=0 → Red
    overlay[(gt == 1) & (pred == 0)] = red_color

    # False Positive: GT=0, Pred=1 → Blue
    overlay[(gt == 0) & (pred == 1)] = blue_color

    # True Positive: GT=1, Pred=1 → Green
    overlay[(gt == 1) & (pred == 1)] = green_color

    return overlay
def draw_pred_mask_rgb(gt, pred):
    """
    gt, pred: ndarray, shape (H, W), values in {0,1}
    return: RGB mask, shape (H, W, 3)
    """
    H, W = gt.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # TP: green
    rgb[(gt == 1) & (pred == 1)] = [0, 255, 0]

    # FN: red
    rgb[(gt == 1) & (pred == 0)] = [255, 0, 0]

    # FP: blue
    rgb[(gt == 0) & (pred == 1)] = [0, 0, 255]

    # TN stays black
    return rgb


def draw_sub_plot(img, fig, nums, idx, mode="gray"):
    img = cv2.resize(img, size)
    ax = fig.add_subplot(1, nums, idx)
    if mode == "gray":
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.axis('off')
def save_overlay_image(input_image, heatmap, save_path, alpha=0.5, colormap=cv2.COLORMAP_JET):


    if input_image.ndim == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    if input_image.max() <= 1.0:
        input_image = (input_image * 255).astype(np.uint8)


    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze()
    heatmap = np.nan_to_num(heatmap)  
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)


    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.resize(heatmap_color, (input_image.shape[1], input_image.shape[0]))

    overlay = cv2.addWeighted(heatmap_color, alpha, input_image, 1 - alpha, 0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)

    return overlay

def extract_activations_and_gradients(model, input_tensor, target_layer):
    activations, gradients = None, None
    
    def forward_hook(m, inp, out):
        nonlocal activations
        activations = out.detach()
    def backward_hook(m, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    loss_x5,output,num_pre,x_up3,x_down,feature_map=model(input_tensor[0],input_tensor[1])
    loss = output.mean()  
    model.zero_grad()
    loss.backward(retain_graph=True)
    handle_f.remove()
    handle_b.remove()
    return activations, gradients, output
       
def vis_and_save_heatmap(model, input_img, masks, text_token, img_RGB, labs,base_name):
    model.eval()

    loss_x5,output,num_pre,x_up3,x_down,feature_map =  model(input_img, text_token,mode='test')
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (224,224))

    ground_truth = labs.squeeze()
    original_image = torch.squeeze(input_img, 0).cpu().numpy() * 255
    original_image = original_image.transpose(1, 2, 0).astype(np.uint8)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    pred = pred_mix(ground_truth, predict_save, original_image_rgb)
    path=os.path.join(path_all, f'{base_name[:-4]}_co.png')
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, pred_bgr)####plot!!!!!!!!!!!!!!!!!

    mask_uint8 = (predict_save * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8)
    mask_pil.save(os.path.join(path_all, f'{base_name[:-4]}_pred_mask.png'))####plot!!!!!!!!!!!!!!!!!


    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs,)

    return dice_pred_tmp, iou_tmp


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature=feature_matrix.to(device)
edge_index=edge_index.to(device)
edge_weight=edge_weight.to(device)
model=UNet(feature, edge_index, edge_weight)
model = model.to(device)


def read_text(filename):
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values: 
        text[df.Image[i]] = df.Description[i]
    return text  # return dict (key: values)

if __name__ == '__main__':
    data_m={
        'dice':[],
        'iou':[]
    }

    model_path='/MoNuSeg/session_02.01_09h56/models/best_model-RecLMIS.pth.tar'
    checkpoint = torch.load(model_path, map_location='cuda')
    model=UNet(feature, edge_index, edge_weight)

    model=model.to(device)
    load_res = model.load_state_dict(checkpoint['state_dict'], strict=False)
    img_size=224

    task=config.task_name
    tf_test = ValGenerator(output_size=[img_size, img_size])

    test_dataset =config.test_dataset
    test_text=read_text(config.test_text)
    #test_text=read_text(config.test_text_sp)

    test_dataset=ImageToImage2D(test_dataset,task,test_text,tf_test,image_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    test_num = len(test_loader)
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True, dynamic_ncols=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label, text = sampled_batch["image"], sampled_batch["label"], sampled_batch["text_f"]
            lab = test_label.data.numpy()

            test_data, test_label, text = test_data.cuda(), test_label.cuda(), text.cuda()

            input_tensor = (test_data, text)
            target_layer = model.outc
            acts, grads, pred = extract_activations_and_gradients(model, input_tensor, target_layer)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=input_tensor[0].shape[2:], mode='bilinear', align_corners=False)
            heatmap1 = cam.squeeze().cpu().numpy()
            x_np = test_data.squeeze().cpu().numpy().transpose(1, 2, 0)
            path=os.path.join(path_all, f'{names[0][:-4]}_cam.png')
            save_overlay_image(x_np,heatmap1,path, alpha=0.6)####plot!!!!!!!!!!!!!!!!!


            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, test_data, test_label, text, None, lab,str(names[0]))
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            data_m['dice'].append(dice_pred_t)
            data_m['iou'].append(iou_pred_t)            
            torch.cuda.empty_cache()
            pbar.set_postfix({"dice_pred": round(dice_pred/i,4), "iou_pred": round(iou_pred/i, 4)})
            pbar.update()
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)            

    data_f=pd.DataFrame(data_m)
    xlsx_path = config.xlsx_path
    if not os.path.exists(xlsx_path):
        os.makedirs(xlsx_path)

    data_f.to_excel(xlsx_path,index=False)
