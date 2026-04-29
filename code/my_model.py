import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch.nn.modules.utils import _pair
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule, self).__init__()
        self.middle_layer_size_ratio = 2
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),  # 2, 2*self.
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    '''forward'''

    def forward(self, x):
        x_avg = self.conv_avg(x)
        x_avg = self.relu_avg(x_avg)
        x_avg = torch.mean(x_avg, dim=1)
        x_avg = x_avg.unsqueeze(dim=1)
        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1).values
        x_max = x_max.unsqueeze(dim=1)
        x_out = x_max+x_avg
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)
        x_output = x_output.transpose(1, 3)
        x_output = self.bottleneck(x_output)
        x_output = x_output.transpose(1, 3)
        y = x_output * x
        return y

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        #x = torch.cat([skip_x, up], dim=1)
        return self.nConvs(x)


class GraphNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(GraphNetwork, self).__init__()
        self.gcn = GCNConv(embed_dim, embed_dim)

    def forward(self, node_features, edge_index, edge_weight):
        new_node_features = self.gcn(node_features, edge_index, edge_weight)
        gcn_weight = self.gcn.lin.weight
        return new_node_features,gcn_weight


class Embeddings(nn.Module):
    def __init__(self, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class FusionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cov1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.cov2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.cov3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4, bias=False)

        self.cov1_f = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cov2_f = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cov3_f = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, f1, f2):
        fusion=f1*f2
        f1 = self.cov1(f1) + self.cov1_f(fusion)

        f1 = self.cov2(f1) + self.cov2_f(fusion)
        f1 = self.cov3(f1) + self.cov3_f(fusion)
        return f1
class text2img(nn.Module):

    def __init__(self, d_model,num_heads,num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads,batch_first=True) for _ in range(num_layers)])


    def forward(self, x, text):
        for layer in self.layers:
            x,weight=layer(x,text,text)
        return x
class img5(nn.Module):
    def __init__(self, d_model,num_heads,num_layers):
        super().__init__()
        self.text1=text2img(d_model,num_heads,num_layers)
        self.text2=text2img(d_model,num_heads,num_layers)
        self.text3=text2img(d_model,num_heads,num_layers)
        self.text4=text2img(d_model,num_heads,num_layers)
        
        self.cov1=nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.cov2=nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.cov3=nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.cov4=nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.cov_img=nn.Conv1d(in_channels=196*3,out_channels=196,kernel_size=3, padding=1)

    def forward(self, x, text1,text2,text3):
        text1=self.cov1(text1.transpose(1, 2)).transpose(1, 2)
        text2=self.cov2(text2.transpose(1, 2)).transpose(1, 2)
        text3=self.cov3(text3.transpose(1, 2)).transpose(1, 2)

        img1=self.text1(x,text1)
        img2=self.text2(x,text2)
        img3=self.text3(x,text3)
        img=torch.cat((img1, img2, img3), dim=1)
        img=self.cov_img(img)
        return img1,img2,img3,img3,img
def multi_positive_infonce(text_feats, pos_mask, temperature=0.1):
    eps = 1e-8

    text_feats = F.normalize(text_feats, dim=-1)

    sim = torch.matmul(text_feats, text_feats.T) / max(temperature, eps)

    sim = sim.clamp(min=-50, max=50)

    exp_sim = torch.exp(sim)

    pos_sum = (exp_sim * pos_mask.float()).sum(dim=1) + eps
    denom = exp_sim.sum(dim=1) + eps

    prob = pos_sum / denom
    prob = prob.clamp(min=eps)

    loss = -torch.log(prob)
    return loss.mean()

class UNet(nn.Module):
    def __init__(self,feature, edge_index, edge_weight, n_channels=3, n_classes=1):
        super().__init__()
        self.feature=feature
        self.edge_index=edge_index
        self.edge_weight=edge_weight

        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels =64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()


        self.gcn = GraphNetwork(embed_dim=768)

        self.cr_text2text1 = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)

        self.norm1 = nn.LayerNorm(768)

        self.text_conv1 = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=3, padding=1)
        self.text_conv2 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.text_conv3 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)

        self.text_conv4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)


        self.resize5 = Embeddings(patch_size=1, img_size=14, in_channels=512)        
        self.cr_textimg5 = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True, dropout=0.1)
        self.cr_imgtext5 = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.cr_imgtext55 = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)

        self.cov_dim1 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0,stride=1)
        self.cov_dim2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0,stride=1)
        self.cov_dim3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0,stride=1)

        #self.block_forward = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1 * 28 * 28, 3))#MosMedplus,MoNuSeg,Kvasir
        self.block_forward = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1 * 28 * 28, 2))#Covid19


        self.fusion0=FusionModule(in_channels=256)
        self.fusion1=FusionModule(in_channels=128)
        self.fusion2=FusionModule(in_channels=64)
       
        self.act = nn.Sigmoid()
        self.img5=img5(d_model=512,num_heads=4,num_layers=2)        
        self.mse_loss = nn.MSELoss(reduction='none')



    def forward(self, x, img_text,mode='train'):
        
        text_graph,gcn_weight = self.gcn(self.feature, self.edge_index, self.edge_weight)
        text_graph = text_graph.expand(x.shape[0], -1, -1)

        weight_text,att=self.cr_text2text1(img_text,text_graph,text_graph)
        weight_text = self.norm1(weight_text)
        text = img_text * weight_text

        text5=self.text_conv4(text.transpose(1, 2)).transpose(1, 2)


        features = {}
        x = x.float()  # x [4,3,224,224]
        x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        x2 = self.down1(x1)  # [4,128,112,112]
        x3 = self.down2(x2)  # [4,256,56,56]
        x4 = self.down3(x3)  # [4,512,28,28]
        x5 = self.down4(x4)  # [4,512,14,14]


        x5_resized = self.resize5(x5)
        img_f5, img_wigh5 = self.cr_imgtext5(x5_resized, text5, text5)
        
        
        loss_rec=torch.tensor(0)
        if mode=='train':
            threshold = 0.8
            text1=img_text[:,0,:]
            text2=img_text[:,1,:]
            text3=img_text[:,2:,:].mean(dim=1)
            sim_matrix1 = torch.matmul(text1, text1.T)  # cosine similarity
            pos_mask1 = sim_matrix1 > threshold
            pos_mask1.fill_diagonal_(False)

            sim_matrix2 = torch.matmul(text2, text2.T)  # cosine similarity
            pos_mask2 = sim_matrix2 > threshold
            pos_mask2.fill_diagonal_(False)
        
            img_text0=text[:,0,:].unsqueeze(dim=1)
            img_text1=text[:,1,:].unsqueeze(dim=1)
            img_text22=text[:,2:,:]

            img1,img2,img3,img4,x5_new=self.img5(x5_resized,img_text0,img_text1,img_text22)
            loss_rec=multi_positive_infonce(x5_resized.reshape(-1, 512*14*14),pos_mask1)+multi_positive_infonce(x5_resized.reshape(-1, 512*14*14),pos_mask2)+self.mse_loss(img_f5, img3).mean()+self.mse_loss(img_f5, img1).mean()+self.mse_loss(img_f5, img2).mean()
        
        
        x_up4 = self.up4(img_f5.transpose(-1,-2).view(-1, 512, 14, 14), x4)
        features["upsample4"] = x_up4

        x_up44=self.cov_dim1(x_up4)
        x_up44=self.act(x_up44)
        features["upsample44"]=x_up44
        
        x_up444=x_up44.view(x_up44.size(0),-1)
        x_num=self.block_forward(x_up444)


        x_upup4=self.fusion0(x_up4,x_up44)
        x_up3 = self.up3(x_upup4, x3)
        features["upsample3"] = x_up3

        x_up33=self.cov_dim2(x_up3)
        x_up33=self.act(x_up33)
        features["upsample33"] = x_up33
        x_upup3 = self.fusion1(x_up3, x_up33)

        x_up2 = self.up2(x_upup3, x2)
        features["upsample2"] = x_up2

        x_up22 = self.cov_dim3(x_up2)
        x_up22 = self.act(x_up22)
        features["upsample22"] = x_up22

        x_upup2=self.fusion2(x_up2,x_up22)

        x_up1 = self.up1(x_upup2, x1)
        features["upsample1"] = x_up1

        logits = self.outc(x_up1)
        logits = self.act(logits)
        features["final"] = logits

        return loss_rec,logits, x_num,x_up33,x_up22, features  