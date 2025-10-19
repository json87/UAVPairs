from __future__ import print_function

import glob
import sys
import random
from os.path import join, isfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import h5py
import numpy as np
import os
import time
import NetVlad
from PIL import ImageFile
from PIL import Image
import torchvision.transforms as transforms
import sqlite3 as sqlite
import GeM

def getImageList(db_path):
    db_con = sqlite.connect(db_path)
    db_cur = db_con.cursor()
    db_cur.execute("select name from images order by image_id")
    res = db_cur.fetchall()
    db_cur.close()
    db_con.close()
    # im_list = [name[0] for name in res]
    im_list = []
    for name in res:
        # im_name = image_path + name[0][22:]
        im_name = os.path.join(TestDatasetPath, name[0])
        im_list.append(im_name)
    return im_list


ImageFile.LOAD_TRUNCATED_IMAGES = True


threads          = 10           # number of threads for each data loader to use
seed             = 42
nGPU             = 1            # number of GPU to use
num_clusters = 64

EncoderType   = "VGG16"           # "VGG16"or"Alexnet"or"Resnet50"
PoolingType   = "NetVlad"         # "MaxPooling"or"NetVlad"or"GeM"

DatasetDir = "F:/ljh/dataset/netvlad_train/uavpairs"
ProjectDir = "F:/ljh/Code/UAVPairs-main"
CheckpointsDir = "F:/ljh/Code/UAVPairs-main/runs"

TestDatasetPath = os.path.join(DatasetDir, "testset/images/cug/images")
hdf5Path     = os.path.join(CheckpointsDir, "centroids/VGG16_mydataset_480_64_desc_cen.hdf5")  #聚类中心
database_path = os.path.join(DatasetDir, "testset/database/campus_test.db")
output_path = os.path.join(DatasetDir, "testset/campus_" + EncoderType + "_" + PoolingType + ".npy")
imageList = getImageList(database_path)

CheckPointPath = os.path.join(CheckpointsDir, "runs/RankedList_VGG16_NetVlad/checkpoints")

class FeatureFusion(nn.Module):
    def __init__(self, NetVlad, GeM):
        super().__init__()
        self.netvlad = NetVlad
        self.gem = nn.Sequential(*[GeM, Flatten(), L2Norm()])

    def forward(self, input):
        gem_f = self.gem(input).reshape(1, -1)
        netvlad_f = self.netvlad(input)
        return torch.concat((netvlad_f, gem_f), 1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


if __name__ == '__main__':
    time1 = time.time()
    print("Encoder:", EncoderType, ", PoolingType:", PoolingType)
    device = torch.device("cuda")
    random.seed()
    np.random.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('===> Building model...')

    if EncoderType == "Alexnet":
        encoder_dim = 256
        NewWidth = 243
        NewHeight = 243
        encoder = models.alexnet(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        encoder = nn.Sequential(*layers)
        model = nn.Module()
        model.add_module('encoder', encoder)
    elif EncoderType == "VGG16":
        encoder_dim = 512
        encoder = models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        encoder = nn.Sequential(*layers)
        model = nn.Module()
        model.add_module('encoder', encoder)
    elif EncoderType == "Resnet50":
        encoder_dim = 1024
        model = models.resnet50(pretrained=True)
        print(model)
        encoder = nn.Sequential()
        for name, module in model.named_children():
            if name not in ["avgpool", "fc"]:
                if name == "layer4":
                    module_layer4 = nn.Sequential()
                    for l4ch_name, l4ch_module in module.named_children():
                        if l4ch_name == "2":
                            module_layer4_2 = nn.Sequential()
                            for l4ch_2ch_name, l4ch_2ch_module in l4ch_module.named_children():
                                if l4ch_2ch_name not in ["bn3", "relu"]:
                                    module_layer4_2.add_module(l4ch_2ch_name, l4ch_2ch_module)
                            module_layer4.add_module(l4ch_name, module_layer4_2)
                        else:
                            module_layer4.add_module(l4ch_name, l4ch_module)
                    encoder.add_module(name, module_layer4)
                else:
                    encoder.add_module(name, module)
        for p in encoder.parameters():
            p.requires_grad = True
        model = nn.Module()
        model.add_module('encoder', encoder)
    elif EncoderType == "Resnet18":
        encoder_dim = 512
        model = models.resnet18(pretrained=True)
        print(model)
        encoder = nn.Sequential()
        for name, module in model.named_children():
            if name not in ["avgpool", "fc"]:
                if name == "layer4":
                    module_layer4 = nn.Sequential()
                    for l4ch_name, l4ch_module in module.named_children():
                        if l4ch_name == "1":
                            module_layer4_2 = nn.Sequential()
                            for l4ch_2ch_name, l4ch_2ch_module in l4ch_module.named_children():
                                if l4ch_2ch_name not in ["bn2"]:
                                    module_layer4_2.add_module(l4ch_2ch_name, l4ch_2ch_module)
                            module_layer4.add_module(l4ch_name, module_layer4_2)
                        else:
                            module_layer4.add_module(l4ch_name, l4ch_module)
                    encoder.add_module(name, module_layer4)
                else:
                    encoder.add_module(name, module)
        for p in encoder.parameters():
            p.requires_grad = True
        model = nn.Module()
        model.add_module('encoder', encoder)

    if PoolingType == "MaxPooling":
        max = nn.AdaptiveMaxPool2d(1)
        model.add_module('pool', nn.Sequential(*[max, Flatten(), L2Norm()]))
    elif PoolingType == "NetVlad":
        net_vlad = NetVlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=False)
        if not os.path.exists(hdf5Path):
            raise FileNotFoundError("Could not find clusters, please run cluster.py before proceeding!")
        with h5py.File(hdf5Path, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            net_vlad.init_params(clsts, traindescs)
            del clsts, traindescs
        model.add_module('pool', net_vlad)
        gem = GeM.GeM()
        feature_fusion = FeatureFusion(net_vlad, gem)
        model.add_module('pool', feature_fusion)
    elif PoolingType == "GeM":
        gem = GeM.GeM()
        model.add_module('pool', nn.Sequential(*[gem, Flatten(), L2Norm()]))

    model = model.to(device)
    print("Network Structure:")
    print(model)

    CheckPointFile = join(CheckPointPath, 'model_best.pth.tar')

    if isfile(CheckPointFile):
        print("=> loading checkpoint '{}'".format(CheckPointFile))
        checkpoint = torch.load(CheckPointFile, map_location=lambda storage, loc: storage)
        TestEpoch = checkpoint['epoch']
        BestMetric = checkpoint['best_score']
        if EncoderType == "Resnet" or EncoderType == "Alexnet":
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.to(device)

        # for parameters in model.parameters():  # 打印出参数矩阵及值
        #     print(parameters)

        print("=> loaded checkpoint '{}' (epoch {})".format(CheckPointFile, TestEpoch))
        print('===> Start testing')
        DbImageFeat = torch.zeros(0).to(torch.device("cpu"))  # Avoid using too much GPU memory
        model.eval()
        with torch.no_grad():
            for image_name in imageList:
                print(image_name)
                DatabaseImg = Image.open(image_name)
                if len(DatabaseImg.split()) != 3:
                    DatabaseImg = DatabaseImg.convert('RGB')
                DatabaseImg = transforms.Compose([transforms.ToTensor()])(DatabaseImg)
                DatabaseImg = DatabaseImg.to(device)
                DatabaseImg = torch.unsqueeze(DatabaseImg, 0)
                DbImg_Encoding = model.encoder(DatabaseImg)

                DbImg_Encoding = model.pool(DbImg_Encoding)
                DbImageFeat = torch.cat((DbImageFeat, DbImg_Encoding.to(torch.device("cpu"))), 0)

        DbImageFeat = DbImageFeat.cpu().detach().numpy()
        np.save(output_path, DbImageFeat)
        print(DbImageFeat.shape)

    else:
        print("Can't find checkpoint file: ", CheckPointFile)

    time2 = time.time()
    print(time2-time1)