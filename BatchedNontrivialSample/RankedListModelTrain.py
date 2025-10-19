from __future__ import print_function
import argparse
import math
import os
import sys
from math import log10, ceil
import random, shutil, json
from multiprocessing import pool
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss
from tensorboardX import SummaryWriter
import numpy as np
from joblib import Parallel, delayed
import os
import time
import multiprocessing
from numpy import mean
from collections import defaultdict

import RankedListdataset

import NetVlad

import GeM

threads          = 12            # number of threads for each data loader to use
seed             = 42
batchSize        = 5           # number of triples
cacheBatchSize   = 20          # batch size for caching and testing
cacheRefreshRate = 0         # how often to refresh cache, in number of queries. 0 for off
margin = 0.1          # margin for triplet loss. Default=0.1
alpha = 1.35
nGPU             = 1            # number of GPU to use
LearnRateStep    = 5            # decay LR ever N steps
LearnRateGamma   = 0.5          # multiply LR by Gamma for decaying
momentum         = 0.9          # momentum for SGD
nEpochs          = 20           # number of epochs to train for
StartEpoch       = 0            # manual epoch number (useful on restarts)
evalEvery        = 1            # do a validation set run, and save, every N epochs
patience         = 0            # patience for early stopping. 0 is off
NewWidth         = 480
NewHeight        = 320
optimtype = "adam"
LearnRate        = 1e-5       # learning Rate
weightDecay      = 5e-4        # weight decays for SGD

EncoderType   = "VGG16"         # "VGG16"or"Alexnet"or"Resnet"
PoolingType   = "NetVlad"       # "MaxPooling"or"NetVlad"
num_clusters  = 64

losstype = "RankedList"

DatasetDir = "F:/ljh/dataset/netvlad_train/uavpairs"
ProjectDir = "F:/ljh/Code/UAVPairs-main"
CheckpointsDir = "F:/ljh/Code/UAVPairs-main"

DatasetPath  = os.path.join(DatasetDir, "trainset/images")
TestDatasetPath = os.path.join(DatasetDir, "testset/images/cug/images")
TrainMatPath = os.path.join(DatasetDir, "trainset/BatchedNontrivialSample_train2.mat")
TestMatPath  = os.path.join(DatasetDir, "uavpairs/trainset/test.mat")
hdf5Path     = os.path.join(CheckpointsDir, "centroids/VGG16_mydataset_480_64_desc_cen.hdf5")  #聚类中心
runsPath     = os.path.join(CheckpointsDir, "runs/")
gt_test_file = os.path.join(DatasetDir, "trainset/true_pair_100.txt")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def TrainOneEpochTriplet(epoch):
    TrainData.GetDataType = 'RankedList'
    epoch_loss = 0
    startIter = 1
    if cacheRefreshRate > 0:
        subsetN = ceil(len(TrainData) / cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(TrainData)), subsetN)  # Divide TrainData into subsetN groups
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(TrainData))]
    nBatches = (len(TrainData) + batchSize - 1) // batchSize
    print("number of batches: ", nBatches)
    print("Divide TrainData into", subsetN, "groups")

    for subIter in range(subsetN):  # 遍历每一组TrainData
        print("Currently the number ", str(subIter + 1), "group of TrainData")

        # 不启用 BatchNormalization 和 Dropout
        model.eval()

        SubData = Subset(dataset=TrainData, indices=subsetIdx[subIter])
        SubQueryDataLoader = DataLoader(dataset=SubData, num_workers=threads, batch_size=batchSize, shuffle=False, collate_fn=RankedListdataset.collate_fn, pin_memory=True)
        # # 启用 BatchNormalization 和 Dropout
        model.train()
        # index的长度为batchsize
        for iteration, (query, positives, posCounts, index) in enumerate(SubQueryDataLoader, startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            # qdbindex = qdbindex.numpy()
            if query is None: continue  # in case we get an empty batch
            B, C, H, W = query.shape

            nPos = torch.sum(posCounts)  # B x num_pos

            input = torch.cat([query, positives])

            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)

            vladQ, vladP = torch.split(vlad_encoding, [B, nPos])

            optimizer.zero_grad()

            loss = torch.tensor(0.0).to(device)
            count = torch.tensor(0.0).to(device)
            for i, posCount in enumerate(posCounts):
                for j in range(posCount):
                    posIx = (torch.sum(posCounts[:i]) + j).item()
                    for k in range(B):
                        if k != i:
                            for nj in range(posCounts[k]):
                                negIx = (torch.sum(posCounts[:k]) + nj).item()
                                loss_item = criterion(vladQ[i], vladP[posIx], vladP[negIx])
                                if loss_item > 0:
                                    loss += loss_item
                                    count += 1
            loss /= (nPos*(nPos-int(nPos/B))).float().to(device)
            loss /= (count + 1e-6).float().to(device)
            if loss <= 0:
                continue
            loss.backward()
            optimizer.step()

            del input, image_encoding, vlad_encoding, vladQ, vladP
            del query, positives

            batch_loss = loss.item()

            epoch_loss += batch_loss

            if iteration % 100 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', 1, ((epoch - 1) * nBatches) + iteration)

        startIter += len(SubQueryDataLoader)
        del SubQueryDataLoader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        TrainData.GetDataType = 'None'

    avg_loss = epoch_loss / nBatches
    print("===> Epoch {} Complete!  Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    return avg_loss


def TrainOneEpoch(epoch):
    TrainData.GetDataType = 'RankedList'
    epoch_loss = 0
    epoch_loss_p = 0
    epoch_loss_n = 0
    startIter = 1
    if cacheRefreshRate > 0:
        subsetN = ceil(len(TrainData) / cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(TrainData)), subsetN)  # Divide TrainData into subsetN groups
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(TrainData))]
    nBatches = (len(TrainData) + batchSize - 1) // batchSize
    print("number of batches: ", nBatches)
    print("Divide TrainData into", subsetN, "groups")

    for subIter in range(subsetN):  # 遍历每一组TrainData
        print("Currently the number ", str(subIter + 1), "group of TrainData")
        model.eval()

        SubData = Subset(dataset=TrainData, indices=subsetIdx[subIter])
        SubQueryDataLoader = DataLoader(dataset=SubData, num_workers=threads, batch_size=batchSize, shuffle=False, collate_fn=RankedListdataset.collate_fn, pin_memory=True)
        # # 启用 BatchNormalization 和 Dropout
        model.train()
        # index的长度为batchsize
        for iteration, (query, positives, posCounts, index) in enumerate(SubQueryDataLoader, startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            # qdbindex = qdbindex.numpy()
            if query is None: continue  # in case we get an empty batch
            B, C, H, W = query.shape

            nPos = torch.sum(posCounts)  # B x num_pos

            input = torch.cat([query, positives])

            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)

            vladQ, vladP = torch.split(vlad_encoding, [B, nPos])

            optimizer.zero_grad()

            loss_p = torch.tensor(0.0).to(device)
            loss_n = torch.tensor(0.0).to(device)
            count_n = torch.tensor(0.0).to(device)
            count_p = torch.tensor(0.0).to(device)
            for i, posCount in enumerate(posCounts):
                for j in range(posCount):
                    posIx = (torch.sum(posCounts[:i]) + j).item()
                    dist_ap = pdist(vladQ[i], vladP[posIx])
                    dist_aa = torch.zeros_like(dist_ap)
                    y = torch.ones_like(dist_ap)
                    loss_p_item = criterionP(dist_aa, dist_ap, y)
                    if loss_p_item > 0:
                        loss_p += loss_p_item
                        count_p += 1

                    for k in range(B):
                        if k != i:
                            dist_an = pdist(vladQ[k], vladP[posIx])
                            dist_aa = torch.zeros_like(dist_an)
                            y = torch.ones_like(dist_an)
                            loss_n_item = criterionN(dist_an, dist_aa, y)
                            if loss_n_item > 0:
                                loss_n += loss_n_item
                                count_n += 1

            loss_p /= (count_p + 1e-6).float().to(device)
            loss_n /= (count_n + 1e-6).float().to(device)
            loss = loss_p + loss_n

            if loss <= 0:
                continue
            loss.backward()
            optimizer.step()

            del input, image_encoding, vlad_encoding, vladQ, vladP
            del query, positives

            batch_loss = loss.item()
            batch_loss_n = loss_n.item()
            batch_loss_p = loss_p.item()

            epoch_loss += batch_loss
            epoch_loss_n += batch_loss_n
            epoch_loss_p += batch_loss_p

            if iteration % 100 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}, Loss_p: {:.4f}, Loss_n: {:.4f}".format(epoch, iteration, nBatches, batch_loss, batch_loss_p, batch_loss_n), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', 1, ((epoch - 1) * nBatches) + iteration)

        startIter += len(SubQueryDataLoader)
        del SubQueryDataLoader, loss_p, loss_n, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        TrainData.GetDataType = 'None'

    avg_loss = epoch_loss / nBatches
    avg_loss_p = epoch_loss_p / nBatches
    avg_loss_n = epoch_loss_n / nBatches
    print("===> Epoch {} Complete!  Avg. Loss: {:.4f}, Loss_p: {:.4f}, Loss_n: {:.4f}".format(epoch, avg_loss, avg_loss_p, avg_loss_n), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    return avg_loss


def Validate_campus(TestData, epoch=0, write_tboard=False):
    time1 = time.time()
    model.eval()

    QueryFeat = torch.zeros(0).to(device)

    TestData.GetDataType = 'TestQuery'
    Databaseloader = DataLoader(dataset=TestData, num_workers=8, batch_size=5, shuffle=False, pin_memory=True)
    with torch.no_grad():
        for iteration, (QueryImg, Index) in enumerate(Databaseloader, 1):
            # sys.stdout.flush()
            # torch.cuda.empty_cache()
            QueryImg = QueryImg.to(device)
            QueryImg_Encoding = model.encoder(QueryImg)
            QueryFeat = torch.cat((QueryFeat, model.pool(QueryImg_Encoding)), 0)
            # del QueryImg, QueryImg_Encoding

    QueryFeat = QueryFeat.cpu().detach().numpy()
    print(QueryFeat.shape)
    test_dict = defaultdict(list)
    gt_dict = defaultdict(list)

    netvlad_index = faiss.IndexFlatL2(faiss_dim)  # build the index
    netvlad_index.add(QueryFeat)

    num_test = 0
    D, I = netvlad_index.search(QueryFeat, 31)
    del QueryFeat
    for index, values in enumerate(I):
        im1 = TestData.TestDatabase[index].replace(TestDatasetPath + "\\", '')
        for value in values[1:]:
            im2 = TestData.TestDatabase[value].replace(TestDatasetPath + "\\", '')
            test_dict[im1].append(im2)
            num_test = num_test + 1

    with open(gt_test_file, 'r') as lines:
        for line in lines:
            line = line.strip('\n')
            strs = line.split(" ")
            gt_dict[strs[0]].append(strs[1])
            gt_dict[strs[1]].append(strs[0])

    global_num = 0
    for key in test_dict:
        set_c = set(test_dict[key]) & set(gt_dict[key])
        list_c = list(set_c)
        local_num = len(list_c)
        global_num = global_num + local_num
    mAP = global_num / (num_test * 1.0)

    print("-------------------------------")
    print("===> global_num: {}".format(global_num))
    print("====> mAP: {:.5f}".format(mAP))
    time2 = time.time()
    print(time2-time1)
    print("-------------------------------")
    torch.cuda.empty_cache()
    return mAP


def save_checkpoint(state, is_best, filename):
    model_out_path = join(savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(savePath, 'model_best.pth.tar'))


if __name__ == '__main__':
    print("Encoder:", EncoderType, ", PoolingType:", PoolingType)  # 模型类型和池化类型
    device = torch.device("cuda")  # device = cuda

    random.seed()
    np.random.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('===> Loading training dataset(s)...')
    TrainData = RankedListdataset.Dataset(TrainMatPath, DatasetPath, True)

    TrainDataLoader = DataLoader(dataset=TrainData, num_workers=threads, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)

    TestData = RankedListdataset.Dataset(TestMatPath, TestDatasetPath, False)
    TestDataLoader = DataLoader(dataset=TestData, num_workers=threads, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
    print('Number of original triples:', len(TrainData.Query))
    print('Number of triples after filtering:', len(TrainData))
    print('Number of database images:', len(TrainData.Database))

    print('===> Building model...')

    num_clusters = 64

    if EncoderType == "Alexnet":
        encoder_dim = 256
        encoder = models.alexnet(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        for l in layers:
            for p in l.parameters():
                p.requires_grad = True
        encoder = nn.Sequential(*layers)
        model = nn.Module()
    elif EncoderType == "VGG16":
        encoder_dim = 512
        encoder = models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        for l in layers:
            for p in l.parameters():
                p.requires_grad = True
        encoder = nn.Sequential(*layers)
    elif EncoderType == "Resnet50":
        encoder_dim = 2048
        model = models.resnet50(pretrained=True)
        encoder = nn.Sequential()
        for name, module in model.named_children():
            if name not in ["avgpool", "fc"]:  # "layer4"
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
                encoder.add_module(name, module)
        for p in encoder.parameters():
            p.requires_grad = True

    model = nn.Module()
    model.add_module('encoder', encoder)

    if PoolingType == "MaxPooling":
        max = nn.AdaptiveMaxPool2d(1)
        model.add_module('pool', nn.Sequential(*[max, Flatten(), L2Norm()]))
        faiss_dim = encoder_dim
    elif PoolingType == "NetVlad":
        net_vlad = NetVlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, normalize_input=True, vladv2=False)
        if not os.path.exists(hdf5Path):
            raise FileNotFoundError("Could not find clusters, please run cluster.py before proceeding")
        with h5py.File(hdf5Path, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]

            net_vlad.init_params(clsts, traindescs)

            del clsts, traindescs
        model.add_module('pool', net_vlad)
        faiss_dim = encoder_dim * num_clusters
    elif PoolingType == "GeM":
        gem = GeM.GeM(p=3)
        model.add_module('pool', nn.Sequential(*[gem, Flatten(), L2Norm()]))
        faiss_dim = encoder_dim

    isParallel = False
    model = model.to(device)
    print("Network Structure:")
    print(model)

    parameters = []
    if PoolingType == "GeM":
        parameters.append({'params': model.encoder.parameters()})
        parameters.append({'params': model.pool.parameters(), 'lr': LearnRate * 10, 'weight_decay': 0})
    elif PoolingType == "NetVlad":
        parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        parameters = filter(lambda p: p.requires_grad, model.parameters())

    # define optimizer
    if optimtype == 'sgd':
        optimizer = optim.SGD(parameters, LearnRate, momentum=momentum, weight_decay=weightDecay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LearnRateStep, gamma=LearnRateGamma)
    elif optimtype == 'adam':
        optimizer = optim.Adam(parameters, LearnRate, weight_decay=weightDecay)
        exp_decay = math.exp(-0.1)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    if losstype == "Triplet":
       criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction='sum').to(device)
    else:
        criterionP = nn.MarginRankingLoss(margin=margin-alpha, reduction='none').to(device)
        criterionN = nn.MarginRankingLoss(margin=alpha, reduction='none').to(device)
        # criterionRank = nn.MarginRankingLoss(margin=0, reduction='none').to(device)
        pdist = nn.PairwiseDistance(p=2)

    print('===> Training model...')



    OutputFile = EncoderType + "_" + PoolingType + ".txt"
    with open(OutputFile, 'a') as Output:
        Output.write(EncoderType + "_" + PoolingType + "\n")
    writer = SummaryWriter(log_dir=join(runsPath, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + EncoderType + '_' + PoolingType))
    logdir = writer.file_writer.get_logdir()
    savePath = join(logdir, "checkpoints")
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    makedirs(savePath)

    BestmAP = 0

    mAP = Validate_campus(TestData, 0, write_tboard=True)
    with open(OutputFile, 'a') as Output:
        Output.write(str(0) + "\t" + str(alpha-margin) + "\t" + str(mAP) + "\n")
    CheckPointFile = "CheckPoint_" + EncoderType + "_" + PoolingType + "_" + str(0) + ".pth.tar"
    save_checkpoint({'epoch': 0, 'state_dict': model.state_dict(), 'mAP': mAP, 'best_score': mAP,
                     'optimizer': optimizer.state_dict(), 'parallel': isParallel, }, False, CheckPointFile)

    for epoch in range(StartEpoch + 1, nEpochs + 1):
        if losstype == "Triplet":
            AveLoss = TrainOneEpochTriplet(epoch)
        else:
            AveLoss = TrainOneEpoch(epoch)
        scheduler.step(epoch)
        mAP = Validate_campus(TestData, epoch, write_tboard=True)
        if mAP > BestmAP:
            BestmAP = mAP
            IsBestFlag = True
        else:
            IsBestFlag = False

        with open(OutputFile, 'a') as Output:
            Output.write(str(epoch) + "\t" + str(AveLoss) + "\t" + str(mAP) + "\n")
        CheckPointFile = "CheckPoint_" + EncoderType + "_" + PoolingType + "_" + str(epoch) + ".pth.tar"
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'mAP': mAP, 'best_score': BestmAP, 'optimizer': optimizer.state_dict(), 'parallel': isParallel, }, IsBestFlag, CheckPointFile)

