import numpy as np
import os
from collections import defaultdict

DatasetDir = "F:/ljh/dataset/netvlad_train/uavpairs"
vlad_file = os.path.join(DatasetDir, "testset/campus_VGG16_NetVlad.npy")
save_file = os.path.join(DatasetDir, "testset/campus_VGG16_NetVlad_pairs.txt")
database_path = os.path.join(DatasetDir, "testset/database/campus_test.db")

test_data_file = os.path.join(DatasetDir, "testset/evaluation/exhaustive_pair/campus/netvlad_triplet_exhaustive.txt")
inlier_data_file = os.path.join(DatasetDir, "testset/evaluation/exhaustive_inlier/campus/netvlad_triplet_exhaustive.txt")
# head = 22
test_dict = defaultdict(list)
inlier_dict = defaultdict(list)

num_test = 0
with open(test_data_file, 'r') as lines:
    for line in lines:
        num_test = num_test + 1
        line = line.strip('\n')
        strs = line.split(" ")
        # strs[0] = strs[0][head:]
        # strs[1] = strs[1][head:]
        test_dict[strs[0]].append(strs[1])

with open(inlier_data_file, 'r') as lines:
    for line in lines:
        line = line.strip('\n')
        strs = line.split(" ")
        inlier_dict[strs[0]].append(strs[1])
        inlier_dict[strs[1]].append(strs[0])

global_num = 0
for key in test_dict:
    set_c = set(test_dict[key]) & set(inlier_dict[key])
    list_c = list(set_c)
    local_num = len(list_c)
    global_num = global_num + local_num
print(global_num)
print(num_test)
print(global_num / (num_test * 1.0))

