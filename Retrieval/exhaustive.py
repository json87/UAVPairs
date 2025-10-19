import time
import numpy as np
import faiss
import sqlite3 as sqlite
import os

def getImageList(db_path):
    db_con = sqlite.connect(db_path)
    db_cur = db_con.cursor()
    db_cur.execute("select name from images order by image_id")
    res = db_cur.fetchall()
    db_cur.close()
    db_con.close()
    im_list = [name[0] for name in res]
    return im_list


DatasetDir = "F:/ljh/dataset/netvlad_train/uavpairs"
vlad_file = os.path.join(DatasetDir, "testset/campus_VGG16_NetVlad.npy")
save_file = os.path.join(DatasetDir, "testset/campus_VGG16_NetVlad_pairs.txt")
database_path = os.path.join(DatasetDir, "testset/database/campus_test.db")
image_list = getImageList(database_path)

vlad_vector = np.load(vlad_file)
print(vlad_vector.dtype)
num = vlad_vector.shape[0]
dim = vlad_vector.shape[1]
k = 31  # we want to see k nearest neighbors
head_split = 0  # campus 29 SZU 26 2w 21

t_start = time.time()
vlad_index = faiss.IndexFlatIP(dim)   # build the index
print(vlad_index.is_trained)
vlad_index.add(vlad_vector)  # add vectors to the index
print(vlad_index.ntotal)

D, I = vlad_index.search(vlad_vector, k)   # num x K; D: vector; I: index
t_end = time.time()
print("time: " + str(t_end - t_start))
output_file = open(save_file, 'w')
for index, values in enumerate(I):
    im1 = image_list[index][head_split:].replace('\\', '/')
    for value in values[1:]:
        im2 = image_list[value][head_split:].replace('\\', '/')
        output_file.write(im1 + ' ' + im2 + '\n')


