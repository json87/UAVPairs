import os.path
import random
import threading
from pathlib import Path
import torchvision.transforms as transforms
import h5py
import torch
from sklearn.neighbors import NearestNeighbors
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image, ImageFile
from scipy.io import loadmat
from torch.utils import data
from pathlib import Path
from PIL import Image
from io import BytesIO

from RankedListModelTrain import NewWidth, NewHeight


class Dataset(data.Dataset):
    def __init__(self, MatPath, DatasetPath, DataIsTrain):
        super().__init__()
        Mat = loadmat(MatPath)
        DbStruct = Mat['DbStruct'].item()
        if DataIsTrain:
            self.Query = [os.path.join(DatasetPath, f[0].strip()) for f in DbStruct[0][0]]
            self.Positives = [[os.path.join(DatasetPath, f_one[0].strip()) for f_one in f] for f in DbStruct[1]]
            self.Database = [os.path.join(DatasetPath, f[0].strip()) for f in DbStruct[2][0]]
            self.GetDataType = 'None'
        else:
            self.inliers = []
            self.Query = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[0]]
            self.TestDatabaseIndex = DbStruct[1]
            self.TestDatabase = [os.path.join(DatasetPath, f[0].item()) for f in DbStruct[2]]
            for i in range(DbStruct[3][0].shape[0]):
                one_inlier = [os.path.join(DatasetPath, f.strip()) for f in DbStruct[3][0][i]]
                self.inliers.append(one_inlier)
            self.GetDataType = 'None'
            self.OneDBindex = -1

    def __len__(self):
        if self.GetDataType == 'Database':
            return len(self.TestDatabase)
        elif self.GetDataType == 'Cluster':
            return len(self.Database)
        else:
            return len(self.Query)

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.GetDataType == 'RankedList':
            Query = Image.open(self.Query[index])
            if len(Query.split()) != 3:
                Query = Query.convert('RGB')
            Query = Query.resize((NewWidth, NewHeight))
            Query = transforms.Compose([transforms.ToTensor()])(Query)

            Positives = []
            for positive in self.Positives[index]:
                Positive = Image.open(positive)
                if len(Positive.split()) != 3:
                    Positive = Positive.convert('RGB')
                Positive = Positive.resize((NewWidth, NewHeight))
                Positive = transforms.Compose([transforms.ToTensor()])(Positive)
                Positives.append(Positive)
            Positives = torch.stack(Positives, 0)
            return Query, Positives, index
        elif self.GetDataType == 'Database':
            DbImg = Image.open(self.TestDatabase[index])
            if len(DbImg.split()) != 3:
                DbImg = DbImg.convert('RGB')
            DbImg = transforms.Compose([transforms.ToTensor()])(DbImg)
            return DbImg, index
        elif self.GetDataType == 'TestQuery':
            TestQueryImg = Image.open(self.Query[index])
            if len(TestQueryImg.split()) != 3:
                TestQueryImg = TestQueryImg.convert('RGB')
            # TestWidth = int(TestQueryImg.size[0] / 5)
            # TestHeight = int(TestQueryImg.size[1] / 5)
            # TestQueryImg = TestQueryImg.resize((TestWidth, TestHeight))
            TestQueryImg = transforms.Compose([transforms.ToTensor()])(TestQueryImg)
            return TestQueryImg, index
        elif self.GetDataType == 'Cluster':
            DbImg = Image.open(self.Database[self.OneDBindex][index])
            if len(DbImg.split()) != 3:
                DbImg = DbImg.convert('RGB')
            DbImg = DbImg.resize((NewWidth, NewHeight))
            DbImg = transforms.Compose([transforms.ToTensor()])(DbImg)
            return DbImg, index
def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positives: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positives: torch tensor of shape (batch_size, n, 3, h, w).
        posCounts: torch tensor of shape (batch_size, n).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None, None, None

    query, positives, indices = zip(*batch)
    query = data.dataloader.default_collate(query)
    posCounts = data.dataloader.default_collate([x.shape[0] for x in positives])
    positives = torch.cat(positives, 0)

    import itertools
    indices = list(itertools.chain(*[indices]))

    return query, positives, posCounts, indices
