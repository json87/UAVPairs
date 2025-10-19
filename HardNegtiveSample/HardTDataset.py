import os.path
import random
import threading
from pathlib import Path
import torchvision.transforms as transforms
import h5py
import torch

from PIL import Image, ImageFile
from scipy.io import loadmat
from torch.utils import data
from pathlib import Path
from PIL import Image
from io import BytesIO

from HardTModelTrainMultiN import NewWidth, NewHeight


class Dataset(data.Dataset):
    def __init__(self, MatPath, DatasetPath, DataIsTrain, IsRandom, NRandom):
        super().__init__()
        Mat = loadmat(MatPath)
        DbStruct = Mat['DbStruct'].item()
        if DataIsTrain:
            if IsRandom:
                self.AllQuery = [os.path.join(DatasetPath, f[0].strip()) for f in DbStruct[0][0]]
                self.AllSPositive = [os.path.join(DatasetPath, f[0].strip()) for f in DbStruct[1][0]]
                self.AllQDBindex = DbStruct[3][0]
                random.seed()
                RandomIndex = random.sample([int(i) for i in range(0, len(self.AllQuery))], NRandom)
                self.Query = []
                self.SPositive = []
                self.QDBindex = []
                for i in range(NRandom):
                    self.Query.append(self.AllQuery[RandomIndex[i]])
                    self.SPositive.append(self.AllSPositive[RandomIndex[i]])
                    self.QDBindex.append(self.AllQDBindex[RandomIndex[i]])

                # self.Database = [os.path.join(DatasetPath, f[0]) for f in DbStruct[2][0]]
                self.Database = [[os.path.join(DatasetPath, f_one.strip()) for f_one in f] for f in DbStruct[2][0]]
                self.Negative = [[] for i in range(len(self.Query))]
                self.GetDataType = 'None'
            else:
                self.Query = [os.path.join(DatasetPath, f[0].strip()) for f in DbStruct[0][0]]
                self.SPositive = [os.path.join(DatasetPath, f[0].strip()) for f in DbStruct[1][0]]
                self.Database = [[os.path.join(DatasetPath, f_one.strip()) for f_one in f] for f in DbStruct[2][0]]
                self.QDBindex = DbStruct[3][0]
                self.Negative = [[] for i in range(len(self.Query))]
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
        elif self.GetDataType == 'OneScene':
            return len(self.Database[self.OneDBindex])
        else:
            return len(self.Query)

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.GetDataType == 'Triplet':
            Query = Image.open(self.Query[index])
            if len(Query.split()) != 3:
                Query = Query.convert('RGB')
            Query = Query.resize((NewWidth, NewHeight))
            Query = transforms.Compose([transforms.ToTensor()])(Query)

            SPositive = Image.open(self.SPositive[index])
            if len(SPositive.split()) != 3:
                SPositive = SPositive.convert('RGB')
            SPositive = SPositive.resize((NewWidth, NewHeight))
            SPositive = transforms.Compose([transforms.ToTensor()])(SPositive)

            QDBid = self.QDBindex[index]

            Negatives = []
            for negative in self.Negative[index]:
                Negative = Image.open(negative)
                if len(Negative.split()) != 3:
                    Negative = Negative.convert('RGB')
                Negative = Negative.resize((NewWidth, NewHeight))
                Negative = transforms.Compose([transforms.ToTensor()])(Negative)
                Negatives.append(Negative)
            Negatives = torch.stack(Negatives, 0)
            return Query, SPositive, Negatives, QDBid, index
        elif self.GetDataType == 'HardN':
            Query = Image.open(self.Query[index])
            if len(Query.split()) != 3:
                Query = Query.convert('RGB')
            Query = Query.resize((NewWidth, NewHeight))
            Query = transforms.Compose([transforms.ToTensor()])(Query)
            QDBid = self.QDBindex[index]
            return Query, QDBid, index
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
            TestWidth = int(TestQueryImg.size[0] / 5)
            TestHeight = int(TestQueryImg.size[1] / 5)
            TestQueryImg = TestQueryImg.resize((TestWidth, TestHeight))
            TestQueryImg = transforms.Compose([transforms.ToTensor()])(TestQueryImg)
            return TestQueryImg, index
        elif self.GetDataType == 'Cluster':
            DbImg = Image.open(self.Database[self.OneDBindex][index])
            if len(DbImg.split()) != 3:
                DbImg = DbImg.convert('RGB')
            DbImg = DbImg.resize((NewWidth, NewHeight))
            DbImg = transforms.Compose([transforms.ToTensor()])(DbImg)
            return DbImg, index
        elif self.GetDataType == 'OneScene':
            DbImg = Image.open(self.Database[self.OneDBindex][index])
            if len(DbImg.split()) != 3:
                DbImg = DbImg.convert('RGB')
            DbImg = DbImg.resize((NewWidth, NewHeight))
            DbImg = transforms.Compose([transforms.ToTensor()])(DbImg)
            return DbImg, index
def collate_fn(batch):

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None, None

    query, spositive, negatives, qdbindex, indices = zip(*batch)
    query = data.dataloader.default_collate(query)
    spositive = data.dataloader.default_collate(spositive)
    qdbindex = data.dataloader.default_collate(qdbindex)

    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)

    import itertools
    indices = list(itertools.chain(*[indices]))

    return query, spositive, negatives, negCounts, qdbindex, indices