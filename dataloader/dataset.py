import random
import numpy as np
from torch.utils.data.dataset import Dataset
from config import cfg
import torch.distributed as dist
class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True):
        print('='*20, 'MultipleDatasets', '='*20)
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len
        if dist.is_initialized():
            np.random.seed(dist.get_rank())
        else:
            np.random.seed(0)
    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num 
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = np.random.randint(0,len(self.dbs[db_idx]))
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]


class RandomSampleDataset(Dataset):
    def __init__(self, db, sample_rate_dict=None):
        self.db = db
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        np.random.seed(self.rank)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 0
        self.keys = sorted(list(self.db.keys()))
        if sample_rate_dict is None:
            self.sample_rate_dict = {key: 1.0/len(self.db) for key in self.db.keys()}
        else:
            self.sample_rate_dict = sample_rate_dict
        st = 0.
        self.sample_range = []
        for key in self.keys:
            self.sample_range.append(st + self.sample_rate_dict[key])
            st = self.sample_range[-1]
        assert st == 1.0
        if self.rank == 0:
            print(self.sample_rate_dict)
    def __len__(self):
        length = sum([int(len(self.db[key]) * self.sample_rate_dict[key]) for key in self.keys])
        return length

    def __getitem__(self, index):
        p = np.random.rand()
        for i, key in enumerate(self.keys):
            if p < self.sample_range[i]:
                idx = np.random.randint(0, len(self.db[key]))
                # print('rank: {} key: {} idx: {} p: {}'.format(self.rank, key, idx, p))
                # import pdb;pdb.set_trace()
                # offset = (len(self.db[key]) // self.world_size) * self.rank
                # idx = (idx + offset) % len(self.db[key])
                return self.db[key][idx]