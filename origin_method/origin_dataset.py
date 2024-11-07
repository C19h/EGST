# author:c19h
# datetime:2022/12/7 15:03

from tqdm import tqdm
import os
import numpy as np
import glob
import scipy.io as sio
import torch
import torch.utils.data
from torch_geometric.data import Data, Dataset
import os.path as osp
import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from origin_config import Config
def files_exist(files):
    return all([osp.exists(f) for f in files])

class EV_Gait_3DGraph_Dataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(EV_Gait_3DGraph_Dataset, self).__init__(root, transform, pre_transform)

    # return file list of self.raw_dir
    @property
    def raw_file_names(self):
        all_filenames = glob.glob(os.path.join(self.raw_dir, "*.mat"))
        # get all file names
        file_names = [f.split(os.sep)[-1] for f in all_filenames]
        return file_names

    # get all file names in  self.processed_dir
    @property
    def processed_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir, "*.mat"))
        file = [f.split(os.sep)[-1] for f in filenames]
        saved_file = [f.replace(".mat", ".pt") for f in file]
        return saved_file

    def __len__(self):
        return len(self.processed_file_names)

    def len(self):
        return len(self.processed_file_names)

    def download(self):
        pass


    # convert the mat files of self.raw_dir to torch_geometric.Data format, save the result files in self.processed_dir
    # this method will only execute one time at the first running.
    def process(self):
        for raw_path in self.raw_paths:

            content = sio.loadmat(raw_path)
            feature = torch.tensor(content["feature"])[:, 0:1].float()
            edge_index = torch.tensor(
                np.array(content["edges"]).astype(np.int32), dtype=torch.long
            )
            pos = torch.tensor(np.array(content["pseudo"]), dtype=torch.float32)
            label_idx = torch.tensor(int(content["label"]), dtype=torch.long)
            data = Data(
                x=feature, edge_index=edge_index, pos=pos, y=label_idx.unsqueeze(0)
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            saved_name = raw_path.split(os.sep)[-1].replace(".mat", ".pt")
            torch.save(data, osp.join(self.processed_dir, saved_name))

    def get(self, idx):
        data = torch.load(osp.join(self.processed_paths[idx]))
        return data

class PlData(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.96, 1]), T.RandomJitter(0.01)])
        self.test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.999, 1])])

    def train_dataloader(self):
        dt = EV_Gait_3DGraph_Dataset(Config.graph_train_dir, transform=self.train_data_aug)
        dl = DataLoader(dt, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=True)
        return dl

    def val_dataloader(self):
        dt = EV_Gait_3DGraph_Dataset(Config.graph_test_dir, transform=self.test_data_aug)
        dl = DataLoader(dt, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True)
        return dl