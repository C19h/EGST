# author:c19h
# datetime:2022/12/7 15:02
# -- coding: utf-8 --**
# GCN model
import os
import sys

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from pytorch_lightning import LightningModule
from torch_geometric.nn import voxel_grid, max_pool, max_pool_x, GMMConv,GATv2Conv,global_mean_pool
from torchmetrics import Accuracy

from origin_config import Config

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = GMMConv(in_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        self.left_conv2 = GMMConv(out_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)

        self.shortcut_conv = GMMConv(in_channel, out_channel, dim=3, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)

    def forward(self, data):
        data.x = F.elu(self.left_bn2(
            self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                            data.edge_index, data.edge_attr)) + self.shortcut_bn(
            self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))

        return data


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GMMConv(1, 64, dim=3, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.block1 = ResidualBlock(64, 128)
        self.block2 = ResidualBlock(128, 256)
        self.block3 = ResidualBlock(256, 512)

        self.fc1 = torch.nn.Linear(8 * 512, 1024)
        self.bn = torch.nn.BatchNorm1d(1024)
        self.drop_out = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(1024, 20)

    def forward(self, data):
        data.x = F.elu(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        cluster = voxel_grid(data.pos, batch=data.batch, size=4)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block1(data)
        cluster = voxel_grid(data.pos, batch=data.batch, size=6)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block2(data)
        cluster = voxel_grid(data.pos, batch=data.batch, size=24)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block3(data)
        cluster = voxel_grid(data.pos, batch=data.batch, size=64)
        x = max_pool_x(cluster, data.x, batch=data.batch, size=8)

        # if your torch-geometric version is below 1.3.2(roughly, we do not test all versions), use x.view() instead of x[0].view()
        # x = x.view(-1, self.fc1.weight.size(1))
        x = x[0].view(-1, self.fc1.weight.size(1))
        x = self.fc1(x)
        x = F.elu(x)
        x = self.bn(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class PlGraphAttention(LightningModule):
    def __init__(self, learning_rate):
        super(PlGraphAttention, self).__init__()
        self.save_hyperparameters()
        self.model = Net()
        self.accuracy = Accuracy(num_classes=20)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        y = batch.y
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def training_step_end(self, outputs):
        train_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("train_acc", train_acc, prog_bar=True, batch_size=Config.batch_size)
        return {"loss": outputs["loss"].mean()}

    def validation_step(self, batch, batch_idex):
        preds = self(batch)
        y = batch.y
        loss = F.nll_loss(preds, y)
        return {"loss": loss, "preds": preds, "y": y}

    def validation_step_end(self, outputs):
        val_acc = self.accuracy(outputs['preds'], outputs['y'])
        self.log("val_loss", outputs["loss"].mean(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=Config.batch_size)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False, batch_size=Config.batch_size)
