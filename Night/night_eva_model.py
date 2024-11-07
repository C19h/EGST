# author:c19h
# datetime:2022/12/24 14:37
# author:c19h
# datetime:2022/12/14 18:48

import pytorch_lightning as pl
import torch

from night_config import Config
from utils.graphtransformer_dataset import PlData
from model.net import PlGraphAttention

train_dataset = PlData(Config)
trainer = pl.Trainer(gpus=1)
checkpoint = torch.load(
    "/root/autodl-tmp/GraphTransformer/pretrained/Night/10_10_5_0.9635/1224_1011_model_1/weights/epoch=75-val_loss=0.10193-val_acc=0.96350.ckpt")
hyper_parameters = checkpoint["hyper_parameters"]
t = PlGraphAttention(**hyper_parameters)
model_weights = checkpoint["state_dict"]
t.load_state_dict(model_weights)
result = trainer.test(t, train_dataset.val_dataloader())
