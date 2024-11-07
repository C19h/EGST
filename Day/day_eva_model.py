
import pytorch_lightning as pl
import torch


from day_config import Config
from utils.graphtransformer_dataset import PlData
from model.net import PlGraphAttention

train_dataset = PlData(Config)
trainer = pl.Trainer()
checkpoint = torch.load(
    "../pretrained/Day/6_5_5_0.982/0627_1451_model_2/weights/epoch=87-val_loss=0.06009-val_acc=0.98150.ckpt",map_location=torch.device('cpu'))
hyper_parameters = checkpoint["hyper_parameters"]
t = PlGraphAttention(**hyper_parameters)
model_weights = checkpoint["state_dict"]
t.load_state_dict(model_weights)
result = trainer.test(t, train_dataset.val_dataloader())
