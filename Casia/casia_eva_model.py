# author:c19h
# datetime:2022/12/27 19:41
import pytorch_lightning as pl
import torch

from casia_config import Config
from utils.graphtransformer_dataset import PlData_casia
from model.net import PlGraphAttention_casia_2, PlGraphAttention_casia

model = PlGraphAttention_casia(Config, num_attention=1, learning_rate=0.001, f_graph_dim=32, heads=2)
clone = model.load_from_checkpoint(
    '/root/autodl-tmp/GraphTransformer/pretrained/CASIA_B_000_1/2_5_5/1227_1853_model_1/weights/epoch=85-val_loss=0.26784-val_acc=0.99330.ckpt')
step2_train_dataset = PlData_casia(Config, step='2')
trainer = pl.Trainer(gpus=1)
checkpoint = torch.load(
    "/root/autodl-tmp/GraphTransformer/pretrained/CASIA_B_000_2/2_5_5/1227_1922_model_0/weights/epoch=95-val_loss=-48.96880-val_acc=0.78846.ckpt")
hyper_parameters = checkpoint["hyper_parameters"]
t = PlGraphAttention_casia_2(pretrained_model=clone, **hyper_parameters)
model_weights = checkpoint["state_dict"]
t.load_state_dict(model_weights)
result = trainer.test(t, step2_train_dataset.val_dataloader())
