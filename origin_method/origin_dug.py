# author:c19h
# datetime:2022/12/7 15:11
# author:c19h
# datetime:2022/12/2 14:48
from thop import profile
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from utils import training_utils
from origin_dataset import PlData
from origin_net import PlGraphAttention
from ptflops import get_model_complexity_info
# %%
path_model = training_utils.create_model_folder('pretrained_origin', '/test')
train_dataset = PlData()
datas = next(iter(train_dataset.train_dataloader()))
model = PlGraphAttention(learning_rate=0.001)
datas = next(iter(train_dataset.train_dataloader()))

# pl.seed_everything(1)
# callback = [
#     EarlyStopping(**{'monitor': 'val_loss', 'min_delta': 0.001, 'patience': 5, 'check_finite': True, 'mode': 'min'}),
#     ModelCheckpoint(
#         **{'monitor': 'val_loss', 'mode': 'min', 'filename': '{epoch}-{loss_total:.5f}-{val_acc:.5f}',
#            'dirpath': path_model,
#            'save_last': False,
#            'save_weights_only': False, 'save_top_k': 1, 'every_n_epochs': 1}),
#     ModelCheckpoint(**{'monitor': 'val_acc', 'mode': 'max',
#                        'dirpath': path_model,
#                        'filename': '{epoch}-{val_loss:.5}-{val_acc:.5f}',
#                        'save_last': False, 'save_weights_only': False, 'save_top_k': 1, 'every_n_epochs': 1}),
#     LearningRateMonitor(**{'logging_interval': 'epoch'})]
# logger = TensorBoardLogger('log_dir_origin', name='gt')
# trainer = pl.Trainer(gpus=1, max_epochs=200, callbacks=callback, logger=logger, check_val_every_n_epoch=5,
#                      accumulate_grad_batches=1)
# trainer.fit(model, train_dataset)
