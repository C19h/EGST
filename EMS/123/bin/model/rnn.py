import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader

class THISRNN(nn.Module):
    def __init__(self):
        super(THISRNN, self).__init__()

        # self.rnn=nn.RNN(
        #     input_size=2,
        #     hidden_size=16,
        #     batch_first=True,
        #     num_layers=2,
        # )
        self.rnn1 = nn.LSTM(2,10,2, dropout=0.1)
        self.rnn2 = nn.LSTM(10,20,1, dropout=0.1)
        self.actf = nn.ReLU()
        self.out=nn.Linear(20,1)

    def forward(self,x):
        r_out,h_state=self.rnn1(x,None)
        r_out = self.actf(r_out)
        r_out,h_state=self.rnn2(r_out,None)
        r_out = self.actf(r_out)
        outs=[]
        for time_step in range(r_out.size(1)):
            out = self.out(r_out[:,time_step,:])
            outs.append(out)

        return torch.stack(outs,dim=1),h_state

class Model():
    def __init__(self):
        self.model = THISRNN()

    def train(self, dataset, dstest, norm_y, epoches = 100):
        plotp = 0
        ploty = 0
        dl = DataLoader(dataset, batch_size = 40, shuffle=True)
        dltest = DataLoader(dstest, batch_size = len(dstest), shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criteon = nn.MSELoss()
        size = len(dl)
        for epoch in range(epoches):
            traindat = None
            trainy = None
            for i, batch in enumerate(dl): 
                # [seq, b] => [b, 1] => [b]
                x, y = batch
                prediction, h_state = self.model(x)
                #h_state = h_state.data  # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错
                if traindat is None:
                    traindat = prediction.detach().numpy()[:,-1,0]
                    trainy = y.detach().numpy()[:,-1,0]
                else:
                    traindat = np.hstack([traindat,prediction.detach().numpy()[:,-1,0]] )
                    trainy = np.hstack([trainy,y.detach().numpy()[:,-1,0]] )
                loss = criteon(prediction, y)     # cross entropy loss
                optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                     # backpropagation, compute gradients
                optimizer.step()
            trainy = norm_y.decode(trainy[:, np.newaxis])[:, 0]
            traindat = norm_y.decode(traindat[:, np.newaxis])[:, 0]
            for i, batch in enumerate(dltest):
                x, y = batch
                prediction, h_state = self.model(x)
                nppred = prediction.detach().numpy()[:,-1,:]
                npy = y.detach().numpy()[:,-1,:]
                nppred = norm_y.decode(nppred)[:, 0]
                npy = norm_y.decode(npy)[:, 0]
                loss = np.average(np.abs(npy-nppred) / npy)
                trainloss = np.average(np.abs(trainy-traindat) / trainy)
                print('epoch', epoch, 'trainloss', trainloss, 'loss', loss)
                #plotdata1.append(y.data.numpy()[0,-1,0])
                # i = size * epoch + i
                # dy = y.data.numpy()[0,-1,0]
                # dp = prediction.data.numpy()[0,-1,0]
                # plt.plot([i, i+1], [ploty, dy], 'r-')
                # plt.plot([i, i+1], [plotp, dp], 'b-')
                # ploty = dy
                # plotp = dp
                # plt.draw(); plt.pause(0.01)
            
    
    def predict(self, x):
        self.model.eval()
        x = torch.tensor(x)
        y,h = self.model(x)
        return y.detach().numpy()


class DataSet(torch.utils.data.Dataset):
    def __init__(self, x, y, step = 6):
        self.X = x.astype(np.float32)
        self.Y = y.astype(np.float32)
        self.Step = step
        self.Size = x.shape[0] - step + 1

    def __len__(self):
        return self.Size

    def __getitem__(self, index):
        if index < 0:
            index += self.Size
        x = self.X[index:index + self.Step]
        y = self.Y[index:index + self.Step]
        return x, y

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

