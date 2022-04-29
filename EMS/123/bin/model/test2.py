import torch, numpy as np
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
steps=np.linspace(0,np.pi*4,100,dtype=np.float32)
x_np=np.sin(steps)
y_np=np.cos(steps)

class THISRNN(nn.Module):
    def __init__(self):
        super(THISRNN, self).__init__()

        self.rnn=nn.RNN(
            input_size=1,
            hidden_size=32,
            batch_first=True,
            num_layers=1,
        )
        self.out=nn.Linear(32,1)

    def forward(self,x,h_state):
        r_out,h_state=self.rnn(x,h_state)
        outs=[]
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:,time_step,:]))

        return torch.stack(outs,dim=1),h_state

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

dataset = DataSet(x_np, y_np)
dl = DataLoader(dataset, batch_size = 10, shuffle=True)
model = THISRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criteon = nn.MSELoss()	# 二分类交叉熵损失函数
i = 1

datas = []
for step in range(100):
    start, end = step *0.8 *np.pi, (step+1)*np.pi* 0.8   # time steps
    # sin 预测 cos
    steps = np.linspace(start, end, 10, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)
    datas.append([steps, x_np, y_np])

lastp = 0
lasty = 0
laststep =0
for row in datas:
    # sin 预测 cos
    steps = row[0]
    x_np = row[1]    # float32 for converting torch FloatTensor
    y_np = row[2]
 
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))
    

    h_state = None
    prediction, h_state = model(x, h_state)   # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
    # !!  下一步十分重要 !!
    h_state = h_state.data  # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错
 
    loss = criteon(prediction, y)     # cross entropy loss
    optimizer.zero_grad()               # clear gradients for this training step
    loss.backward()                     # backpropagation, compute gradients
    optimizer.step()
    newy = y_np[-1]
    newp = prediction.data.numpy()[0, -1, 0]
    plt.plot([laststep, laststep + 1], [lasty, newy], 'r-')
    plt.plot([laststep, laststep + 1], [lastp, newp], 'b-')
    laststep += 1
    lasty = newy
    lastp = newp

    plt.draw(); plt.pause(0.05)


start, end = 1 * np.pi, (1+1)*np.pi   # time steps
# sin 预测 cos
steps = np.linspace(start, end, 10, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
prediction, h_state = model(x, h_state)
y = prediction.detach().numpy()

npdata = [y[0,:,0],y_np]
for i, r in enumerate(npdata):
    plt.plot(r, label='data%d' % (i+1))
plt.legend()
plt.show()
print(1)



for epoch in range(30):
    for i, batch in enumerate(dl): 
        # [seq, b] => [b, 1] => [b]
        x, y = batch
        pred = model(x,h_state)
        loss = criteon(pred, batch.label)
        acc = binary_acc(pred, batch.label).item()
        avg_acc.append(acc)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
