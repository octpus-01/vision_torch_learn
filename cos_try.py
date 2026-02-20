# import everything
import torch
import torch.nn as nn  # neural network
from torch.utils.data import DataLoader    # load data through batch for train
from torch.utils.data import TensorDataset   # pack x, y to data
from torch.optim.adam import Adam   # optimizer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# set matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# dataset prepare
x = np.linspace(-2*np.pi,2*np.pi,400)
y = np.cos(x)

X = np.expand_dims(x,axis=1)   
Y = np.expand_dims(y, axis=1)
#Y = y.reshape(400,-1)

dataset = TensorDataset(torch.tensor(X,dtype=torch.float),torch.tensor(Y,dtype=torch.float))
dataloader = DataLoader(dataset,batch_size=10,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=1,out_features=10),nn.ReLU(),
            nn.Linear(10,100),nn.ReLU(),
            nn.Linear(100,10),nn.ReLU(),
            nn.Linear(10,1)
        )
    
    def forward(self, input:torch.FloatTensor):
        return self.net(input)
    
net = Net()

optim = Adam(net.parameters(),lr = 0.001)
Loss = nn.MSELoss()

for epoch in range(100):
    loss = None
    for batch_x , batch_y in dataloader:
        y_predict = net(batch_x)
        loss=Loss(y_predict,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    if (epoch+1)%10 == 0:
        if loss is not None:
            print("训练步骤：{0}，模型损失{1}".format(epoch+1,loss.item()))


predict = net(torch.tensor(X,dtype=torch.float))

plt.figure(figsize=(12,7), dpi=160)
plt.plot(x,y,label="real",marker="X")
plt.plot(x,predict.detach().numpy(),label="predict",marker='o')
plt.xlabel("x",size=15)
plt.ylabel("cos(x)",size=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(fontsize=15)
plt.savefig('cos.png')

