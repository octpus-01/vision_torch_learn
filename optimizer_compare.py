# import everything
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# set matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# dataset prepare
x = torch.unsqueeze(torch.linspace(-1,1,500), dim=1)
y = x.pow(3)

# set parameters
LR = 0.01
batch_size = 15
epoches = 5
torch.manual_seed(10)

# load data
dataset = TensorDataset(x,y)
loader  = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(n_input, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        x = torch.relu(self.hidden_layer(input))
        output = self.output_layer(x)
        return output
    

def train():
    net_SGD = Net(1,10,1)
    net_Momentum = Net(1,10,1)
    net_AdaGrad = Net(1,10,1)
    net_RMSprop = Net(1,10,1)
    net_Adam = Net(1,10,1)
    nets = [net_SGD, net_Momentum,net_AdaGrad,net_RMSprop,net_Adam]

    # optimzer
    optimizer_SGD = SGD(net_SGD.parameters(), lr=LR, momentum=0, weight_decay=0)
    optimizer_Momentum = SGD(net_Momentum.parameters(), lr=LR, momentum=0.9)
    optimizer_AdaGrad = Adagrad(net_AdaGrad.parameters(), lr=LR, weight_decay=0)
    optimizer_RMSprop = RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    optimizer_Adam = Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

    optimizers = [optimizer_SGD,optimizer_Momentum,optimizer_AdaGrad,optimizer_RMSprop,optimizer_Adam]

    loss_func = nn.MSELoss()
    losses = [[],[],[],[],[]]

    for epoch in range(epoches):
        for step, (batch_x,batch_y) in enumerate(loader):
            for net, optimizer, loss_list in zip(nets,optimizers,losses):
                pred_y = net(batch_x)
                loss = loss_func(pred_y,batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.data.numpy())
                print(loss.data.numpy())

    plt.figure(figsize=(12,7))
    labels = ['SGD','Momentum','AdaGrad','RMSprop','Adam']
    for i, loss in enumerate(losses):
        plt.plot(loss, label= labels[i])
    plt.legend(loc = 'upper right',fontsize = 15)
    plt.tick_params(labelsize=13)
    plt.xlabel("训练步骤",size=15)
    plt.ylabel("模型损失",size=15)
    plt.ylim((0,0.3))

    plt.savefig('optimizerscompare.png')


train()
