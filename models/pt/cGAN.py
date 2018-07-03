import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from graphviz import Digraph
import torchvision

from tensorflow.examples.tutorials.mnist import input_data


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.g_conv1=nn.Conv2d(4,8,3,padding=0)
        self.z_conv1=nn.Conv2d(16,12,5,padding=0)
        self.z_pool1=nn.MaxPool2d(2,2)
        self.z_conv2=nn.Conv2d(12,8,7)
        self.m_fc=nn.Linear(10*10*16,28*28*1)

    def forward(self, g,z):
        gc1 = self.g_conv1(g)
        gc1=F.relu(gc1)
        zc1 = self.z_pool1(self.z_conv1(z))
        zc1=F.relu(zc1)
        zc2 = self.z_conv2(zc1)
        zc2=F.relu(zc2)
        merged = torch.cat([gc1, zc2], 2)
        o = self.m_fc(merged)
        return o.reshape((28, 28, 1))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()


        self.g_conv1 = nn.Conv2d(4, 8, 3, padding=0)
        self.x_conv1 = nn.Conv2d(1, 16, 3, padding=0)
        self.x_pool1 = nn.MaxPool2d(2, 2)
        self.x_conv2 = nn.Conv2d(16, 64, 5)

        self.m_conv = nn.Conv2d(72, 128, 3)
        self.m_fc1 = nn.Linear(8 * 8 * 128, 2048)
        self.m_fc2 = nn.Linear(2048, 256)
        self.m_fc3 = nn.Linear(256, 1)

    def forward(self, g,x):
        gc1 = self.g_conv1(g)
        gc1=F.relu(gc1)
        xc2 = self.x_conv2(F.relu(self.x_pool1(self.x_conv1(x))))
        xc2=F.relu(xc2)
        merged = torch.cat([xc2, gc1], 2)
        mc = self.m_conv(merged)
        m_fc = self.m_fc3(self.m_fc2(self.m_fc1(mc)))
        return torch.sigmoid(m_fc)



class cDCGAN(nn.Module):
    def __init__(self):
        super(cDCGAN, self).__init__()
        self.G=Generator()
        self.D=Discriminator()
        self.G_LOSS=nn.CrossEntropyLoss()
        self.rD_LOSS=nn.CrossEntropyLoss()
        self.fD_LOSS=nn.CrossEntropyLoss()


    def setTrainParas(self,batch_num,lr,iters,epoch):
        self.lr,self.batch_num,self.iters,self.epoch=lr,batch_num,iters,epoch

    def feedTrainData(self,GI,tGI):
        self.data={
            'train':GI,
            'test':tGI
        }

    def trainNetwork(self):
        G_optimizer=optim.SGD(self.G.parameters(),lr=self.lr,momentum=0.9)
        D_optimizer=optim.SGD(self.D.parameters(),lr=self.lr,momentum=0.9)

        for  e in range(self.epoch):
            G_losses, D_losses = [], []
            eporch_starttime=time.time()
            print('Epoch {}/{}'.format(e+1,self.epoch))
            print("-"*10)
            for phase in ['train','test']:
                print('Current Phase:{}'.format(phase))

                if phase=='train':
                    G_optimizer.step()
                    D_optimizer.step()
                    self.D.train(True)
                    self.G.train(True)
                else:
                    self.D.train(False)
                    self.G.train(False)

                for i, data in enumerate(self.data[phase],0):
                    z=torch.randn((32,32,16))

                    g,img=data
                    G_optimizer.zero_grad()
                    D_optimizer.zero_grad()
                    g,img,z=Variable(g),Variable(img),Variable(z)

                    x=self.G(g,z)
                    r_d_out=self.D(g,img)
                    f_d_out=self.D(g,x)
                    real_d_loss=self.rD_LOSS(r_d_out,torch.ones((1)))
                    fake_d_loss=self.fD_LOSS(f_d_out,torch.zeros((1)))
                    d_loss=real_d_loss+fake_d_loss
                    g_loss=self.G_LOSS(f_d_out,torch.ones((1)))

                    print("G_loss: {}, D_loss:{}".format(g_loss,d_loss))
                    G_losses.append(g_loss)
                    D_losses.append(d_loss)
                    if phase=='train':
                        g_loss.backward()
                        G_optimizer.step()
                        d_loss.backward()
                        D_optimizer.step()

