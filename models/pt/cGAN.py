
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time


import scipy.misc as misc

import os

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.use_gpu = torch.cuda.is_available()
        self.g_dense1=nn.Linear(4*10*10,1024)
        self.z_dense1=nn.Linear(4*16*16,1024)
        self.m_dense1=nn.Linear(1024,4096)
        self.m_dense2=nn.Linear(4096,56*56)


    def forward(self, g, z):
        if self.use_gpu:
            g = g.view(-1, 4*10*10).cuda()
            z = z.view(-1, 4*16*16).cuda()
            # print(g.shape,z.shape)
            g1 = self.g_dense1(g)

            z1=self.z_dense1(z)
            # print('g1.shape: ',g1.shape)
            # print('z1.shape: ',z1.shape)
            m=g1+z1
            # print("m.shape: ",m.shape)

            m1= self.m_dense1(m)

            # print('m1.shape',m1.shape)
            m2=self.m_dense2(m1)
            # print('m2.shape',m2.shape)

            # o = self.m_fc(merged_r)
            #         print('output.shape',o.shape)
            return m2.view((-1, 1, 56, 56)).cuda()
        else:
            g = g.view(-1, 4*10*10)
            z = z.view(-1, 4*16*16)
            print(g.shape,z.shape)
            g1 = self.g_dense1(g)

            z1=self.z_dense1(z)
            print('g1.shape: ',g1.shape)
            print('z1.shape: ',z1.shape)
            m=g1+z1
            print("m.shape: ",m.shape)

            m1= self.m_dense1(m)

            print('m1.shape',m1.shape)
            m2=self.m_dense2(m1)
            print('m2.shape',m2.shape)

            # o = self.m_fc(merged_r)
            #         print('output.shape',o.shape)
            return m2.view((-1, 1, 56, 56))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.g_conv1 = nn.Conv2d(4, 8, 3, padding=1)
        self.x_conv1 = nn.Conv2d(1, 16, 3, padding=0)
        self.x_pool1 = nn.MaxPool2d(2, 2)
        self.x_conv2 = nn.Conv2d(16, 64, 8)
        self.m_conv = nn.Conv2d(72, 128, 3)
        self.m_fc1 = nn.Linear(1 * 128 * 8 * 8, 2048)
        self.m_fc2 = nn.Linear(2048, 256)
        self.m_fc3 = nn.Linear(256, 2)
        self.d_softmax = nn.Softmax()
        self.lrelu=nn.LeakyReLU()

        self.bn_g = nn.modules.BatchNorm2d(num_features=8)
        self.bn_x = nn.modules.BatchNorm2d(num_features=16)
        self.use_gpu=torch.cuda.is_available()
    def forward(self, g, x):
        if self.use_gpu:
            g = g.view(-1, 4, 10, 10).cuda()
            x = x.view(-1, 1, 56, 56).cuda()
            gc1 = self.g_conv1(g)
            gc1 = self.lrelu(gc1)
            gc1 = self.bn_g(gc1)
        #         print('gc1.shape:',gc1.shape)
            xc1 = self.x_conv1(x)
        #         print('xc1.shape:',xc1.shape)
            xp1 = self.lrelu(self.x_pool1(xc1))
            xp1 = self.bn_x(xp1)
            #         print('xp1.shape:',xp1.shape)
            xc2 = self.x_conv2(xp1)
            #         print('xc2.shape:',xc2.shape)
            xp2 = self.lrelu(self.x_pool1(xc2))
            #         print('xp2.shape:',xp2.shape)
            merged = torch.cat([xp2, gc1], 1)
            #         print('merged.shape:',merged.shape)
            mc = self.lrelu(self.m_conv(merged))
            #         print('mc.shape:',mc.shape)
            mc = mc.view(-1, 128 * 8 * 8).cuda()
            m_fc1 = self.m_fc1(mc)
            #         print('m_fc1.shape',m_fc1.shape)
            m_fc2 = self.m_fc2(m_fc1)
            #         print('m_fc2.shape',m_fc2.shape)
            m_fc3 = self.m_fc3(m_fc2)
            #         print('m_fc3.shape',m_fc3.shape)
            return self.d_softmax(m_fc3)
        else:
            g = g.view(-1, 4, 10, 10)
            x = x.view(-1, 1, 56, 56)
            gc1 = self.g_conv1(g)
            gc1 = self.lrelu(gc1)
            gc1 = self.bn_g(gc1)
            #         print('gc1.shape:',gc1.shape)
            xc1 = self.x_conv1(x)
            #         print('xc1.shape:',xc1.shape)
            xp1 = self.lrelu(self.x_pool1(xc1))
            xp1 = self.bn_x(xp1)
            #         print('xp1.shape:',xp1.shape)
            xc2 = self.x_conv2(xp1)
            #         print('xc2.shape:',xc2.shape)
            xp2 = self.lrelu(self.x_pool1(xc2))
            #         print('xp2.shape:',xp2.shape)
            merged = torch.cat([xp2, gc1], 1)
            #         print('merged.shape:',merged.shape)
            mc = self.lrelu(self.m_conv(merged))
            #         print('mc.shape:',mc.shape)
            mc = mc.view(-1, 128 * 8 * 8)
            m_fc1 = self.m_fc1(mc)
            #         print('m_fc1.shape',m_fc1.shape)
            m_fc2 = self.m_fc2(m_fc1)
            #         print('m_fc2.shape',m_fc2.shape)
            m_fc3 = self.m_fc3(m_fc2)
            #         print('m_fc3.shape',m_fc3.shape)
            return self.d_softmax(m_fc3)

class cDCGAN(nn.Module):
    def __init__(self):
        super(cDCGAN, self).__init__()

        self.G_LOSS = nn.CrossEntropyLoss()
        self.rD_LOSS = nn.CrossEntropyLoss()
        self.fD_LOSS = nn.CrossEntropyLoss()
        self.use_gpu = torch.cuda.is_available()
        self.G = Generator()
        self.D = Discriminator()

        if self.use_gpu:
            print('use cuda')
            self.G = self.G.cuda()
            self.D = self.D.cuda()

        self.g_lr = 0.001
        self.d_lr=0.001
        self.batch_size = 25
        self.iters = 1000
        self.epoch = 200

        self.SAVE_DIR = './out/10/'
        try:
            os.makedirs(self.SAVE_DIR)
        except:
            pass

    def setTrainParas(self, batch_num, lr, iters, epoch):
        self.lr, self.batch_num, self.iters, self.epoch = lr, batch_num, iters, epoch

    def feedData(self, GI, ratio=1):
        G, I = GI
        num_of_records = G.shape[0]

        self.dataLoader = {
            'train': [],
            'test': []
        }
        batch_num = num_of_records // self.batch_size
        print("{} records, {} batches of size {} each".format(num_of_records, batch_num, self.batch_size))
        cur_batch_G = []
        cur_batch_I = []
        batched_data = []
        for i in range(num_of_records):
            cur_batch_G.append(G[i])
            cur_batch_I.append(I[i])
            if i % self.batch_size == self.batch_size - 1:
                batched_data.append((np.array(cur_batch_G), np.array(cur_batch_I)))
                cur_batch_G = []
                cur_batch_I = []
        self.dataLoader['train'] = batched_data[:int(batch_num * ratio)]
        self.dataLoader['test'] = batched_data[int(batch_num * ratio):]
        print(self.dataLoader['train'][0][0][0].shape, self.dataLoader['train'][0][0][1].shape)

    def convert2Cuda(self, l):
        return [_.cuda() for _ in l]

    def trainNetwork(self):
        G_optimizer = optim.SGD(self.G.parameters(), lr=self.g_lr, momentum=0.9)
        # G_optimizer= optim.Adadelta(self.G.parameters())
        D_optimizer = optim.SGD(self.D.parameters(), lr=self.d_lr, momentum=0.9)
        # D_optimizer=optim.Adadelta(self.D.parameters())

        for e in range(self.epoch):
            G_losses, D_losses = [], []
            eporch_starttime = time.time()
            print('Epoch {}/{}'.format(e + 1, self.epoch))
            print("-" * 10)
            for phase in ['train', 'test']:
                print('Current Phase:{}'.format(phase))

                if phase == 'train':
                    # G_optimizer.step()
                    # D_optimizer.step()
                    self.D.train(True)
                    self.G.train(True)
                else:
                    self.D.train(False)
                    self.G.train(False)

                for i, data in enumerate(self.dataLoader[phase], 0):
                    z = torch.randn((self.batch_size, 4,16,16))

                    g, img = data
                    G_optimizer.zero_grad()
                    D_optimizer.zero_grad()
                    g, img, z = Variable(torch.Tensor(g)), Variable(torch.Tensor(img)), Variable(torch.Tensor(z))
                    rd_label = Variable(torch.LongTensor(np.zeros((self.batch_size))), requires_grad=False)
                    #                     rd_label.requires_grad=False
                    fd_label = Variable(torch.LongTensor(np.ones((self.batch_size))), requires_grad=False)
                    #                     fd_label.requires_grad=False

                    if self.use_gpu:
                        g, img, z, rd_label, fd_label, self.convert2Cuda([g, img, z, rd_label, fd_label])


                    x = self.G(g, z)
                    r_d_out = self.D(g, img)
                    f_d_out = self.D(g, x)

                    if self.use_gpu:
                        real_d_loss = self.rD_LOSS(r_d_out, rd_label.cuda())
                        fake_d_loss = self.fD_LOSS(f_d_out, fd_label.cuda())
                        d_loss = 0.5*(real_d_loss + fake_d_loss)
                        g_loss = self.G_LOSS(f_d_out, rd_label.cuda())
                    else:
                        real_d_loss = self.rD_LOSS(r_d_out, rd_label)
                        fake_d_loss = self.fD_LOSS(f_d_out, fd_label)
                        d_loss = 0.5*(real_d_loss + fake_d_loss)
                        g_loss = self.G_LOSS(f_d_out, rd_label)


                    G_losses.append(g_loss.data[0])
                    D_losses.append(d_loss.data[0])
                    if phase == 'train':
                        d_loss.backward(retain_graph=True)
                        D_optimizer.step()
                        g_loss.backward()
                        G_optimizer.step()

                    if i % 50 == 0:
                        ind = np.random.randint(0, self.batch_size)
                        print("{}  G_loss: {}, D_loss:{}".format(phase, g_loss.data[0], d_loss.data[0]))
                        #                         display(g,img,x,meta=' ')
                        misc.imsave(self.SAVE_DIR + '{}_{}_{}_img.png'.format(e, phase, i),
                                    np.array(img[ind].data).reshape((56, 56)))
                        misc.imsave(self.SAVE_DIR + '{}_{}_{}_x.png'.format(e, phase, i),
                                    np.array(x[ind].data).reshape((56, 56)))
            if e % 10 == 9:
                self.saveCheckpoint('{}'.format(e))

    def saveCheckpoint(self, e):
        torch.save(self.D.state_dict(),'checkpoint/D_temp{}.pth.tar'.format(e))
        torch.save(self.G.state_dict(),'checkpoint/G_temp{}.pth.tar'.format(e))

    def loadCheckpoint(self, e):
        self.D.load_state_dict(torch.load('checkpoint/D_temp{}.pth.tar'.format(e)))
        self.G.load_state_dict(torch.load('checkpoint/G_temp{}.pth.tar'.format(e)))


if __name__ == '__main__':
    data = [np.load('../../data/doodle/G1000.npy'), np.load('../../data/doodle/I1000.npy')]
    dcgan = cDCGAN()
    dcgan.feedData(data,ratio=0.8)
#    dcgan.loadCheckpoint('19')
    dcgan.trainNetwork()

