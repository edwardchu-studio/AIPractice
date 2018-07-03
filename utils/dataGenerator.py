from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell


class doodleGenerator:
    def __init__(self):
        self.edges=['u','r','d','l']
        self.MNIST= input_data.read_data_sets("../../MNIST_data/", one_hot=True, reshape=[])
        self.MNIST_TRAIN_NUM = self.MNIST.train.num_examples
        self.MNIST_TRAIN_SET = self.MNIST.train.images, self.MNIST.train.labels, np.argmax(self.MNIST.train.labels, axis=1)
        
    def getRecordInfo(self,ind):
        return self.MNIST_TRAIN_SET[0][ind].reshape((28, 28)), self.MNIST_TRAIN_SET[1][ind], self.MNIST_TRAIN_SET[2][ind]

    def fetchRandomPair(self,shape=(10,10,4)):
        g = np.zeros(shape=shape)
        ind1, ind2 = np.random.randint(0, self.MNIST_TRAIN_NUM), np.random.randint(0, self.MNIST_TRAIN_NUM)
        pos = np.random.randint(0, 4)
        img1, label1, num1 = self.getRecordInfo(ind1)
        img2, label2, num2 = self.getRecordInfo(ind2)
        g[num1, num2, pos] = 1
        plt.subplot(131)
        plt.imshow(img1, cmap='gray')
        plt.subplot(132)
        plt.imshow(img2, cmap='gray')
        plt.subplot(133)
        newImg = {
            0: np.concatenate([np.zeros((56, 14)), np.concatenate([img2, img1], axis=0), np.zeros((56, 14))], axis=1),
            1: np.concatenate([np.zeros((14, 56)), np.concatenate([img1, img2], axis=1), np.zeros((14, 56))], axis=0),
            2: np.concatenate([np.zeros((56, 14)), np.concatenate([img1, img2], axis=0), np.zeros((56, 14))], axis=1),
            3: np.concatenate([np.zeros((14, 56)), np.concatenate([img2, img1], axis=1), np.zeros((14, 56))], axis=0)
        }.get(pos)
        plt.imshow(newImg, cmap='gray')
        print("{}--{}-->{}".format(num1, self.edges[pos], num2))
        return g, newImg


    def createDataSet(self,num=100000):
        G = []
        I = []
        for i in range(num):
            g, img = self.fetchRandomPair()
            G.append(g)
            I.append(img)

        I = np.array(I).reshape((num, 56, 56, 1))
        G = np.array(G)
        np.save('G' + str(num), G)
        np.save('I' + str(num), I)
        self.TRAIN_SET=[G,I]
        return G, I

if __name__ == '__main__':
    dg=doodleGenerator()
    data=dg.createDataSet()
