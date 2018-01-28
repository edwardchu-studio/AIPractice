import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

torch.set_num_threads(8)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batchSize=4
MNIST=torchvision.datasets.MNIST('../data/MNIST')
dataSet={
    'train':MNIST.train_data,
    #'test': MNIST.test_data
}
labelSet={
    'train':MNIST.train_labels,
    #'test':MNIST.test_labels
}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.conv3 = nn.Conv2d(8,12,5)
        self.conv4 = nn.Conv2d(12,16,5)
        # self.conv5=nn.Conv2d(16,20,5)

        self.pool = nn.MaxPool2d(2)
        # self.padding=nn.ReflectionPad2d(2)
        # self.drop=nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(12 * 3 * 3, 100)
        self.fc2 = nn.Linear(100,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=x.view(1,28,28)
        print(x.shape)
        print(self.conv1.stride)
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=self.fc3(self.fc2(self.fc1(x)))

        return F.sigmoid(x)


net=Net()
use_gpu=torch.cuda.is_available()

dataSetSize={
    'train':60000,
    'test':20000

}

if use_gpu:
    net=net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
best_model_wts = net.state_dict()
best_acc = 0.0
num_epochs=32

def trainNetworks():
    since = time.time()
    for epoch in range(num_epochs):
        running_loss=0.0
        running_corrects=0
        best_acc=0.0
        for phase in ['train']:
            print('current period:{}'.format(phase))
            if phase == 'train':
                net.train(True)
            else:
                net.train(False)
            for i,d in enumerate(zip(dataSet[phase],labelSet[phase])):
                print(i)
                inputs,labels =d
                inputs.unsqueeze_(0)
                print(inputs.shape)
                labels=torch.IntTensor(labels)
                optimizer.zero_grad()

                if use_gpu:
                    optimizer.zero_grad()
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())# zero the parameter gradients
                else:
                    optimizer.zero_grad()
                    inputs, labels = Variable(inputs), Variable(labels)

                outputs=net(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase =='train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]

                running_corrects += torch.sum(preds == labels.data)
                if i % 200 == 199:    # print every 1000 mini-batches
                    print("current period:{}, with current i:{}, current batch loss:{}, and running correct:{}".format(phase,i,loss.data[0],running_corrects))
                    print('[%d, %5d] loss: %.3f, Acc: %.3f' %
                          (epoch + 1, i + 1, running_loss / (batchSize*(i+1)),running_corrects/(batchSize*(i+1))))


        epoch_loss = running_loss / dataSetSize[phase]
        epoch_acc = running_corrects / dataSetSize[phase]
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

        # deep copy the model

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = net.state_dict()

    time_elapsed=time.time()-since

    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def main():
    trainNetworks()

if __name__ == '__main__':
    main()
