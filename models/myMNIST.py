import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import time

torch.set_num_threads(8)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

batchSize=4
MNIST_ROOT='../data/MNIST'

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=MNIST_ROOT, train=True, download=True,
                   transform=transform),
    batch_size=batchSize, shuffle=True)
test_loader=torch.utils.data.DataLoader(
    datasets.MNIST(root=MNIST_ROOT, train=False, download=True,
                   transform=transform),
    batch_size=batchSize, shuffle=True)

dataLoader={
    'train':train_loader,
    'test':test_loader

}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5)
        # self.conv5=nn.Conv2d(16,20,5)

        self.pool = nn.MaxPool2d(2)
        # self.padding=nn.ReflectionPad2d(2)
        # self.drop=nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(16* 3 * 3, 100)
        self.fc2 = nn.Linear(100,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        # x=x.view(1,28,28)
        # print(x.shape)
        # print(self.conv1.stride)
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        # print(x.shape)
        x=x.view(-1,16*3*3)
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
        print("==============={}/{}===============".format(epoch+1,num_epochs))
        best_acc=0.0
        for phase in ['train','test']:
            running_loss=0.0
            running_corrects=0
            print('current period:{}'.format(phase))
            if phase == 'train':
                net.train(True)
            else:
                net.train(False)
            for i,d in enumerate(dataLoader[phase]):
                # print(i)
                inputs,labels =d

                # labels=torch.IntTensor(labels)
                # optimizer.zero_grad()

                #print(inputs.shape)
                #print(labels.shape)
                if use_gpu:
                    optimizer.zero_grad()
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())# zero the parameter gradients
                else:
                    optimizer.zero_grad()
                    inputs, labels = Variable(inputs), Variable(labels)

                #print(inputs.shape)
                #print(labels.shape)

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


        epoch_loss = running_loss / (i+1)*batchSize
        epoch_acc = running_corrects / (i+1)*batchSize
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
