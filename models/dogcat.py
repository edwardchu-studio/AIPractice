import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from graphviz import Digraph

torch.set_num_threads(8)
transform = transforms.Compose(
    [
        transforms.Resize((256,256), interpolation=2),
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batchSize=4
trainDataSet=torchvision.datasets.ImageFolder('../data/dogvscat/train',transform=transform)
trainDataLoader=torch.utils.data.DataLoader(trainDataSet,batch_size=batchSize,shuffle=True)
testDataSet=torchvision.datasets.ImageFolder('../data/dogvscat/test',transform=transform)
testDataLoader=torch.utils.data.DataLoader(testDataSet,batch_size=batchSize,shuffle=True)
valDataSet=torchvision.datasets.ImageFolder('../data/dogvscat/val',transform=transform)
valDataLoader=torch.utils.data.DataLoader(valDataSet,batch_size=batchSize,shuffle=True)

classes=trainDataLoader.dataset.classes
dataSetSize={
    'train':len(trainDataSet.imgs),
    'val':len(valDataSet.imgs),
    'test':len(testDataSet.imgs)
}

use_gpu = torch.cuda.is_available()

dataLoader={
    'train':trainDataLoader,
    'val':valDataLoader,
    'test':testDataLoader
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 10, 5)
        self.conv3 = nn.Conv2d(10,12,5)
        self.conv4 = nn.Conv2d(12,16,5)
        self.conv5=nn.Conv2d(16,20,5)

        self.pool = nn.MaxPool2d(2, 2)
        self.padding=nn.ReflectionPad2d(2)

        self.drop=nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(20 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3=nn.Linear(512,2)


    def forward(self, x):
        #print(x.size())

        x=F.relu(self.conv1(x))
        x = self.pool(self.padding(x))
        #print(x.size())

        x = F.relu(self.conv2(x))
        x=self.pool(self.padding(x))
        #print(x.size())

        x = F.relu(self.conv3(x))
        x=self.pool(self.padding(x))
        #print(x.size())

        x=self.drop(x)

        x = F.relu(self.conv4(x))
        x=self.pool(self.padding(x))
        #print(x.size())

        x = F.relu(self.conv5(x))
        x=self.pool(self.padding(x))
        #print(x.size())

        x=self.drop(x)

        x= x.view(-1, 20 * 8* 8)

        x=self.fc1(x)
        x = F.relu(x)

        x=self.drop(x)

        x = self.fc2(x)

        x = F.sigmoid(self.fc3(x))
        return x

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

net=Net()
if use_gpu:
    net=net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
since = time.time()
best_model_wts = net.state_dict()
best_acc = 0.0
num_epochs=32


def trainNetworks():
    since=time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train','val','test']:

            print("current period: {}, with data set of size: {}".format(phase,dataSetSize[phase]))
            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                optimizer.step()
                net.train(True)  # Set model to training mode
            else:
                 net.train(False)  # Set model to evaluate mode

            for i, data in enumerate(dataLoader[phase], 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    optimizer.zero_grad()
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())# zero the parameter gradients
                else:
                    optimizer.zero_grad()
                    inputs, labels = Variable(inputs), Variable(labels)
                # forward + backward + optimize
                outputs = net(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
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


    torch.save(best_model_wts,'../cache/dogvscat/dogvscat1.pt')

    time_elapsed=time.time()

    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

def main():
    trainNetworks()

if __name__ == '__main__':
    main()
