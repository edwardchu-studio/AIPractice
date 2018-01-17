# Required Python Packages
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from time import sleep
from models.utility import Utility

import librosa

# plt.ion()   # interactive mode

def sigsofttest():
    util=Utility()
    graph_x = range(-21, 21)
    graph_y_sig = util.sigmoid(graph_x)
    graph_y_soft = util.softmax(graph_x)

    print("Graph X readings: {}".format(graph_x))
    print("Graph Y sigmoid readings: {}".format(graph_y_sig))
    print("Graph Y softmax readings: {}".format(graph_y_soft))

    util.line_graph(graph_x, graph_y_sig, "Inputs", "Sigmoid Scores")
    util.line_graph(graph_x, graph_y_soft, "Inputs", "Softmax Scores")
def simpleTest():

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    dtype=torch.FloatTensor
    # Create random input and output data

    # Create random Tensors to hold input and outputs, and wrap them in Variables.
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Variables during the backward pass.
    x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
    y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

    # Create random Tensors for weights, and wrap them in Variables.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Variables during the backward pass.
    w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        # Forward pass: compute predicted y using operations on Variables; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # Compute and print loss using operations on Variables.
        # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
        # (1,); loss.data[0] is a scalar value holding the loss.
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.data[0])

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Variables with requires_grad=True.
        # After this call w1.grad and w2.grad will be Variables holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Update weights using gradient descent; w1.data and w2.data are Tensors,
        # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
        # Tensors.
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

        # Manually zero the gradients after updating weights
        w1.grad.data.zero_()
        w2.grad.data.zero_()
def nnTest():
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
    x = Variable(torch.randn(N, D_in))
    y = Variable(torch.randn(N, D_out), requires_grad=False)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Variables for its weight and bias.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(size_average=False)

    learning_rate = 1e-4
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Variable of input data to the Module and it produces
        # a Variable of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Variables containing the predicted and true
        # values of y, and the loss function returns a Variable containing the
        # loss.
        loss = loss_fn(y_pred, y)
        print(t, loss.data[0])

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Variables with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Variable, so
        # we can access its data and gradients like we did before.
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data
def audioTest():
    # 1. Get the file path to the included audio example
    filename = librosa.util.example_audio_file()

    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(filename)

    # 3. Run the default beat tracker
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    # 4. Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print('Saving output to beat_times.csv')

    librosa.output.times_csv('../cache/beat_times.csv', beat_times)

    

if __name__ == '__main__':
    audioTest()
