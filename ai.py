# AI for Self Driving Car

# Importing the libraries

import numpy as np #play and work with arrays
import random # random samples
import os # load the model
import torch #it can handle dynamic graphs
import torch.nn as nn #neural networks
import torch.nn.functional as F #different functions to be used
import torch.optim as optim #optimizers to perform categorization
import torch.autograd as autograd #to make conversions
from torch.autograd import Variable

#Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action): # refer to this object, 5 sensors, action
        super(Network, self).__init__() # use tools of module class
        self.input_size = input_size # assign input size
        self.nb_action = nb_action # assign nb_action
        self.fc1 = nn.Linear(input_size, 30) # full connection between hidden layer to input layer
        self.fc2 = nn.Linear(30, nb_action) # full connection between hidden layer to output layer
        
    # activate neuron function for forward propagation
    def forward(self, state, ): # self, directional state
        x = F.relu(self.fc1(state)) #activate hidden neurons
        q_values = self.fc2(x) # activate ouput neurons
        return q_values # return q_values of whether to go left, right, etc.
        

