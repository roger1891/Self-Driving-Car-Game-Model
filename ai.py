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
    
# implementing experience replay class
        
class ReplayMemory(object):
    def __init__(self, capacity): # constuctor
        self.capacity = capacity # number of transitions in event
        self.memory = [] #list for memories
        
    def push(self, event): #push function
        self.memory.append(event) # append event to list
        if len(self.memory) > self.capacity: # make sure it has 100 events
            del self.memory[0] #remove first element to ensure same length for capacity
            
    def sample(self, batch_size): #sample fucntions
        #zip function = if list = ((1,2,3), (4,5,6)) will turn to (1,4), (2,3) (5,6)
        samples = zip(*random.sample(self.memory, batch_size)) #random samples that have a fixed size of batch size
        return map(lambda x : Variable(torch.cat(x, 0)), samples) # return map and concatinate with first dimension and conert sesors into torch variables
            
        