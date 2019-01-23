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
            
    # implementing deep q learning
    class Dqn():
        def __init__(self, input_size, nb_action, gamma): #constructor
            self.gamma = gamma # delay coefficient
            self.reward_window = [] # reward window
            self.model = Network(input_size, nb_action) # neural network
            self.memory = ReplayMemory(100000) # memory
            self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #connect optimizer to neural network
            self.last_state = torch.Tensor(input_size).unsqueeze(0) #last state: 5 elements of input states 
            self.last_action = 0 # last action
            self.last_reward = 0 # last reward
                    
        def select_action(self, state): # action function
            probs = F.softmax(self.model(Variable(state, volatile = True)) * 7) # probabiliy distribution for q values
            action = probs.multinomial() # ranodm draw for final action
            return action.data[0,0] #return
        
        def learn(self, batch_state, batch_next_state, batch_reward, batch_action): #learn function
            outputs = self.model(batch_state).gather(1, batch_action).unsqueeze(1).squeeze(1) #output of model
            next_outputs = self.model(batch_next_state).detach().max(1)[0]
            target = self.gamma * next_outputs + batch_reward
            td_loss = F.smooth_l1_loss(outputs, target) #best loss function for q learning
            self.optimizer.zero_grad() #weights
            td_loss.backward(retain_variables = True) #backward propagation
            self.optimizer.step() #update weights
            
        #connection between AI and game
        def update(self, reward, new_signal): # last reward = reward, last signal = new_signal in map class
            new_state = torch.Tensor(new_signal).float().unsqueeze(0) #update all elements in transition
            self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #update memory
            action = self.select_action(new_state) # action state
            if len(self.memory.memory) > 100 : # if memory is larger than 100
                batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100) # learn from 100 transitions of memory
                self.learn(batch_state, batch_next_state, batch_reward, batch_action) #learning
            self.last_action = action #action
            self.last_action = new_state #new state
            self.last_reward = reward #reward
            self.reward_window.append(reward)
            if len(self.reward_window) > 1000: #if larger than a 1000
                del self.reward_window[0] # delete
            return action # return
        
        def score(self): #score function
            return sum(self.reward_window)/(len(self.reward_window)+1.) #score
        
        def save(self): #save functino for reusabilitly
            torch.save({'state_dict': self.model.state_dict(), #save model in python dictionary      
                        'optimizer' : self.optimizer.state_dict(),
                       }, 'previous_brain.pth') #name of file
        def load(self): #load file
            if os.path.isfile('previous_brain.pth'):
                print("=> loading checkpoint... ")
                checkpoint = torch.load('previous_brain.pth')
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("finish !")
            else:
                print("checkpoint not found...")        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            