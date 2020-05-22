import csv
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot as plt

#%%
bins = 128
seedmax = 20 # opens seed files 0 - 19. Lost too much data due to kernel crashes, so these got broken up
trainx = []
trainy = []
validx = []
validy = []

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/'

# %% This is not a ... pythonic [barf]... way of reading data, but python is stupid about pointers, so deal with it
for i in range(seedmax):
    with open(path+'test_pots/test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])
    with open(path+'test_out/test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
    with open(path+'valid_pots/valid_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validx.append([float(num) for num in row])
    with open(path+'valid_out/valid_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validy.append([float(num) for num in row])
            
# %%
trainx_np = numpy.array(trainx)
trainy_np = numpy.array(trainy)

xmax = numpy.amax(numpy.array([numpy.amax(k) for k in trainx_np]))
ymax = numpy.amax(numpy.array([numpy.amax(k) for k in trainy_np]))

trainx_norm = (trainx_np / xmax * 0.99) + 0.01
trainy_norm = (trainy_np / ymax * 0.98) + 0.01

# In[4]:
# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        # inputs = numpy.array(inputs_list, ndmin=2).T
        # targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        # inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# In[5]:
# number of input, hidden and output nodes
input_nodes = 127
hidden_nodes = 127
output_nodes = 127

# learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# In[7]:
# load the mnist training data CSV file into a list
training_data_file = open("/home/domi/Dokumente/Machine Learning/myownnn/mnist-in-csv/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# In[8]:
# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 10000

for e in range(epochs):
    print('start epoch: ', e + 1)
    # go through all records in the training data set
    for record in range(len(trainx_norm)):
        inputs = trainx_norm[record]
        targets = trainy_norm[record]
        n.train(inputs, targets)
        pass
    print('epoch: ', e + 1, ' complete')
    pass


# test with our own image 

# In[9]:
# query the network
outputs = n.query(trainx_norm[10])
