import numpy as np               # for array creation
import matplotlib.pyplot as plt  # for data visualization

############### TASK 1 ########################################################################################################################################################

# Generate input values
x = np.random.uniform(0, 1, size=100)
# Generate targets
t = x**3-x**2+1
# plot function
plt.plot(x,t,".")              
plt.show()

################ TASK 2 ########################################################################################################################################################

# ReLu function
def relu(z):
    return np.maximum(0,z)

# ReLu derivative function
def d_ReLu(z):
    return (z > 0).astype(int)

# Layer class
class Layer:
    
    # constructor
    def __init__(self, n_units, input_units):
        self.n_units = n_units                                        # number of units in layer
        self.input_units = input_units                                # number of units in preceding layer
        
        self.bias = np.zeros((self.n_units,1))                        # bias vector
        self.weight = np.random.rand(self.n_units, self.input_units)  # weight matrix (!!!only random between 0-1)
        
        self.input = None                                             # empty attribute for layer-input
        self.preactivation = None                                     # empty attribute for layer-preactivation
        self.activation = None                                        # empty attribute for layer-activation (= output)
    
    
    # Method 1: returns each unit's activation (i.e. output) using ReLu as activation function
    def forward_step(self):
        w_b = np.append(self.weight,self.bias,axis=1)       # link weights and bias in one matrix
        input_new = np.append(self.input,[[1]],axis=0)      # add 1 to input vector to make up for w_b
        self.preactivation = w_b@input_new                  # calculate drive/preactivation
        self.activation = relu(self.preactivation)          # calculate activation/output with ReLu activation function  
        return self.activation
    
    
    # Method 2: updates each unit's parameters
    # setting learning rate to 0.05 by default as it generally should be smaller
    def backward_step(self, following_layer, y, t, error_signal_previous_layer, weight_new, learning_rate = 0.05):
        # get derivative of activation function respective the drive
        x1 = d_ReLu(self.preactivation)
        
        # get derivative of L with respect to activation
        if following_layer == None:
            # for output layer use loss functions derivative
            x2 = -(t-y) 
        else:
            # for hidden layer(s) use error signal of l+1 * weights of previous layer
            x2 = np.multiply(error_signal_previous_layer,weight_new)
        
        # get error signal
        error_signal = np.multiply(x1, x2)   
        # get transpose of the layer's input
        input_transposed = np.transpose(self.input)
        # get rate of change for biases b and weights w
        b_gradient = np.multiply(error_signal, 1)                          
        w_gradient = np.multiply(error_signal, input_transposed) 
        # get new parameters
        bias_new = np.subtract(self.bias, np.multiply(learning_rate, b_gradient))
        weight_new = np.subtract(self.weight, np.multiply(learning_rate, w_gradient))
        # update attributes of layer
        setattr(self, 'bias', bias_new)
        setattr(self, 'weight', weight_new)
        # return new biases and weights of layer
        return bias_new, weight_new, error_signal

################ TASK 3 #########################################################################################################################################################

# MLP class
class MLP:
    
    # constructor
    def __init__(self, n_units_in_layers):
        self.n_units_in_layers = n_units_in_layers  # array storing # of units in layer 'index'
        self.mlp_layers = np.empty(0)               # array that stores all layer objects of MLP
        self.following_layer = None                 # stores layer l+1
    
    # Method 1: pass input through entire network
    # x = input value of specific training data point
    def forwardpropagation(self, x):
        
        # make single input value to 2-d array 
        input_layer = [[x]] # NOTE: Only works for 1 input neuron!!!
    
        # make input array to vector
        input_layer = np.transpose(input_layer)
        
        # iterate over all layers starting at 1st hidden layer
        for i, n_units in enumerate(self.n_units_in_layers[1:]):
            # create object of class Layer
            input_units = self.n_units_in_layers[i]
            layer_x = Layer(n_units, input_units)
            # assign input vector to attribute
            setattr(layer_x,'input', input_layer)
            # store layer in array of all layers
            setattr(layer_x, 'mlp_layers', np.append(self.mlp_layers,layer_x))
            # conduct forward step + save outputs as inputs for next layer
            input_layer = layer_x.forward_step()
            # save current layer to array
            a = np.append(self.mlp_layers,layer_x)
            setattr(self,'mlp_layers', a)
    
        # return output y of MLP
        return input_layer


    # Method 2: update all weights and biases in the network given a loss value
    def backpropagation(self, y, t):
        error_signal_previous_layer = 0 # no previous layer for output layer
        weight_new = None
        # backward iteration over layer objects
        for layer in self.mlp_layers[::-1]:
            # conduct backward step
            bias_new, weight_new, error_signal_previous_layer = layer.backward_step(self.following_layer, y, t, error_signal_previous_layer, weight_new, learning_rate=0.02)
            # set current layer as l+1 for previous layer l
            setattr(layer, 'following_layer', layer)

################ TASK 4 #########################################################################################################################################################

# mean squared error function
def mse(y,t):
    result = np.multiply(np.divide(1,2), np.square(np.subtract(t,y)))
    return result

network = MLP([1,10,1])    # i = # of layer, x[i] = # of units in layer i
n_epochs = 1000

epochs_data = np.arange(1,n_epochs+1) # x-axis for data vizualization
loss_data = np.zeros(n_epochs)        # array containing average loss of each epoch
loss_x = np.zeros(x.size)             # array containing loss values for each training data within 1 epoch

count_epoch = 0   #helper for counting epochs
count_tdata = 0   #helper for counting training data

# for each epoch
while count_epoch < n_epochs:
    
    # for each input data per epoch
    while count_tdata < x.size:
        # perform forwardpropagation
        y = network.forwardpropagation(x[count_tdata])
        # record the loss of current data point by using MSE as loss function
        loss_x[count_tdata] = mse(y, t[count_tdata])
        # perform backpropagation
        network.backpropagation(y, t[count_tdata])
        # prepare for next input data
        count_tdata += 1

    print(count_epoch)

    # reset count for next epoch
    count_tdata = 0
    # compute average loss of current epoch
    loss_data[count_epoch] = np.mean(loss_x)
    # prepare for next epoch
    count_epoch += 1

################ TASK 5 #########################################################################################################################################################

# plot training progress
plt.plot(epochs_data, loss_data)
plt.show()