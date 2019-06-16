# -*- coding: utf-8 -*-

import numpy as np
import h5py
import time
import copy
from random import randint
import sys

# load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

####################################################################################
# Implementation of stochastic gradient descent algorithm 
# for convolution nerual networks with multiple channels

# number of inputs
num_inputs_y = 28
num_inputs_x = 28
num_inputs = num_inputs_y * num_inputs_x

# number of outputs
num_outputs = 10

# nonlinearity type
#func = 'tanh'
#func = 'sigmoidal'
func = 'ReLU'

# filter/kernel size
kernel_y = 3
kernel_x = 3

# number of channels
num_channels = 3

# initialization
model = {}
model['W'] = np.random.randn(num_outputs,num_inputs_y-kernel_y+1,num_inputs_x-kernel_x+1,num_channels) / np.sqrt(num_inputs)
model['K'] = np.random.randn(kernel_y,kernel_x,num_channels) / np.sqrt(kernel_y*kernel_x)
model['b'] = np.random.randn(num_outputs) / np.sqrt(num_outputs)
model_grads = copy.deepcopy(model)

# element-wise nonliearity
def sigma(z):
    if func == 'tanh':
        ZZ = np.tanh(z)
    elif func == 'sigmoidal':
        ZZ = np.exp(z)/(1 + np.exp(z))
    elif func == 'ReLU':
        ZZ = np.maximum(z,0)
    else:
        sys.exit("Unsupported function type!")
    return ZZ

# element-wise nonliearity, derivative
def d_sigma(z):
    if func == 'tanh':
        dZZ = 1.0 - np.tanh(z)**2
    elif func == 'sigmoidal':
        dZZ = np.exp(z)/(1 + np.exp(z)) * (1 - np.exp(z)/(1 + np.exp(z)))
    elif func == 'ReLU':
        dZZ = (z > 0).astype(int)
    else:
        sys.exit("Unsupported function type!")
    return dZZ

# softmax function
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

# convolution
def conv(A,B):
    d = A.shape[0]
    ky = B.shape[0]
    kx = B.shape[1]
    C = B.shape[2]
    ZZ = np.zeros((d-ky+1,d-kx+1,C))
    for p in range(C):
        for i in range(d-ky+1):
            for j in range(d-kx+1):
                ZZ[i,j,p] = np.sum(np.multiply(B[:,:,p],A[i:i+ky,j:j+kx]))
    return ZZ
                
# forward step
def forward(x, y, model):
    Z = conv(X,model['K'])
    H = sigma(Z)
    U = np.zeros(num_outputs)
    for k in range(num_outputs):
        U[k] = np.sum(np.multiply(model['W'][k,:,:,:],H))
    f = softmax_function(U)
    return (Z, H, f)

# backward step
def backward(x, y, f, Z, H, model, model_grads):
    dU = - 1.0*f
    dU[y] = dU[y] + 1.0
    db = dU
    dW = np.zeros(model['W'].shape)
    for k in range(num_outputs):
        dW[k,:,:,:] = dU[k] * H
    delta = np.zeros((num_inputs_y-kernel_y+1,num_inputs_x-kernel_x+1,num_channels))
    for p in range(num_channels):
        for i in range(num_inputs_y-kernel_y+1):
            for j in range(num_inputs_x-kernel_x+1):
                delta[i,j,p] = np.dot(dU,model['W'][:,i,j,p])
    dZZ = d_sigma(Z)
    dZZ_delta = np.multiply(delta, dZZ)
    dK = conv(X,dZZ_delta)
    model_grads['W'] = dW
    model_grads['K'] = dK
    model_grads['b'] = db      
    return model_grads

time1 = time.time()
# initialize learning rate
LR = .01
num_epochs = 10

for epochs in range(num_epochs):
    
    time01 = time.time()
    
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001     
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
        
    total_correct = 0
    
    for n in range( len(x_train)):
        # randomly select a new data sample
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        X = np.reshape(x,(num_inputs_y,num_inputs_x))
        
        # forward step
        (Z, H, f) = forward(X, y, model)
        
        # check the prediction accuracy
        prediction = np.argmax(f)
        if (prediction == y):
            total_correct += 1
        
        # backward step
        model_grads = backward(X, y, f, Z, H, model, model_grads)
        
        # update parameters
        model['W'] = model['W'] + LR*model_grads['W']
        model['K'] = model['K'] + LR*model_grads['K']
        model['b'] = model['b'] + LR*model_grads['b']
        
    print("Epoch # %3d,  Accuracy %8.4f" % ( epochs, total_correct/np.float(len(x_train) ) ) )
    time02 = time.time()
    print("Training Time for This Epoch: %8.4f (s)" % ( time02-time01 ) )

time2 = time.time()
print("Total Training Time for %3d Epochs: %8.4f (s)" % ( num_epochs, time2-time1 ) )

######################################################
# test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    X = np.reshape(x,(num_inputs_y,num_inputs_x))
    (_, _, f) = forward(x, y, model)
    prediction = np.argmax(f)
    if (prediction == y):
        total_correct += 1
print("Test Accuracy %8.4f" % ( total_correct/np.float(len(x_test) ) ) )