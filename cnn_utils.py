"""
Functions for training and evaluating a convolutional neural network on the MNIST dataset

@author: Haoyang (Ryan) Li
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import copy
from random import randint
import sys

def load_mnist(mnist_dir):
    """
    Load the MNIST dataset
    
    Parameters
    ----------
    func : str
        directory of the MNIST data
        
    Returns
    -------
    mnist : dict
        a dictionary containing the training and test data as well as data 
        sizes and shapes
    """
    MNIST_data = h5py.File(mnist_dir, 'r')
    mnist = {}
    mnist['x_train'] = np.float32( MNIST_data['x_train'][:] )
    mnist['y_train'] = np.int32( np.array( MNIST_data['y_train'][:,0] ) )
    mnist['x_test'] = np.float32( MNIST_data['x_test'][:] )
    mnist['y_test'] = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
    MNIST_data.close()
    
    # TODO: pre-specify MNIST data info; read the data; consistency check
    mnist['n_train'] = mnist['x_train'].shape[0] # 60000
    mnist['n_test'] = mnist['x_test'].shape[0] # 10000
    mnist['n_input'] = mnist['x_train'].shape[1] # image size 28*28=784
    mnist['input_x'] = 28
    mnist['input_y'] = 28
    mnist['n_output'] = len(np.unique(mnist['y_test'])) # num of labels = 10
    
    # print data info
    print("\nMNIST data info")
    print("----------------")
    print("Number of training data : %d" % mnist['n_train'])
    print("Number of test data : %d"  % mnist['n_test'])
    print("Input data shape : %d x %d = %d" % 
          (mnist['input_x'], mnist['input_y'], mnist['n_input']))
    print("Output data shape : %d" % mnist['n_output'])
    
    return mnist

def parse_params():
    """
    Parse the arguments/hyperparameters
    
    Parameters
    ----------
    None
        
    Returns
    -------
    params : argparse.Namespace
        hyperparameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--decay', type=float, default=0.1, 
                        help='learning rate decay (default: 0.1)')
    parser.add_argument('--interval', type=int, default=5, 
                        help='staircase interval for learning rate decay (default: 5')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--k_x', type=int, default=3,
                        help='x-dimension of the kernel (default: 3)')
    parser.add_argument('--k_y', type=int, default=3,
                        help='y-dimension of the kernel (default: 3)')
    parser.add_argument('--n_ch', type=int, default=3,
                        help='number of channels (default: 3)')
    parser.add_argument('--sigma', type=str, default='relu',
                        help='type of activation function (default: relu)')
    parser.add_argument('--quicktest', type=bool, default=True,
                        help='whether or not to perform a quick test of the \
                        pipeline (default: False)')
    params = parser.parse_args()
    
    # modify parameters for a quick test
    if (params.quicktest == True):
        params.n_epochs = 1
        print("\nPerforming a quick test...")
        print("----------------------------")
    
    # print hyperparameters for training
    print("\nHyperparameters")
    print("-----------------")
    print("Initial learning rate : %6.4f" % params.lr)
    print("Learning rate decay : %6.4f" % params.decay)
    print("Staircase learning rate decay interval : %d" % params.interval)
    print("Number of epochs : %d" % params.n_epochs)
    print("Kernal size : %d x %d" % (params.k_x, params.k_y))
    print("Number of channels : %d" % params.n_ch)
    print("Activation function : %s" % params.sigma)

    return params

def init_model(mnist, params):
    """
    Initialize neural network model
    
    Parameters
    ----------
    mnist : dict
        contains mnist training and test data
    params : argparse.Namespace
        comtains hyperparameters for training
        
    Returns
    -------
    model : dict
        parameters/weights of the nerual network
    model_grads : dict
        gradients of the parameters/weights of the nerual network
    """
    # TODO: pre-specify dimensions with shorter variable names
    model = {}
    model['W'] = np.random.randn(mnist['n_output'],
                                 mnist['input_y']-params.k_y+1,
                                 mnist['input_x']-params.k_x+1,
                                 params.n_ch) / np.sqrt(mnist['n_input'])
    model['K'] = np.random.randn(params.k_y,
                                 params.k_x,
                                 params.n_ch) / np.sqrt(params.k_y*params.k_x)
    model['b'] = np.random.randn(mnist['n_output']) / np.sqrt(mnist['n_output'])
    model_grads = copy.deepcopy(model)
    return (model, model_grads)

def sigma(z, func):
    """
    Activation functions
    
    Parameters
    ----------
    z : ndarray of float
        input
    func : str
        the type of activation adopted
        
    Returns
    -------
    ZZ : ndarray of float
        output
    """
    if func == 'tanh':
        ZZ = np.tanh(z)
    elif func == 'sigmoid':
        ZZ = np.exp(z)/(1 + np.exp(z))
    elif func == 'relu':
        ZZ = np.maximum(z,0)
    else:
        sys.exit("Unsupported function type!")
    return ZZ

def d_sigma(z, func):
    """
    Derivative of activation functions
    
    Parameters
    ----------
    z : ndarray of float
        input
    func : str
        the type of activation
        
    Returns
    -------
    dZZ : ndarray of float
        output
    """
    if func == 'tanh':
        dZZ = 1.0 - np.tanh(z)**2
    elif func == 'sigmoid':
        dZZ = np.exp(z)/(1 + np.exp(z)) * (1 - np.exp(z)/(1 + np.exp(z)))
    elif func == 'relu':
        dZZ = (z > 0).astype(int)
    else:
        sys.exit("Unsupported function type!")
    return dZZ

def softmax_function(z):
    """
    Softmax function
    
    Parameters
    ----------
    z : ndarray of float
        input
        
    Returns
    -------
    ZZ : ndarray of float
        output
    """
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def conv(A,B):
    """
    2D convolution with multiple channels
    
    Parameters
    ----------
    A : ndarray of float
        input
        shape = x_dim_of_input * y_dim_of_input
        
    B : ndarray of float
        convolution kernel
        shape = x_dim_of_kernel * x_dim_of_kernel * n_channels
        
    Returns
    -------
    ZZ : ndarray of float
        output
    """
    # TODO: optimize, avoid using loops
    dy = A.shape[0] # y-dim of input
    dx = A.shape[1] # x-dim of input
    ky = B.shape[0] # y-dim of kernel
    kx = B.shape[1] # x-dim of kernel
    C = B.shape[2]  # number of channels
    ZZ = np.zeros((dy-ky+1,dx-kx+1,C))
    for p in range(C):
        for i in range(dy-ky+1):
            for j in range(dx-kx+1):
                ZZ[i,j,p] = np.sum(np.multiply(B[:,:,p],A[i:i+ky,j:j+kx]))
    return ZZ

def forward(X, model, func):
    """
    Forward propagation of the neural network
    
    Parameters
    ----------
    x : ndarray of float
        input
    model : dict
        parameters/weights of the nerual network
    func : str
        the type of activation
        
    Returns
    -------
    Z : ndarray of float
        output of the linear layer
    H : ndarray of float
        output after the activation
    f : ndarray of float
        output of the forward propagation
    """
    # TODO: optimize, avoid using loops
    Z = conv(X,model['K'])
    H = sigma(Z,func)
    U = np.zeros(model['W'].shape[0])
    for k in range(len(U)):
        U[k] = np.sum(np.multiply(model['W'][k,:,:,:],H))
    f = softmax_function(U)
    return (Z, H, f)

def backprop(X, y, f, Z, H, model, model_grads, func):
    """
    Backpropagation of the neural network
    
    Parameters
    ----------
    x : ndarray of float
        input
    y : ndarray of int
        ground truth label
    f : ndarray of float
        output of the forward propagation
    Z : ndarray of float
        output of the linear layer
    H : ndarray of float
        output after the activation
    model : dict
        parameters/weights of the nerual network
    model_grads : dict
        gradients of the parameters/weights of the nerual network
    func : str
        the type of activation
        
    Returns
    -------
    model_grads : dict
        updated gradients of the parameters/weights of the nerual network
    """
    # TODO: optimize, avoid using loops
    dU = - 1.0*f
    dU[y] = dU[y] + 1.0
    db = dU
    dW = np.zeros(model['W'].shape)
    for k in range(model['W'].shape[0]):
        dW[k,:,:,:] = dU[k] * H
    delta = np.zeros(model['W'].shape[1:])
    for p in range(model['W'].shape[-1]):
        for i in range(model['W'].shape[1]):
            for j in range(model['W'].shape[2]):
                delta[i,j,p] = np.dot(dU,model['W'][:,i,j,p])
    dZZ = d_sigma(Z,func)
    dZZ_delta = np.multiply(delta, dZZ)
    dK = conv(X,dZZ_delta)
    model_grads['W'] = dW
    model_grads['K'] = dK
    model_grads['b'] = db      
    return model_grads

def plot_predict(x, y, pred):
    """
    Plot and display the test figure and prediction
    
    Parameters
    ----------
    x : ndarray of float
        input
    y : ndarray of int
        ground truth label
    pred : ndarray of int
        predicted label
        
    Returns
    -------
    None
    """
    plt.figure(figsize=(3,3))
    x = x.reshape(28,28) # MNIST image input size: 28*28
    plt.gray()
    plt.axis('off')
    plt.title("Truth: %d    Predict: %d" % (y, pred))
    plt.imshow(x)
    plt.show()
    

def cnn_train(model, model_grads, params, mnist):
    """
    Training the model with stochastic gradient descent
    
    Parameters
    ----------
    model : dict
        parameters/weights of the nerual network
    model_grads : dict
        gradients of the parameters/weights of the nerual network
    params : argparse.Namespace
        comtains hyperparameters for training
    mnist : dict
        contains mnist training and test data
        
    Returns
    -------
    model : dict
        updated parameters/weights of the nerual network
    """
    # initial learning rate
    LR = params.lr
    
    for epochs in range(params.n_epochs):
        
        # learning rate schedule: staircase decay
        if (epochs > 0 and epochs % params.interval == 0):
            LR *= params.decay
            
        total_correct = 0
        
        if (params.quicktest == False):
            n_samples = mnist['n_train']
        else:
            n_samples = int(mnist['n_train'] / 100)
        
        for n in range( n_samples ):
            
            # randomly select a new data sample
            n_random = randint(0,mnist['n_train']-1 )
            y = mnist['y_train'][n_random]
            x = mnist['x_train'][n_random][:]
            X = np.reshape(x,(mnist['input_y'],mnist['input_x']))
            
            # forward step
            (Z, H, f) = forward(X, model, params.sigma)
            
            # check the prediction accuracy
            prediction = np.argmax(f)
            if (prediction == y):
                total_correct += 1
            
            # backpropagation step
            model_grads = backprop(X, y, f, Z, H, model, model_grads, params.sigma)
            
            # update parameters
            model['W'] = model['W'] + LR*model_grads['W']
            model['K'] = model['K'] + LR*model_grads['K']
            model['b'] = model['b'] + LR*model_grads['b']
            
        print("Epoch %3d,  Accuracy %6.4f" % 
              ( epochs, total_correct/np.float(mnist['n_train'] ) ) )
        
    return model

def cnn_test(model, params, mnist):
    """
    Testing the model
    
    Parameters
    ----------
    model : dict
        parameters/weights of the nerual network
    params : argparse.Namespace
        comtains hyperparameters for training
    mnist : dict
        contains mnist training and test data
        
    Returns
    -------
    None
    """
    total_correct = 0
    count_correct = 0
    count_wrong = 0
    k = 5
    
    if (params.quicktest == False):
        n_samples = mnist['n_test']
    else:
        n_samples = int(mnist['n_test'] / 100)
    
    for n in range( n_samples):
        
        # load test data sample
        y = mnist['y_test'][n]
        x = mnist['x_test'][n][:]
        X = np.reshape(x,(mnist['input_y'],mnist['input_x']))
        
        # forward step and prediction
        (_, _, f) = forward(X, model, params.sigma)
        prediction = np.argmax(f)
        
        # check prediction accuracy
        if (prediction == y):
            total_correct += 1
            # display the first k correct predictions
            if (count_correct < k and params.quicktest == False):
                plot_predict(x, y, prediction)
                count_correct += 1
        
        # display the first k incorrect predictions
        if (prediction != y and count_wrong < k and params.quicktest == False):
            plot_predict(x, y, prediction)
            count_wrong += 1
            
    print("Test Accuracy : %6.4f" % 
          ( total_correct/np.float(mnist['n_test']) ) )
