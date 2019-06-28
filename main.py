"""
Pipeline for training and evaluating a convolutional neural network on the MNIST dataset

@author: Haoyang (Ryan) Li
"""

import time
import cnn_utils

def main():

    # parse arguments/hyperparameters
    params = cnn_utils.parse_params()
    
    # load the MNIST dataset
    mnist_dir = './MNISTdata.hdf5'
    mnist = cnn_utils.load_mnist(mnist_dir)
    
    # initialization
    (model, model_grads) = cnn_utils.init_model(mnist,params)
    
    # training the model
    print("\nStart training")
    print("---------------")
    time_start = time.time()
    model = cnn_utils.cnn_train(model, model_grads, params, mnist)
    time_end = time.time()
    print("Training Time : %8.4f (s)" % ( time_end-time_start ) )
    
    # testing the model
    print("\nStart testing")
    print("--------------")
    cnn_utils.cnn_test(model, params, mnist)
    
if __name__ == "__main__":
    main()