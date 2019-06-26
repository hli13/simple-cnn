# A Convolutional Neural Network with One Hidden Layer

In this project, a convolutional neural network (CNN) with one hidden layer is implemented from scratch in Python. The model is trained using stocastic gradient descent (SGD) and evaluated on the MNIST dataset.

## Dependencies

```
numpy==1.16.4
h5py==2.9.0
matplotlib==3.1.0
```

## Dataset

The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset is used to train and evaluate the neural network model in this project. It is a database of handwritten digits that is commonly used to train image processing models. The dataset in hdf5 format is included in the repository.

## Implementation

#### Activation functions

Rectified Linear Unit (ReLU) is defined as the activation function in this project. The evaluation of the function itself and its derivative are implemented as follow.

```python
TBD
```

#### Softmax function

The softmax function is applied in the output layer of the neural network.
```python
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
```

#### Forward propagation

```python
TBD
```

#### Backpropagation

```python
TBD
```

## Hyerparameters

TBD

## Running the model

TBD

## Result

TBD
