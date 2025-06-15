from time import time

def make_model(framework, seed):
    """
    Returns an untrained neural network for the given framework, able to be trained on CIFAR10 in the flattened layout (3072 = 32 * 32 * 3 columns and 10 classes)

    Args:
        framework (str): sklearn, keras, or torch
        seed (int): random seed that controls the initial weights of the network
    """
    pass


def train_network(network, X, y, optimizer, learning_rate, batch_size, timeout):
    """
    Trains the given network on the data X, y using the optimizer `optimizer` with learning rate `learning_rate` and using a batch_size of `batch_size`.
    Training must be terminated within `timeout` seconds.

    Args:
        network: MLPClassifier, keras.Sequential, or torch.nn.Sequential
        X: 2D numpy array
        y: 1D numpy array
        optimizer (str): 'adam' or 'sgd'
        learning_rate (float): the learning rate to be used with the optimizer
        batch_size (int): number of instances to be processed in a forward pass
        timeout (int): time available for training (in seconds)
    """
    pass
