from keras import models, layers

def get_convolutional_layers(model):
    """
    Returns a list of all convolutional layers in the model.

    Args:
        model (keras.model): A keras model

    Returns:
        List of convolutional layers in `model`
    """
    pass

def get_filters(model):
    """
    Returns a list of all filters in the model.

    Args:
        model (keras.model): A keras model

    Returns:
        List of convolutional filters in `model`
    """
    pass

def get_kernel(model, num_conv_layer, input_channel, output_channel):
    """
    Returns the weight matrix belonging to the kernel that connects a specific input and output channel..

    Args:
        model (keras.model): A keras model

    Returns:
        List of convolutional filters in `model`
    """
    pass

def get_convolutional_activations(model, x):
    """
    Returns a list of the activation values of the keras model `model` after each convolutional layer on instance `x`.
    There is one entry in the output list for each convolutional layer, and the shape is according to the feature map shape (convolution shape)

    Args:
        model (keras.model): A keras model
        x (np.ndarray): A 2D numpy array for the instance on which the activations are desired.
    """
    pass