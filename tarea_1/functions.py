import torch
import tensorflow as tf
import numpy as np
pi = tf.constant(np.pi, dtype=tf.float32)


def torch_fun_1(x):
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Input must be a pytorch tensor but is {type(x)}")
    if (2,) != x.shape:
        raise ValueError(f"Input must be of shape (2, ) but has shape {x.shape}")
    
    g = torch.Generator()
    g.manual_seed(42)
    y = x + torch.randint(4, 20, size=(2, ), generator=g)
    return sum(y**2)

def torch_fun_2(x):
    g = torch.Generator()
    g.manual_seed(42)
    y = x + torch.randint(4, 20, size=(10, ), generator=g)
    return sum(y**2)

def tf_fun_1(x):
    if not isinstance(x, tf.Variable):
        raise ValueError(f"Input must be a tensorflow Variable but is {type(x)}")
    if (2,) != x.shape:
        raise ValueError(f"Input must be of shape (2, ) but has shape {x.shape}")
    g = tf.random.Generator.from_seed(42)
    y = x + g.uniform(shape=(2,), minval=4, maxval=20, dtype=tf.float32)
    return sum(y**2)

def tf_fun_2(x):
    if not isinstance(x, tf.Variable):
        raise ValueError(f"Input must be a tensorflow Variable but is {type(x)}")
    if (10,) != x.shape:
        raise ValueError(f"Input must be of shape (10, ) but has shape {x.shape}")
    
    g = tf.random.Generator.from_seed(42)
    y = x + g.uniform(shape=(10,), minval=4, maxval=20, dtype=tf.float32)
    return sum(y**2)
