import torch
import tensorflow as tf


def optimize_torch_fun1(f):
    """
    Finds arg min f

    Args:
        f: a function with torch operators only that receives a torch tensor of shape (2, ) and will evalue to a float

    Return: torch tensor of shape (2, )
    """
    pass

def optimize_torch_fun2(f):
    """
    Finds arg min f

    Args:
        f: a function with torch operators only that receives a torch tensor of shape (10, ) and will evalue to a float
    
    Return: torch tensor of shape (10, )
    """
    pass

def optimize_tf_fun1(f):
    """
    Finds arg min f

    Args:
        f: a function with tensorflow operators only that receives a tensorflow Variable of shape (2, ) and will evalue to a float
    
    Return: tensorflow Variable of shape (2, )
    """
    pass

def optimize_tf_fun2(f):
    """
    Finds arg min f

    Args:
        f: a function with tensorflow operators only that receives a tensorflow Variable of shape (10, ) and will evalue to a float
    
    Return: tensorflow Variable of shape (10, )
    """
    pass
