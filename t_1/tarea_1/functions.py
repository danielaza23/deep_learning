import torch
import tensorflow as tf
import numpy as np
pi = tf.constant(np.pi, dtype=tf.float32)


def torch_fun_1(x):
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Input must be a pytorch tensor but is {type(x)}")
    if (2,) != x.shape:
        raise ValueError(f"Input must be of shape (2, ) but has shape {x.shape}")
    
    # 1. Múltiples mínimos locales (similar a Rastrigin)
    A = 10.0
    # Se escala x y se pasa por tanh para crear regiones planas
    y = torch.tanh(x) * 5.0
    rastrigin_term = 2 * A + torch.sum(y**2 - A * torch.cos(2 * torch.pi * y))

    # 2. Discontinuidad y valles no lineales
    # torch.floor tiene gradiente cero en casi todas partes.
    discontinuous_term = torch.sum(torch.abs(x * torch.floor(x)))

    # 3. Curvatura patológica (valle estrecho y rotado)
    skew_term = (3.0 * x[0] + x[1])**2 * 0.1

    return rastrigin_term + discontinuous_term + skew_term


def torch_fun_2(x):
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Input must be a pytorch tensor but is {type(x)}")
    if (10,) != x.shape:
        raise ValueError(f"Input must be of shape (10, ) but has shape {x.shape}")
    
    # 1. Múltiples mínimos locales (Rastrigin)
    A = 10.0
    d = x.shape[0]
    rastrigin_term = A * d + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x))

    # 2. Curvatura patológica (Rosenbrock)
    rosen_term = torch.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    # 3. Discontinuidades con un factor aleatorio (pero fijo)
    g = torch.Generator()
    g.manual_seed(42)
    random_multipliers = torch.randint(-3, 3, size=x.shape, generator=g, dtype=x.dtype)
    # Se evita el multiplicador 0 para no anular el término
    random_multipliers[random_multipliers == 0] = -1
    discontinuous_term = torch.sum((torch.floor(x * random_multipliers))**2)

    # Se combinan los términos para una mayor complejidad
    return rastrigin_term + rosen_term * 0.1 + discontinuous_term


def tf_fun_1(x):
    """
    Simula un paisaje de pérdida difícil con múltiples mínimos locales y 
    grandes mesetas de gradiente cero que pueden "atrapar" al optimizador.
    """
    if not isinstance(x, tf.Variable):
        raise ValueError(f"La entrada debe ser una tf.Variable, pero es {type(x)}")
    if (2,) != x.shape:
        raise ValueError(f"La entrada debe tener forma (2,), pero tiene {x.shape}")

    # --- Componentes de la función de pérdida ---

    # 1. Múltiples mínimos locales (similar a Rastrigin)
    # Esto crea una superficie "ondulada" que puede hacer que el optimizador salte
    # erráticamente al principio, como se ve en los primeros pasos del log.
    A = 10.0
    # Se escala 'y' para controlar la frecuencia de las ondulaciones
    y = tf.math.tanh(x) * 5.0 
    rastrigin_term = 2 * A + tf.reduce_sum(y**2 - A * tf.math.cos(2 * pi * y))

    # 2. Discontinuidad y mesetas (la clave para el "atasco")
    # tf.math.floor() tiene un gradiente de CERO en casi todas partes. Si el 
    # optimizador cae en una región donde el resultado de floor(x) es constante,
    # el gradiente se anula y el optimizador deja de moverse. Esto simula
    # perfectamente el atasco en la pérdida de ~288.
    discontinuous_term = tf.reduce_sum(tf.math.abs(x * tf.math.floor(x))) * 5.0

    # 3. Curvatura patológica (valle estrecho y rotado)
    # Esto simplemente hace que el problema sea más difícil de navegar antes de 
    # quedar atrapado.
    skew_term = (3.0 * x[0] + x[1])**2 * 0.1

    return rastrigin_term + discontinuous_term + skew_term


def tf_fun_2(x):
    """
    Simula un paisaje de pérdida complejo, dominado por un valle estrecho 
    (Rosenbrock) con mínimos locales más pequeños (Rastrigin). Es difícil 
    pero converge de manera más estable.
    """
    if not isinstance(x, tf.Variable):
        raise ValueError(f"La entrada debe ser una tf.Variable, pero es {type(x)}")
    if (10,) != x.shape:
        raise ValueError(f"La entrada debe tener forma (10,), pero tiene {x.shape}")

    # --- Componentes de la función de pérdida ---

    # 1. Múltiples mínimos locales (Rastrigin)
    # Añade complejidad y pequeñas fluctuaciones al viaje del optimizador.
    A = 10.0
    d = x.shape[0]
    rastrigin_term = A * float(d) + tf.reduce_sum(x**2 - A * tf.math.cos(2 * pi * x))

    # 2. Curvatura patológica (Rosenbrock - la clave para la convergencia estable)
    # La función de Rosenbrock crea un valle largo, estrecho y parabólico. Los 
    # optimizadores pueden progresar de manera constante a lo largo de este valle,
    # lo que explica la mejora continua y gradual de "Mejor hasta ahora" en el log.
    rosen_term = tf.reduce_sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

    # 3. Discontinuidades suaves
    # A diferencia de la función 1, estas discontinuidades son menos severas
    # y no crean trampas tan efectivas, solo añaden un poco más de "ruido"
    # al gradiente.
    g = tf.random.Generator.from_seed(42)
    random_multipliers = g.uniform(shape=x.shape, minval=-3, maxval=3, dtype=tf.float32)
    discontinuous_term = tf.reduce_sum(tf.math.floor(x * random_multipliers)**2)
    
    # Se combinan los términos. El factor 0.1 en rosen_term equilibra su influencia.
    return rastrigin_term + rosen_term * 0.1 + discontinuous_term