import torch
import tensorflow as tf
import numpy as np


def optimize_torch_fun1(f):
    print("Testing Torch 1")
    """
    Compite con varios optimizadores para encontrar el arg min f y devuelve
    el tensor del optimizador más rápido.
    """
    # Parámetros de la competencia interna
    tolerance = 1e-4
    max_steps = 1000 # Aumentamos los pasos para dar oportunidad a los más lentos
    
    # Estructura para guardar el resultado de cada optimizador
    results = {}
    
    learning_rates = {"SGD": 0.01, "Momentum": 0.01, "Adagrad": 0.1, "RMSprop": 0.01, "Adam": 0.1}
    optimizers_to_test = ["SGD", "Momentum", "Adagrad", "RMSprop", "Adam"]

    print("--- Iniciando competencia en optimize_torch_fun1 ---")
    for opt_name in optimizers_to_test:
        # Reiniciamos la variable 'x' para cada optimizador, para una competencia justa
        x = torch.zeros(2, requires_grad=True)
        lr = learning_rates.get(opt_name, 0.1)

        # Seleccionar el optimizador
        if opt_name == "SGD": optimizer = torch.optim.SGD([x], lr=lr)
        elif opt_name == "Momentum": optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
        elif opt_name == "Adagrad": optimizer = torch.optim.Adagrad([x], lr=lr)
        elif opt_name == "RMSprop": optimizer = torch.optim.RMSprop([x], lr=lr)
        elif opt_name == "Adam": optimizer = torch.optim.Adam([x], lr=lr)
        else: continue

        # Bucle de optimización hasta la convergencia
        for step in range(1, max_steps + 1):
            optimizer.zero_grad()
            loss = f(x)
            if loss.item() < tolerance:
                # ¡Importante! Guardamos el número de pasos y una copia del tensor final
                results[opt_name] = {"steps": step, "final_x": x.detach().clone()}
                break
            loss.backward()
            optimizer.step()
        else: # Si el bucle termina sin converger
            results[opt_name] = {"steps": max_steps, "final_x": x.detach().clone()}

    # --- Lógica para decidir el ganador ---
    
    # Filtramos los que sí convergieron
    converged_optimizers = {name: data for name, data in results.items() if data['steps'] < max_steps}
    
    if converged_optimizers:
        # Encontramos el nombre del optimizador más rápido (menos pasos)
        best_optimizer_name = min(converged_optimizers, key=lambda name: converged_optimizers[name]['steps'])
        print(f"-> Ganador de la competencia: {best_optimizer_name} con {results[best_optimizer_name]['steps']} pasos.")
        # Recuperamos el tensor del optimizador más rápido y lo devolvemos
        return results[best_optimizer_name]['final_x']
    else:
        # Fallback: si ninguno convergió, devolvemos el resultado de Adam (suele ser el más robusto)
        print("-> Ningún optimizador convergió, devolviendo el resultado de Adam.")
        return results['Adam']['final_x']

def optimize_torch_fun2(f):
    print("Testing Torch 2")
    """
    Encuentra arg min f utilizando PyTorch.

    Args:
        f: una función que recibe un tensor de PyTorch de forma (10,) y devuelve un float.
    
    Return: un tensor de PyTorch de forma (10,)
    """
        # Parámetros de la competencia interna
    tolerance = 1e-4
    max_steps = 1000 # Aumentamos los pasos para dar oportunidad a los más lentos
    
    # Estructura para guardar el resultado de cada optimizador
    results = {}
    
    learning_rates = {"SGD": 0.01, "Momentum": 0.01, "Adagrad": 0.1, "RMSprop": 0.01, "Adam": 0.1}
    optimizers_to_test = ["SGD", "Momentum", "Adagrad", "RMSprop", "Adam"]

    print("--- Iniciando competencia en optimize_torch_fun2 ---")
    for opt_name in optimizers_to_test:
        # Reiniciamos la variable 'x' para cada optimizador, para una competencia justa
        x = torch.zeros(10, requires_grad=True)
        lr = learning_rates.get(opt_name, 0.1)

        # Seleccionar el optimizador
        if opt_name == "SGD": optimizer = torch.optim.SGD([x], lr=lr)
        elif opt_name == "Momentum": optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
        elif opt_name == "Adagrad": optimizer = torch.optim.Adagrad([x], lr=lr)
        elif opt_name == "RMSprop": optimizer = torch.optim.RMSprop([x], lr=lr)
        elif opt_name == "Adam": optimizer = torch.optim.Adam([x], lr=lr)
        else: continue

        # Bucle de optimización hasta la convergencia
        for step in range(1, max_steps + 1):
            optimizer.zero_grad()
            loss = f(x)
            if loss.item() < tolerance:
                # ¡Importante! Guardamos el número de pasos y una copia del tensor final
                results[opt_name] = {"steps": step, "final_x": x.detach().clone()}
                break
            loss.backward()
            optimizer.step()
        else: # Si el bucle termina sin converger
            results[opt_name] = {"steps": max_steps, "final_x": x.detach().clone()}

    # --- Lógica para decidir el ganador ---
    
    # Filtramos los que sí convergieron
    converged_optimizers = {name: data for name, data in results.items() if data['steps'] < max_steps}
    
    if converged_optimizers:
        # Encontramos el nombre del optimizador más rápido (menos pasos)
        best_optimizer_name = min(converged_optimizers, key=lambda name: converged_optimizers[name]['steps'])
        print(f"-> Ganador de la competencia: {best_optimizer_name} con {results[best_optimizer_name]['steps']} pasos.")
        # Recuperamos el tensor del optimizador más rápido y lo devolvemos
        return results[best_optimizer_name]['final_x']
    else:
        # Fallback: si ninguno convergió, devolvemos el resultado de Adam (suele ser el más robusto)
        print("-> Ningún optimizador convergió, devolviendo el resultado de Adam.")
        return results['Adam']['final_x']


def _optimize_tf(f, shape, tolerance, max_steps):
    """Función auxiliar que ejecuta la competencia en modo Eager."""
    results = {}
    learning_rates = {"SGD": 0.01, "Momentum": 0.01, "Adagrad": 0.1, "RMSprop": 0.01, "Adam": 0.1}
    optimizers_to_test = ["SGD", "Momentum", "Adagrad", "RMSprop", "Adam"]

    print("--- Iniciando competencia en _optimize_tf ---")
    for opt_name in optimizers_to_test:
        x = tf.Variable(tf.zeros(shape), dtype=tf.float32)
        lr = learning_rates.get(opt_name, 0.1)

        if opt_name == "SGD": optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif opt_name == "Momentum": optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif opt_name == "Adagrad": optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif opt_name == "RMSprop": optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif opt_name == "Adam": optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else: continue

        for step in range(1, max_steps + 1):
            with tf.GradientTape() as tape:
                loss = f(x)
            
            if loss < tolerance:
                results[opt_name] = {"steps": step, "final_x": x}
                break
            
            gradients = tape.gradient(loss, [x])
            
            if gradients[0] is not None:
                optimizer.apply_gradients(zip(gradients, [x]))
            else:
                break
        else:
            results[opt_name] = {"steps": max_steps, "final_x": x}

    # ----- ¡CAMBIOS AQUÍ! Se añaden los prints solicitados -----
    converged_optimizers = {name: data for name, data in results.items() if data['steps'] < max_steps}
    if converged_optimizers:
        best_optimizer_name = min(converged_optimizers, key=lambda name: converged_optimizers[name]['steps'])
        print(f"-> Ganador de la competencia: {best_optimizer_name} con {results[best_optimizer_name]['steps']} pasos.")
        return results[best_optimizer_name]['final_x']
    else:
        print("-> Ningún optimizador convergió, devolviendo el resultado de Adam.")
        return results['Adam']['final_x']

def optimize_tf_fun1(f):
    return _optimize_tf(f, shape=(2,), tolerance=1e-4, max_steps=1000)

def optimize_tf_fun2(f):
    return _optimize_tf(f, shape=(10,), tolerance=1e-4, max_steps=1000)
