import torch
import tensorflow as tf
import numpy as np
from time import time

# Se fija una semilla global para la reproducibilidad
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)


def optimize_torch_fun1(f):
    max_steps = 2000 
    results = {}
    learning_rates = {"Adagrad": 0.9, "RMSprop": 0.9, "Adam": 0.}
    optimizers_to_test = ["Adagrad", "RMSprop", "Adam"]

    print("--- Iniciando competencia en optimize_torch_fun1 (buscando menor pérdida) ---")
    for opt_name in optimizers_to_test:
        print(f"  [INFO] Probando Optimizador: {opt_name}, LR: {learning_rates.get(opt_name, 0.1)}")
        x = torch.zeros(2, requires_grad=True)
        lr = learning_rates.get(opt_name, 0.1)

        best_loss_for_opt = float('inf')
        best_x_for_opt = None

        if opt_name == "Adagrad": optimizer = torch.optim.Adagrad([x], lr=lr)
        elif opt_name == "RMSprop": optimizer = torch.optim.RMSprop([x], lr=lr)
        elif opt_name == "Adam": optimizer = torch.optim.Adam([x], lr=lr)
        else: continue

        for step in range(1, max_steps + 1):
            optimizer.zero_grad()
            loss = f(x)
            
            # AÑADIDO: Print de progreso periódico
            if step == 1 or step % 200 == 0:
                print(f"    (Progreso {opt_name}) Paso {step:4d}: Pérdida = {loss.item():.6f}")
            
            if loss.item() < best_loss_for_opt:
                best_loss_for_opt = loss.item()
                best_x_for_opt = x.detach().clone()
            
            if not torch.isfinite(loss): break
            loss.backward()
            optimizer.step()
        
        results[opt_name] = {"best_loss": best_loss_for_opt, "best_x": best_x_for_opt}
        print(f"  [FINAL] Mejor pérdida para {opt_name}: {best_loss_for_opt:.6f}\n")

    best_overall_name = min(results, key=lambda name: results[name]['best_loss'])
    best_x_solution = results[best_overall_name]['best_x']

    print(f"-> Ganador de la competencia: {best_overall_name} con una pérdida de {results[best_overall_name]['best_loss']:.6f}.")
    # AÑADIDO: Imprimir el mejor vector 'x' encontrado
    print(f"  [SOLUCIÓN] x = {best_x_solution.detach().numpy().round(4)}\n")
    
    return best_x_solution


def optimize_torch_fun2(f):
    max_steps = 2000
    results = {}
    learning_rates = {"Adagrad": 0.9, "RMSprop": 0.9, "Adam": 0.9}
    optimizers_to_test = ["Adagrad", "RMSprop", "Adam"]

    print("--- Iniciando competencia en optimize_torch_fun2 (buscando menor pérdida) ---")
    for opt_name in optimizers_to_test:
        print(f"  [INFO] Probando Optimizador: {opt_name}, LR: {learning_rates.get(opt_name, 0.1)}")
        x = torch.zeros(10, requires_grad=True)
        lr = learning_rates.get(opt_name, 0.1)

        best_loss_for_opt = float('inf')
        best_x_for_opt = None
        
        if opt_name == "SGD": optimizer = torch.optim.SGD([x], lr=lr)
        elif opt_name == "Momentum": optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
        elif opt_name == "Adagrad": optimizer = torch.optim.Adagrad([x], lr=lr)
        elif opt_name == "RMSprop": optimizer = torch.optim.RMSprop([x], lr=lr)
        elif opt_name == "Adam": optimizer = torch.optim.Adam([x], lr=lr)
        else: continue

        for step in range(1, max_steps + 1):
            optimizer.zero_grad()
            loss = f(x)

            # AÑADIDO: Print de progreso periódico
            if step == 1 or step % 200 == 0:
                print(f"    (Progreso {opt_name}) Paso {step:4d}: Pérdida = {loss.item():.6f}")

            if loss.item() < best_loss_for_opt:
                best_loss_for_opt = loss.item()
                best_x_for_opt = x.detach().clone()
            if not torch.isfinite(loss): break
            loss.backward()
            optimizer.step()
        
        results[opt_name] = {"best_loss": best_loss_for_opt, "best_x": best_x_for_opt}
        print(f"  [FINAL] Mejor pérdida para {opt_name}: {best_loss_for_opt:.6f}\n")

    best_overall_name = min(results, key=lambda name: results[name]['best_loss'])
    best_x_solution = results[best_overall_name]['best_x']

    print(f"-> Ganador de la competencia: {best_overall_name} con una pérdida de {results[best_overall_name]['best_loss']:.6f}.")
    # AÑADIDO: Imprimir el mejor vector 'x' encontrado
    print(f"  [SOLUCIÓN] x = {best_x_solution.detach().numpy().round(4)}\n")

    return best_x_solution


def _optimize_tf(f, shape, max_steps):
    results = {}
    learning_rates = {"SGD": 0.9, "Momentum": 0.9, "Adagrad": 0.9, "RMSprop": 0.9, "Adam": 0.9}
    optimizers_to_test = ["SGD", "Momentum", "Adagrad", "RMSprop", "Adam"]

    print(f"--- Iniciando competencia en _optimize_tf para shape {shape} (buscando menor pérdida) ---")
    for opt_name in optimizers_to_test:
        print(f"  [INFO] Probando Optimizador: {opt_name}, LR: {learning_rates.get(opt_name, 0.1)}")
        x = tf.Variable(tf.zeros(shape), dtype=tf.float32)
        lr = learning_rates.get(opt_name, 0.1)

        best_loss_for_opt = float('inf')
        best_x_for_opt = None

        if opt_name == "SGD": optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif opt_name == "Momentum": optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif opt_name == "Adagrad": optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        elif opt_name == "RMSprop": optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif opt_name == "Adam": optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else: continue

        for step in range(1, max_steps + 1):
            with tf.GradientTape() as tape:
                loss = f(x)
            
            if not tf.math.is_finite(loss): break
            
            # AÑADIDO: Print de progreso periódico
            if step == 1 or step % 100 == 0:
                print(f"    (Progreso {opt_name}) Paso {step:4d}: Pérdida = {loss.numpy():.6f}")

            if loss.numpy() < best_loss_for_opt:
                best_loss_for_opt = loss.numpy()
                best_x_for_opt = tf.Variable(x) 
            
            gradients = tape.gradient(loss, [x])
            if gradients[0] is not None:
                optimizer.apply_gradients(zip(gradients, [x]))
            else:
                break
        
        results[opt_name] = {"best_loss": best_loss_for_opt, "best_x": best_x_for_opt}
        print(f"  [FINAL] Mejor pérdida para {opt_name}: {best_loss_for_opt:.6f}\n")

    best_overall_name = min(results, key=lambda name: results[name]['best_loss'])
    best_x_solution = results[best_overall_name]['best_x']

    print(f"-> Ganador de la competencia: {best_overall_name} con una pérdida de {results[best_overall_name]['best_loss']:.6f}.")
    # AÑADIDO: Imprimir el mejor vector 'x' encontrado
    print(f"  [SOLUCIÓN] x = {best_x_solution.numpy().round(4)}\n")

    return best_x_solution


def optimize_tf_fun1(f):
    return _optimize_tf(f, shape=(2,), max_steps=1000)

def optimize_tf_fun2(f):
    return _optimize_tf(f, shape=(10,), max_steps=1000)
