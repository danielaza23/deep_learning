import torch
import tensorflow as tf
import numpy as np
from tensorflow import keras

from time import time

# from pathlib import Path

# def create_animation(loss_history, title, output_filename, frame_skip=10):
#     """
#     Crea una animación GIF a partir de un historial de valores de pérdida.
#     """
#     print(f"\n[INFO] Creando animación para '{title}'...")
#     filenames = []
    
#     # Limpiar la carpeta de frames para cada nueva animación
#     if os.path.exists("frames"):
#         shutil.rmtree("frames")
#     os.makedirs("frames")

#     print(loss_history)

#     for i in range(len(loss_history)):
#         if i % frame_skip != 0 and i != len(loss_history) - 1:
#             continue

#         plt.style.use('seaborn-v0_8-whitegrid')
#         fig, ax = plt.subplots(figsize=(10, 6), dpi=90)
        
#         ax.plot(loss_history[:i+1], color='#0077b6', linewidth=2.5)
#         ax.set_title(f"{title}\nPaso de Optimización: {i}", fontsize=16)
#         ax.set_xlabel("Paso de Optimización", fontsize=12)
#         ax.set_ylabel("Valor de la Función Objetivo (Loss)", fontsize=12)
        
#         if np.max(loss_history) / (np.min(loss_history) + 1e-9) > 100:
#             ax.set_yscale('log')

#         ax.set_xlim(0, len(loss_history))
#         ax.set_ylim(min(loss_history) * 0.9, max(loss_history) * 1.1)

#         filename = f"frames/frame_{len(filenames):04d}.png"
#         filenames.append(filename)
#         plt.savefig(filename)
#         plt.close(fig)

#     gif_path = f"{output_filename}.gif"
#     with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
#         for filename in filenames:
#             image = imageio.imread(filename)
#             writer.append_data(image)

#     print(f"[SUCCESS] Animación guardada como: {gif_path}")


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)
OPTIMIZATION_STEPS = 30000
timeout = 59

def optimize_torch_fun1(f):
    print("------Testing Torch 1-----")
    t_start = time()
    x = torch.randn((2,), requires_grad=True, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.Adam([x], lr=0.8)
    loss_history = []
    for _ in range(OPTIMIZATION_STEPS):
        if time() - t_start > timeout: break
        optimizer.zero_grad()
        loss = f(x)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
    # create_animation(loss_history, "PyTorch Function 1", "pytorch_fun_1_optimization")
    return x

def optimize_torch_fun2(f):
    print("------Testing Torch 2-----")
    t_start = time()
    x = torch.full((10,), 42.0, requires_grad=True)
    x.data.add_(0.01 * torch.randn_like(x))

    optimizer = torch.optim.Adam([x], lr=0.1)
    loss_history = []
    best_x = x.clone()
    best_loss = float('inf')

    for step in range(OPTIMIZATION_STEPS):
        if time() - t_start > timeout:
            break
        optimizer.zero_grad()
        loss = f(x)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
    # create_animation(loss_history, "PyTorch Function 2", "pytorch_fun_2_optimization")
        if not torch.isfinite(loss): 
            print("  [WARN] Loss no finita, deteniendo optimización.")
            break
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = x.detach().clone()
        if step == 1 or step % 100 == 0:
            print(f"    Paso Principal {step:5d}: Loss actual = {loss.item():.6f} (Mejor hasta ahora: {best_loss:.6f})")
        if best_loss < 0.001:
            print(f"  [INFO] Umbral de loss < 0.01 alcanzado en el paso {step}.")
            best_x = x.detach().clone()
            break
    return best_x

def optimize_tf_fun1(f):
    # ESTRATEGIA: Se vuelve a SGD, que empíricamente funcionó mejor para esta función específica.
    shape = (2,)
    tolerance = 0.1
    max_steps = 100000
    learning_rate = 0.1
    timeout = 59
    
    x = tf.Variable(tf.random.normal(shape), dtype=tf.float32)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    t_start = time()
    print("--- Testing optimize_tf_fun1 ---")

    loss = None
    for step in range(1, max_steps + 1):
        if time() - t_start > timeout:
            break
        if time() - t_start > timeout: 
            print("  [INFO] Timeout alcanzado.")
            break
        with tf.GradientTape() as tape:
            loss = f(x)
            
        if loss < tolerance:
            print(f"  [INFO] Convergencia alcanzada en el paso {step}.")
            break
            
        gradients = tape.gradient(loss, [x])
        
        if gradients[0] is not None:
            # ¡NUEVO! Imprimir la norma del gradiente después de calcularlo
            if step == 1 or step % 500 == 0:
                grad_norm = tf.norm(gradients[0]).numpy()
                print(f"    Paso {step:4d}: Pérdida = {loss.numpy():.6f}, Norma Grad. = {grad_norm:.6f}")
            optimizer.apply_gradients(zip(gradients, [x]))
        else:
            print("  [WARN] Gradientes nulos, deteniendo la optimización.")
            break
            
    else:
        print(f"  [INFO] Se alcanzó el número máximo de pasos ({max_steps}).")

    if loss is not None:
        print(f"  [FINAL] Pérdida final: {loss.numpy():.6f} (después de {step} pasos)\n")
        
    return x





def optimize_tf_fun2(f):
    # ESTRATEGIA: Scouting Informado con el sorprendente ganador (SGD).
    print("--- Testing optimize_tf_fun2 with 4-Point Scouting ---")
    shape = (10,)
    max_steps = 100000
    lr = 0.9 
    timeout = 59
    scouting_steps = 100
    
    base_x = tf.constant([0., -1.01, -1.98, -2.98, -4.01, -4.97, -5.83, -6.97, -8.20, -8.97])
    even_mask = tf.constant([-1. if i%2==0 else 1. for i in range(shape[0])])
    start_points = {
        "Pares Opuestos": base_x * even_mask
    }
    t_start=time()
    print("  [INFO] Fase de Scouting...")
    scouted_results = {}
    for name, start_point in start_points.items():
        x_scout = tf.Variable(start_point)
        optimizer_scout = tf.keras.optimizers.SGD(learning_rate=lr)
        loss_scout = tf.constant(float('inf'))
        for _ in range(scouting_steps):
            with tf.GradientTape() as tape: loss_scout = f(x_scout)
            if not tf.math.is_finite(loss_scout): break
            grads = tape.gradient(loss_scout, [x_scout])
            if grads[0] is not None: optimizer_scout.apply_gradients(zip(grads, [x_scout]))
            
        if tf.math.is_finite(loss_scout): scouted_results[name] = loss_scout.numpy()

    best_start_name = min(scouted_results, key=scouted_results.get)
    x = tf.Variable(start_points[best_start_name])
    print(f"  [INFO] Punto de partida elegido: '{best_start_name}' (Loss Scouting: {scouted_results[best_start_name]:.4f})")
        
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    best_loss = float('inf')
    best_x = x
    t_start = time()
    final_step = 0

    for step in range(1, max_steps + 1):
        final_step = step
        if time() - t_start > timeout: break
        with tf.GradientTape() as tape: loss = f(x)
            
        if step == 1 or step % 500 == 0:
            print(f"    Paso {step:4d}: Pérdida = {loss.numpy():.6f}")

        if not tf.math.is_finite(loss): break
        if loss.numpy() < best_loss:
            best_loss = loss.numpy()
            best_x = tf.Variable(x)
        if best_loss < 0.01: 
            print("  [INFO] ¡Convergencia final alcanzada!")
            break
        grads = tape.gradient(loss, [x])
        if grads[0] is not None: optimizer.apply_gradients(zip(grads, [x]))
    
    print(f"  [FINAL] Mejor pérdida encontrada: {best_loss:.6f} (después de {final_step} pasos)")
    print(f"  [SOLUCIÓN] x = {best_x.numpy().round(4)}\n")
    return best_x