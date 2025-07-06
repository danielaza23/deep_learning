import torch
import torch.nn as nn
import imageio
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class LayerMonitor:
    """
    Clase para monitorear el entrenamiento de un modelo de PyTorch.
    Estructura de clase validada como correcta.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.weights = {}
        self.net_values = {}
        self.activations = {}
        self.losses_train = []
        self.losses_val = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.weights[name] = [module.weight.clone().detach().cpu().numpy()]
            
            elif isinstance(module, nn.ReLU):
                self.net_values[name] = []
                self.activations[name] = []
                def forward_hook(layer_name, module, input, output):
                    if self.model.training:
                        self.net_values[layer_name].append(input[0].clone().detach().cpu().numpy())
                        self.activations[layer_name].append(output.clone().detach().cpu().numpy())
                module.register_forward_hook(partial(forward_hook, name))

    def retrieve_weights_after_optimization_step(self):
        """Recupera los pesos después de un paso de optimización."""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    self.weights[name].append(module.weight.clone().detach().cpu().numpy())

    def receive_losses(self, train_loss: float, val_loss: float):
        """Recibe y almacena las pérdidas."""
        self.losses_train.append(train_loss)
        self.losses_val.append(val_loss)

    def make_movie(self, fig_fun, gif_filename: str, fps: int):
        """Crea la película GIF."""
        frames = []
        num_batches = len(self.losses_train)
        for t in range(num_batches):
            fig = fig_fun(t, self.weights, self.net_values, self.activations, self.losses_train, self.losses_val)
            canvas = FigureCanvas(fig)
            canvas.draw()
            frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (4,))
            frames.append(frame)
            plt.close(fig)
        if frames:
            imageio.mimsave(gif_filename, frames, fps=fps)


def fig_fun(t: int, weights: dict, net_values: dict, activations: dict, train_losses: list, val_losses: list) -> plt.Figure:
    """
    Genera la figura para cada cuadro del GIF.
    Contiene el último ajuste sutil para boxplot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Visualization - Batch {t}', fontsize=16)

    # Panel 1: Curva de Pérdidas
    axs[0, 0].plot(np.arange(t + 1), train_losses[:t+1], label='Train Loss', color='blue')
    axs[0, 0].plot(np.arange(t + 1), val_losses[:t+1], label='Val Loss', color='red')
    axs[0, 0].set_title('Loss vs Batch')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Panel 2: Distribución de Pesos
    if t == 0:
        # Lógica para t=0: UN SOLO boxplot consolidado.
        all_initial_weights = []
        for weight_history in weights.values():
            if weight_history:
                all_initial_weights.extend(weight_history[0].flatten())
        
        if all_initial_weights:

            axs[0, 1].boxplot([all_initial_weights], patch_artist=True)
            axs[0, 1].set_title('Initial Weights Distribution (All Layers)')
            axs[0, 1].set_xticklabels(['All Layers'])

    else:
        # Lógica para t > 0: Un boxplot por capa.
        weight_data_for_plot = []
        weight_labels = []
        index_to_plot = t + 1

        for layer_name, weight_history in weights.items():
            if index_to_plot < len(weight_history):
                weight_data_for_plot.append(weight_history[index_to_plot].flatten())
                weight_labels.append(layer_name)
        
        if weight_data_for_plot:
            axs[0, 1].boxplot(weight_data_for_plot, labels=weight_labels, patch_artist=True)
        axs[0, 1].set_title(f'Weights Distribution (After Batch {t})')
    
    axs[0, 1].set_ylabel('Weight Value')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Paneles 3 y 4: Activaciones
    net_labels = list(net_values.keys())
    net_data = [hist[t].flatten() for hist in net_values.values() if t < len(hist)]
    if net_data:
        axs[1, 0].boxplot(net_data, labels=net_labels, patch_artist=True)
    axs[1, 0].set_title(f'Net Values (for Batch {t})')
    
    act_labels = list(activations.keys())
    act_data = [hist[t].flatten() for hist in activations.values() if t < len(hist)]
    if act_data:
        axs[1, 1].boxplot(act_data, labels=act_labels, patch_artist=True)
    axs[1, 1].set_title(f'Activations (for Batch {t})')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig