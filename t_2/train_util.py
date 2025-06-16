import time
import numpy as np

# === Módulos estándar ===
import numpy as np
import random
import time

# === Keras ===
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.initializers import GlorotUniform # type: ignore
from tensorflow.keras.optimizers import SGD as KSGD, Adam as KAdam # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore

# === PyTorch ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# === Sklearn ===
from sklearn.neural_network import MLPClassifier


def make_model(framework, seed):
    random.seed(seed)
    np.random.seed(seed)

    if framework == "keras":
        tf.random.set_seed(seed)
        model = Sequential([
            Dense(1024, activation='relu', kernel_initializer=GlorotUniform(seed), input_shape=(32*32*3,)),
            Dense(512,  activation='relu', kernel_initializer=GlorotUniform(seed)),
            Dense(256,  activation='relu', kernel_initializer=GlorotUniform(seed)),
            Dense(128,  activation='relu', kernel_initializer=GlorotUniform(seed)),
            Dense(10,   activation='softmax',  kernel_initializer=GlorotUniform(seed)),
        ])
        model.summary()
        return model


    elif framework == "torch":
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(3072, 1024), nn.ReLU(),   
            nn.Linear(1024, 512),  nn.ReLU(),   
            nn.Linear(512, 256),   nn.ReLU(),   
            nn.Linear(256, 128),   nn.ReLU(),   
            nn.Linear(128, 10),                
            nn.Softmax(dim=1)                  
        )
        return model


    elif framework == "sklearn":
        model = MLPClassifier(
            hidden_layer_sizes=(1024, 512, 256, 128),
            activation='relu',
            solver='adam',
            max_iter=1,
            random_state=seed
        )
        return model

    else:
        raise ValueError(f"Framework '{framework}' no soportado.")

    

def train_network(network, X, y, optimizer, learning_rate, batch_size, timeout):
    results = []
    start_time = time.time()

    print(f"🔍 Detectando tipo de red: {type(network)}")

    # ======================= KERAS =======================
    try:


        if hasattr(network, "train_on_batch") and hasattr(network, "compile"):
            print("✅ Red Keras detectada. Iniciando entrenamiento...")
            loss_fn = SparseCategoricalCrossentropy()
            opt = KAdam(learning_rate) if optimizer == 'adam' else KSGD(learning_rate)
            network.compile(optimizer=opt, loss=loss_fn)

            num_samples = X.shape[0]
            while True:
                if time.time() - start_time >= timeout - 0.1:
                    break
                idx = np.random.choice(num_samples, batch_size, replace=False)
                x_batch = X[idx]
                y_batch = y[idx]
                loss = network.train_on_batch(x_batch, y_batch)
                elapsed = round(time.time() - start_time, 4)
                results.append((elapsed, float(loss)))
            print("✅ Entrenamiento Keras terminado.")
            return results
    except Exception as e:
        print("❌ Error en bloque Keras:", e)

    # ======================= PYTORCH =======================
    try:


        if isinstance(network, nn.Module):
            print("✅ Red PyTorch detectada. Iniciando entrenamiento...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            network.to(device)
            network.train()

            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.long).to(device)

            opt = optim.Adam(network.parameters(), lr=learning_rate) if optimizer == 'adam' else optim.SGD(network.parameters(), lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()

            num_samples = X.shape[0]
            while True:
                if time.time() - start_time >= timeout - 0.1:
                    break
                idx = np.random.choice(num_samples, batch_size, replace=False)
                x_batch = X_tensor[idx]
                y_batch = y_tensor[idx]

                opt.zero_grad()
                output = network(x_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                opt.step()

                elapsed = round(time.time() - start_time, 4)
                results.append((elapsed, float(loss.item())))
            print("✅ Entrenamiento PyTorch terminado.")
            return results
    except Exception as e:
        print("❌ Error en bloque PyTorch:", e)

    # ======================= SKLEARN =======================
    try:

        if isinstance(network, MLPClassifier):
            print("✅ Red sklearn detectada. Iniciando entrenamiento...")
            num_samples = X.shape[0]
            classes = np.unique(y)
            while True:
                if time.time() - start_time >= timeout - 1:
                    break
                idx = np.random.choice(num_samples, batch_size, replace=False)
                x_batch = X[idx]
                y_batch = y[idx]

                network.partial_fit(x_batch, y_batch, classes=classes)
                if time.time() - start_time >= timeout - 1:
                    break
                probs = network.predict_proba(x_batch)
                if time.time() - start_time >= timeout - 1:
                    break
                log_probs = np.log(probs + 1e-9)
                loss = -np.mean(log_probs[np.arange(len(y_batch)), y_batch])

                elapsed = round(time.time() - start_time, 4)
                results.append((elapsed, float(loss)))
            print("✅ Entrenamiento sklearn terminado.")
            return results
    except Exception as e:
        print("❌ Error en bloque sklearn:", e)

    print("⚠️ Ningún framework compatible fue detectado.")
    raise ValueError(f"Tipo de red no reconocido. Tipo recibido: {type(network)}")
