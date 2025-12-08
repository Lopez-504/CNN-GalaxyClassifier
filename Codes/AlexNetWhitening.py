import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
import h5py
import numpy as np
import os
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------------
# CELDA 1: Rotaciones + flips y guardado en HDF5 (streaming)
#          + plot de ejemplo de todas las transformaciones
# -------------------------------------------------------------

archivo_entrada = '/content/drive/MyDrive/galaxies_project3.h5'
archivo_salida_rot = 'galaxies_project3_rot.h5'

# labels que queremos augmentar
labels_a_augmentar = {3, 4, 6, 7, 8, 9}

with h5py.File(archivo_entrada, 'r') as f_in, \
     h5py.File(archivo_salida_rot, 'w') as f_out:
    
    data_in = f_in['images']   # (N, H, W, C)
    labels_in = f_in['ans']  # (N,)
    
    N, H, W, C = data_in.shape
    print("Dataset original:", data_in.shape)
    
    # Creamos datasets redimensionables
    dset_imgs = f_out.create_dataset(
        'images',
        shape=(0, H, W, C),
        maxshape=(None, H, W, C),
        dtype=data_in.dtype,
        compression='gzip',
        chunks=(1, H, W, C)
    )
    dset_labels = f_out.create_dataset(
        'ans',
        shape=(0,),
        maxshape=(None,),
        dtype=labels_in.dtype,
        compression='gzip',
        chunks=(1024,)
    )
    
    current_n = 0
    
    # Guardaremos SOLO una imagen de ejemplo para mostrar todas las rotaciones y flips
    img_ejemplo = None
    lab_ejemplo = None
    
    for i in range(N):
        img = data_in[i]      # (H, W, C)
        lab = labels_in[i]    # escalar
        
        # 1) Guardar siempre la imagen original
        dset_imgs.resize(current_n + 1, axis=0)
        dset_labels.resize(current_n + 1, axis=0)
        dset_imgs[current_n] = img
        dset_labels[current_n] = lab
        current_n += 1
        
        # Elegimos una imagen de ejemplo de alguna clase a augmentar
        if (img_ejemplo is None) and (lab in labels_a_augmentar):
            img_ejemplo = img[...].copy()
            lab_ejemplo = int(lab)
        
        # 2) Si el label no está en los que augmentamos, seguimos
        if lab not in labels_a_augmentar:
            continue
        
        # 3) Generar rotaciones y flips para este label (streaming)
        for k in [1, 2, 3, 4]:
            rot = np.rot90(img, k=k)   # rota en plano (H, W)
            
            # Evitar duplicar la original: 4*90° = 360° (misma imagen)
            if k != 4:
                dset_imgs.resize(current_n + 1, axis=0)
                dset_labels.resize(current_n + 1, axis=0)
                dset_imgs[current_n] = rot
                dset_labels[current_n] = lab
                current_n += 1
            
            # Flip vertical (up-down)
            rot_flip_v = np.flipud(rot)
            dset_imgs.resize(current_n + 1, axis=0)
            dset_labels.resize(current_n + 1, axis=0)
            dset_imgs[current_n] = rot_flip_v
            dset_labels[current_n] = lab
            current_n += 1
            
            # Flip horizontal (left-right)
            rot_flip_h = np.fliplr(rot)
            dset_imgs.resize(current_n + 1, axis=0)
            dset_labels.resize(current_n + 1, axis=0)
            dset_imgs[current_n] = rot_flip_h
            dset_labels[current_n] = lab
            current_n += 1
        
        if (i + 1) % 1000 == 0:
            print(f"Procesadas {i+1}/{N} imágenes (actual total: {current_n})")
    
    print("Total de imágenes en el archivo rotado:", current_n)

# -------------------------------------------------------------
# Plot de ejemplo: original + rotaciones + flips para una sola imagen
# -------------------------------------------------------------
if img_ejemplo is not None:
    # Generamos todas las variantes SOLO para visualizar
    variantes = []
    titulos = []
    
    variantes.append(img_ejemplo)
    titulos.append(f"Original (label {lab_ejemplo})")
    
    for k in [1, 2, 3, 4]:
        rot = np.rot90(img_ejemplo, k=k)
        variantes.append(rot)
        titulos.append(f"rot90 k={k}")
        
        variantes.append(np.flipud(rot))
        titulos.append(f"rot90 k={k} + flip vertical")
        
        variantes.append(np.fliplr(rot))
        titulos.append(f"rot90 k={k} + flip horizontal")
    
    n_var = len(variantes)
    n_cols = 4
    n_rows = int(np.ceil(n_var / n_cols))
    
    plt.figure(figsize=(3*n_cols, 3*n_rows))
    for i, (img_v, title) in enumerate(zip(variantes, titulos)):
        ax = plt.subplot(n_rows, n_cols, i+1)
        if img_v.ndim == 2:
            ax.imshow(img_v, cmap='gray')
        else:
            ax.imshow(img_v)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("No se encontró ninguna imagen de ejemplo con labels_a_augmentar.")

# ------------------------------------

# -------------------------------------------------------------
# CELDA 2: Muestreo de imágenes ANTES del blanqueo
# -------------------------------------------------------------

archivo_salida_rot = 'galaxies_project3_rot.h5'

num_mostrar = 5  # cuántas imágenes queremos visualizar

with h5py.File(archivo_salida_rot, 'r') as f:
    imgs = f['images']
    labs = f['ans']
    
    N = imgs.shape[0]
    print("Dataset rotado:", imgs.shape)
    
    # Elegimos índices aleatorios reproducibles
    rng = np.random.default_rng(seed=0)
    indices_muestra = rng.choice(N, size=min(num_mostrar, N), replace=False)
    indices_muestra = np.sort(indices_muestra)
    
    print("Índices de la muestra:", indices_muestra)
    print("Labels de la muestra:", labs[indices_muestra])
    
    muestra_antes = [imgs[i][...] for i in indices_muestra]
    labels_muestra = [int(labs[i]) for i in indices_muestra]

# Plot de la muestra antes del blanqueo
n = len(muestra_antes)
plt.figure(figsize=(3*n, 3))
for i, (img, lab) in enumerate(zip(muestra_antes, labels_muestra)):
    ax = plt.subplot(1, n, i+1)
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    ax.set_title(f"Idx {indices_muestra[i]}, label {lab}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()

# OJO: indices_muestra queda en memoria para usarlo después del blanqueo


# -------------------------------------------------------------
# CELDA 3: Blanqueo (whitening) de TODO el dataset rotado
#          -> Se guarda en un nuevo HDF5, streaming
# -------------------------------------------------------------

archivo_salida_rot = 'galaxies_project3_rot.h5'
archivo_salida_whiten = 'galaxies_project3_rot_whiten.h5'

with h5py.File(archivo_salida_rot, 'r') as f_in, \
     h5py.File(archivo_salida_whiten, 'w') as f_out:
    
    imgs_in = f_in['images']   # (N, H, W, C)
    labs_in = f_in['ans']      # (N,)
    
    N, H, W, C = imgs_in.shape
    print("Dataset rotado de entrada:", imgs_in.shape)
    
    # Creamos datasets de salida
    dset_imgs = f_out.create_dataset(
        'images',
        shape=(N, H, W, C),
        dtype=np.float32,        # ahora guardamos como float32 blanqueado
        compression='gzip',
        chunks=(1, H, W, C)
    )
    dset_labels = f_out.create_dataset(
        'ans',
        data=labs_in,
        compression='gzip'
    )
    
    for i in range(N):
        img = imgs_in[i][...].astype(np.float32)
        
        mean = img.mean()
        std = img.std()
        if std < 1e-6:
            std = 1e-6
        
        img_w = (img - mean) / std   # blanqueo: media 0, var ~1
        
        dset_imgs[i] = img_w
        
        if (i + 1) % 1000 == 0 or i == N-1:
            print(f"Blanqueadas {i+1}/{N} imágenes")

print("Archivo blanqueado guardado como:", archivo_salida_whiten)

# -------------------------------------------------

# -------------------------------------------------------------
# CELDA 4: Muestreo de las mismas imágenes DESPUÉS del blanqueo
# -------------------------------------------------------------

archivo_salida_whiten = 'galaxies_project3_rot_whiten.h5'

# Aquí usamos indices_muestra definido en la CELDA 2
with h5py.File(archivo_salida_whiten, 'r') as f:
    imgs_w = f['images']
    labs_w = f['ans']
    
    muestra_despues = [imgs_w[i][...] for i in indices_muestra]
    labels_muestra_w = [int(labs_w[i]) for i in indices_muestra]

n = len(muestra_despues)
plt.figure(figsize=(3*n, 3))
for i, (img, lab) in enumerate(zip(muestra_despues, labels_muestra_w)):
    ax = plt.subplot(1, n, i+1)
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        # Al estar blanqueadas (valores ~N(0,1)), puede ayudar centrar la escala
        # pero matplotlib igual hace un mapeo automático:
        ax.imshow(img)
    ax.set_title(f"Idx {indices_muestra[i]}, label {lab}\n(blanqueada)", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()

# ------------------------------------------------

# Cargar labels desde el archivo (ajusta el nombre si es distinto)
with h5py.File('galaxies_project3_augmented.h5', 'r') as f:
    labels = f['ans'][()]   # <- aquí está tu vector de labels

print("Shape de labels:", labels.shape)

# Opción 1: Histograma directo (si los labels son enteros)
plt.figure(figsize=(6,4))
plt.hist(labels, bins=np.arange(labels.min(), labels.max() + 2) - 0.5, edgecolor='black')
plt.xlabel('Label')
plt.ylabel('Frecuencia')
plt.title('Histograma de labels')
plt.xticks(np.arange(labels.min(), labels.max() + 1))
plt.tight_layout()
plt.show()

# ----------------

# Configuración del Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# --------------------

# Clase Dataset para cargar el archivo .h5
class GalaxyH5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"No se encuentra el archivo: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            # Cargamos los datos a memoria
            self.images = f['images'][:]
            self.labels = f['ans'][:]

        self.num_classes = len(np.unique(self.labels))
        print(f"Dataset cargado. Imágenes: {self.images.shape}, Clases: {self.num_classes}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]      # (69, 69, 3)
        label = self.labels[idx]

        # Normalización [0, 1] y conversión a float32
        img = img.astype(np.float32) / 255.0

        # TRANSPOSICIÓN CLAVE: Keras usa (H,W,C), PyTorch usa (C,H,W)
        # Movemos el canal de la posición 2 a la 0.
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)
        
# -----------------------       

# Arquitectura AlexNet
class AlexNetGalaxy(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetGalaxy, self).__init__()

        self.features = nn.Sequential(
            # --- Bloque 1 ---
            # Keras: Conv2D(96, (11, 11), strides=(4, 4))
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            # Keras: MaxPooling2D((3, 3), strides=(2, 2))
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Keras: BatchNormalization()
            nn.BatchNorm2d(96),

            # --- Bloque 2 ---
            # Keras: Conv2D(256, (5, 5), padding='same')
            # En PyTorch, para kernel 5 mantener tamaño (padding='same'), usamos padding=2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # Keras: MaxPooling2D((3, 3), strides=(2, 2))
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Keras: BatchNormalization()
            nn.BatchNorm2d(256),

            # --- Bloque 3 ---
            # Keras: Conv2D(384, (3, 3), padding='same') -> padding=1 en Torch
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # --- Bloque 4 ---
            # Keras: Conv2D(384, (3, 3), padding='same')
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # --- Bloque 5 ---
            # Keras: Conv2D(256, (3, 3), padding='same')
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Keras: MaxPooling2D((3, 3), strides=(2, 2))
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Keras: BatchNormalization()
            nn.BatchNorm2d(256),
        )

        # --- Clasificador (Dense Layers) ---
        # Nota matemática:
        # Input: 69x69
        # Salida final de features: 256 canales x 1 x 1
        self.flatten_size = 256 * 1 * 1

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # Keras: Dense(4096)
            nn.Linear(self.flatten_size, 4096),
            nn.ReLU(inplace=True),
            # Keras: Dropout(0.5)
            nn.Dropout(0.6),

            # Keras: Dense(4096)
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Keras: Dropout(0.5)
            nn.Dropout(0.6),

            # Keras: Dense(num_classes)
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100):
    # Listas para guardar el historial
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        # --- Entrenamiento ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train

        # Guardar métricas de entrenamiento
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # --- Validación ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total

        # Guardar métricas de validación
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Época [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}%")

    return history
    
# ------------------------------------    

# Crear el Dataset y dividirlo
dataset = GalaxyH5Dataset('/content/drive/MyDrive/galaxies_project3.h5')

# Pipeline para Entrenar (Con Data Augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),        # 1. Tamaño fijo
    transforms.RandomHorizontalFlip(),    # 2. Data Augmentation (flips)
    transforms.RandomRotation(10),        # 2. Data Augmentation (rotaciones)
    transforms.ToTensor(),                # 3. Convertir a Tensor (pone valores 0-1 automáticamente)
    transforms.Normalize(                 # 4. Estandarización (Valores típicos de ImageNet)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Pipeline para VALIDACIÓN (Sin Augmentation, solo lo necesario)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(                 # Debes usar la misma normalización que en train
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_percent = 0.70
val_percent = 0.15
test_percent = 0.15

# Calcular longitudes absolutas
total_len = len(dataset)

longitud_train = int(total_len * train_percent)
longitud_val = int(total_len * val_percent)
longitud_test = total_len - longitud_train - longitud_val
train_set, val_set, test_set = random_split(dataset, [longitud_train, longitud_val, longitud_test])

# Crear los DataLoaders
train_loader = DataLoader(train_set, batch_size=int(sys.argv[2]), shuffle=True)
val_loader   = DataLoader(val_set, batch_size=int(sys.argv[2]), shuffle=False)

# Instanciar el Modelo, Criterio y Optimizador
model = AlexNetGalaxy(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
# El optimizador necesita los parámetros del modelo para poder modificarlos
optimizer = optim.Adam(model.parameters(), lr=0.0008)

# Llamada para entrenar
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=int(sys.argv[1])
)

# ---------- Resultados -------

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Gráfica de Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Curvas de Pérdida (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Gráfica de Precisión (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Curvas de Precisión (Accuracy)')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

plot_training_history(history)

# Make predictions on the training data
y_pred_probabilities = model.predict(test_normalized_images)
y_pred_classes = np.argmax(y_pred_probabilities, axis=1)

# Get true labels (assuming one_hot_labels were used for training)
y_true_classes = np.argmax(test_one_hot_labels, axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ----------------------- 

# --- 1. Preparar el DataLoader de Prueba ---
# Nota: La celda de preparación de datos original solo crea train_loader y val_loader.
# Necesitamos crear el DataLoader para el conjunto de prueba (test_set).
# Asignamos las transformaciones de validación/prueba al dataset.
test_set.dataset.transform = val_transforms
test_loader = DataLoader(test_set, batch_size=int(sys.argv[2]), shuffle=False)

# --- 2. Evaluación del Modelo en el Conjunto de Prueba ---
print("Realizando evaluación en el conjunto de prueba...")
model.eval()
all_targets = []
all_predictions = []

with torch.no_grad():
    for images, targets in test_loader:
        # Mover los datos al dispositivo (CPU/GPU)
        images = images.to(device)
        
        # Predicción
        outputs = model(images)
        _, predictions = torch.max(outputs, 1) # Obtener el índice de la clase con mayor probabilidad

        # Almacenar resultados (mover a CPU para sklearn)
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Convertir a arrays de numpy
all_targets = np.array(all_targets)
all_predictions = np.array(all_predictions)

# --- 3. Calcular la Matriz de Confusión ---
cm = confusion_matrix(all_targets, all_predictions)
print("\nMatriz de Confusión calculada con éxito.")

# --- 4. Plotear la Matriz de Confusión ---
plt.figure(figsize=(10, 8))
# Usamos seaborn para una visualización más agradable.
# El parámetro 'annot=True' muestra los números dentro de las celdas.
# 'fmt="d"' formatea los números como enteros.
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues",
    cbar=True,
    linewidths=.5,
    linecolor='black'
)

# Etiquetas de las clases (ajusta estas si las etiquetas de clase son diferentes de 0-9)
class_labels = [str(i) for i in range(10)]
plt.xticks(ticks=np.arange(10) + 0.5, labels=class_labels, rotation=45, ha='right')
plt.yticks(ticks=np.arange(10) + 0.5, labels=class_labels, rotation=0)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Opcional: Imprimir las métricas generales (precisión/accuracy)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f"\nPrecisión (Accuracy) en Test: {accuracy:.4f}")

# ------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# --- Variables de PyTorch necesarias (ya definidas en tu contexto) ---
# val_set: El conjunto de datos de validación
# val_transforms: Las transformaciones (sin DA) aplicadas a los datos de validación
# model: El modelo AlexNetGalaxy ya entrenado
# device: El dispositivo ('cuda' o 'cpu')
num_classes = 10  # Número de clases de galaxias

# --- 1. Preparar el DataLoader de Validación ---
# Asegúrate de que el val_set tenga las transformaciones correctas
val_set.dataset.transform = val_transforms
val_loader = DataLoader(val_set, batch_size=int(sys.argv[2]), shuffle=False)

# --- 2. Evaluación del Modelo y Obtención de Probabilidades ---
print("Obteniendo probabilidades del modelo en el conjunto de validación...")
model.eval()
all_targets = []
all_probabilities = []

with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        
        # Obtener salidas (logits)
        outputs = model(images)
        
        # Convertir logits a probabilidades (softmax)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Almacenar etiquetas verdaderas y probabilidades (mover a CPU para sklearn)
        all_targets.extend(targets.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# Convertir a arrays de numpy
y_true = np.array(all_targets)
y_score = np.array(all_probabilities)

# --- 3. Binarización de las Etiquetas Verdaderas ---
# Para ROC multiclase, las etiquetas verdaderas deben estar en formato one-hot encoding
y_true_binarized = label_binarize(y_true, classes=range(num_classes))

# --- 4. Calcular Curvas ROC y AUC para cada Clase ---
print("Calculando curvas ROC y AUC...")
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    # Calcula la curva ROC (FPR, TPR, umbrales)
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
    # Calcula el Área Bajo la Curva (AUC)
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f"Clase {i}: AUC = {roc_auc[i]:.4f}")

# --- 5. Trazar las Curvas ROC ---
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('jet', num_classes) # Usar un mapa de color para distinguir las 10 clases

for i in range(num_classes):
    plt.plot(
        fpr[i], 
        tpr[i], 
        color=colors(i),
        lw=2, 
        label=f'ROC Clase {i} (AUC = {roc_auc[i]:.4f})'
    )

# Curva de referencia (clasificador aleatorio)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Aleatorio (AUC = 0.50)')

# Configuración del gráfico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR / Recall)')
plt.title('Curvas ROC (Receiver Operating Characteristic) por Clase', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
