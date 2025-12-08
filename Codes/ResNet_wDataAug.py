# Código de ResNet Con Data Augmentation Simple

# Librerias
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
from torchvision import transforms, models
import h5py
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


'''------------------------ Setup -------------------------'''
# Configuración del Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

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

'''------------------------ Balancear Clases -------------------------'''

def get_sampler_for_subset(subset):
    """
    Calcula un WeightedRandomSampler para equilibrar clases en un Subset.
    """
    print("Calculando pesos para equilibrar clases... (esto puede tardar un poco)")

    # 1. Extraer las etiquetas SOLO del subset de entrenamiento
    targets = []
    for index in subset.indices:
        _, label = subset.dataset[index]
        targets.append(int(label))

    targets = torch.tensor(targets)

    # 2. Contar cuántos hay de cada clase
    class_counts = torch.bincount(targets)

    # 3. Calcular peso por clase (inverso de la frecuencia)
    class_weights = 1. / class_counts.float()

    # 4. Asignar el peso correspondiente a CADA imagen del subset
    sample_weights = class_weights[targets]

    # 5. Crear el Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True                    # Permite oversampling
    )

    print(f"Pesos calculados. Clases encontradas: {len(class_counts)}")
    return sampler

'''Whitening'''
class InstanceWhitening(object):
    """
    Realiza estandarización por instancia (Per-image Whitening).
    Resta la media y divide por la desviación estándar de LA PROPIA IMAGEN.
    """
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def __call__(self, img):
        # Asumir que 'img' ya es un Tensor de PyTorch (Channel, H, W)

        # Calcular media y std de esta imagen específica
        mean = img.mean()
        std = img.std()

        # Evitar división por cero (lógica de tu script original)
        if std < self.epsilon:
            std = self.epsilon

        # Aplicar fórmula: (x - mean) / std
        return (img - mean) / std


'''------------------------ Modelo -------------------------'''
# ResNet directly from torch pretrained-models

class ResNetGalaxy(nn.Module):
    def __init__(self, num_classes=10, freeze_features=True):
        super(ResNetGalaxy, self).__init__()
        # Load a pre-trained ResNet model (e.g., ResNet50)
        self.resnet = models.resnet50(pretrained=True)

        # Freeze all parameters in the feature extraction layers if specified
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer for our specific number of classes
        # The number of input features to the fc layer depends on the ResNet variant.
        # For ResNet50, it's 2048.
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

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


'''------------------------ Entrenamiento y Validacion -------------------------'''
#dataset = GalaxyH5Dataset('/content/drive/MyDrive/Data_Share/galaxies_project3.h5')
dataset = GalaxyH5Dataset('/content/drive/MyDrive/galaxies_project3.h5')

# Definir Transformaciones

# --- TRAIN PIPELINE ---
train_transforms = transforms.Compose([
    transforms.ToPILImage(),              # 1. Asegurar formato imagen
    transforms.Resize((224, 224)),        # 2. Tamaño
    transforms.RandomHorizontalFlip(),    # 3. Augmentation geométrico
    transforms.RandomRotation(10),        # 3. Augmentation geométrico
    transforms.ToTensor(),                # 4. Convertir a Tensor (0.0 a 1.0)
    InstanceWhitening()                   # 5. Blanquear
])

# --- VALIDATION PIPELINE ---
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    InstanceWhitening()                   # También aplicamos blanqueo en validación
])

# Calcular longitudes y dividir índices

train_percent = 0.70
val_percent = 0.15
test_percent = 0.15

total_len = len(dataset)
train_len = int(total_len * train_percent)
val_len   = int(total_len * val_percent)
test_len  = total_len - train_len - val_len

# Obtenemos subsets, pero aún apuntan al dataset original sin transforms
train_subset, val_subset, test_subset = random_split(dataset, [train_len, val_len, test_len])

# CLASE WRAPPER (Bien Geminis)
class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Aplicar las transformaciones correctas a cada split
train_set = ApplyTransform(train_subset, transform=train_transforms) # AQUI va el augmentation
val_set   = ApplyTransform(val_subset, transform=val_transforms)
test_set  = ApplyTransform(test_subset, transform=val_transforms)

# DataLoaders
train_loader = DataLoader(train_set, batch_size=int(sys.argv[2]), shuffle=True)
val_loader   = DataLoader(val_set, batch_size=int(sys.argv[2]), shuffle=False)
test_loader  = DataLoader(test_set, batch_size=int(sys.argv[2]), shuffle=False)

# Modelo y Entrenamiento
model = ResNetGalaxy(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=int(sys.argv[1])          # 1er argumento 
)

import torch.nn.functional as F
# 1. Set the model to evaluation mode
model.eval()

# 2. Initialize empty lists to store true and predicted labels
y_true = []
y_pred = []

# 3. Disable gradient calculations for inference
with torch.no_grad():
    # 4. Iterate through the val_loader
    for images, labels in val_loader:
        # a. Move the images and labels to the computation device
        images, labels = images.to(device), labels.to(device)

        # b. Get the model's outputs
        outputs = model(images)

        # c. Determine the predicted class
        _, predicted = torch.max(outputs.data, 1)

        # d. Append true and predicted labels to their respective lists
        y_true.append(labels.cpu().numpy())
        y_pred.append(predicted.cpu().numpy())

# 5. Concatenate all elements into single NumPy arrays
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# 2. Initialize empty list to store predicted probabilities
y_probabilities = []

# Initialize an empty list to store true labels for ROC curve calculation (if needed later)
y_true_roc = []

# 3. Disable gradient calculations for inference
with torch.no_grad():
    # 4. Iterate through the val_loader
    for images, labels in val_loader:
        # a. Move the images and labels to the computation device
        images = images.to(device)

        # b. Get the model's outputs (logits)
        outputs = model(images)

        # c. Apply a softmax function to these logits to convert them into probabilities
        probabilities = F.softmax(outputs, dim=1)

        # d. Move these probabilities to the CPU and convert them to a NumPy array
        y_probabilities.append(probabilities.cpu().numpy())
        y_true_roc.append(labels.cpu().numpy())

# 5. Concatenate all collected probabilities into a single NumPy array
y_pred_probabilities = np.concatenate(y_probabilities, axis=0)
y_true_roc = np.concatenate(y_true_roc, axis=0)


'''Resultados'''
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Gráfica de Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Curvas de Pérdida (CrossEntropy)')
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


# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=range(dataset.num_classes), yticklabels=range(dataset.num_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Validation Set')

plt.savefig('confusion_matrix.png', dpi=300)
plt.show()



# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(dataset.num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_roc == i, y_pred_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
for i in range(dataset.num_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) curves for each class on Validation Set')
plt.legend(loc="lower right")

plt.savefig('roc_curves.png', dpi=300)
plt.show()


