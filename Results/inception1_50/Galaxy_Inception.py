# Librerias
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.models import Inception_V3_Weights
import h5py
import numpy as np
import os
import glob
import shutil
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch.nn.functional as F

'''------------------------ Gestión de Carpetas -------------------------'''

def create_experiment_folder(base_name="run_inception", suffix="_v3", base_path="./experiments"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    existing_folders = glob.glob(os.path.join(base_path, f"{base_name}*{suffix}"))
    max_run = 0
    for folder in existing_folders:
        folder_name = os.path.basename(folder)
        try:
            part = folder_name.replace(base_name, "").replace(suffix, "")
            num = int(part)
            if num > max_run:
                max_run = num
        except ValueError:
            continue

    new_run_num = max_run + 1
    new_folder_name = f"{base_name}{new_run_num}{suffix}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    
    os.makedirs(new_folder_path)
    print(f"--> Carpeta de experimento creada: {new_folder_path}")
    return new_folder_path

def save_code_copy(dest_folder):
    try:
        current_file = os.path.abspath(__file__)
        shutil.copy(current_file, os.path.join(dest_folder, "source_code_copy.py"))
        print("--> Copia del código guardada.")
    except NameError:
        print("Advertencia: No se pudo copiar el código fuente automáticamente.")
    except Exception as e:
        print(f"Advertencia: No se pudo copiar el código: {e}")

'''------------------------ Setup -------------------------'''
EXP_DIR = create_experiment_folder(base_name="run_inception", suffix="_v3")
save_code_copy(EXP_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Clase Dataset para cargar el archivo .h5
class GalaxyH5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"No se encuentra el archivo: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            self.images = f['images'][:]
            self.labels = f['ans'][:]

        self.num_classes = len(np.unique(self.labels))
        print(f"Dataset cargado. Imágenes: {self.images.shape}, Clases: {self.num_classes}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = img.astype(np.float32) / 255.0
        # PyTorch usa (C,H,W)
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)

'''------------------------ Balancear Clases -------------------------'''

def get_sampler_for_subset(subset):
    print("Calculando pesos para equilibrar clases... (esto puede tardar un poco)")
    targets = []
    for index in subset.indices:
        _, label = subset.dataset[index]
        targets.append(int(label))

    targets = torch.tensor(targets)
    class_counts = torch.bincount(targets)
    class_weights = 1. / (class_counts.float() + 1e-6)
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print(f"Pesos calculados. Clases encontradas en subset: {len(class_counts)}")
    return sampler

'''Whitening'''
class InstanceWhitening(object):
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def __call__(self, img):
        mean = img.mean()
        std = img.std()
        if std < self.epsilon:
            std = self.epsilon
        return (img - mean) / std

'''------------------------ Modelo Inception V3 -------------------------'''

class GalaxyInceptionV3(nn.Module):
    def __init__(self, num_classes, freeze_features=False, use_aux=True):
        super(GalaxyInceptionV3, self).__init__()
        
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=weights, aux_logits=use_aux)
        self.use_aux = use_aux

        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        num_ftrs_main = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs_main, num_classes)

        if self.use_aux and self.model.AuxLogits is not None:
            num_ftrs_aux = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)

    def forward(self, x):
        return self.model(x)

'''------------------------ Funciones de Incertidumbre -------------------------'''

def enable_dropout(m):
    """ Función auxiliar para activar Dropout durante inferencia """
    if type(m) == nn.Dropout:
        m.train()

def compute_entropy(probs, dim=1):
    """ Calcula la entropía de Shannon H(x) = -sum(p * log(p)) """
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=dim)

def uncertainty_estimation(model, loader, n_samples=20):
    """
    Realiza Monte Carlo Dropout para estimar incertidumbre Epistémica y Aleatoria.
    """
    print(f"--> Iniciando estimación de incertidumbre (MC Dropout) con {n_samples} pasadas...")
    
    model.eval()
    model.apply(enable_dropout)
    
    epistemic_uncertainties = []
    aleatoric_uncertainties = []
    total_uncertainties = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            
            mc_outputs = torch.zeros(n_samples, images.size(0), dataset.num_classes).to(device)
            
            for k in range(n_samples):
                out = model(images)
                if isinstance(out, tuple):
                    out = out[0]
                mc_outputs[k] = F.softmax(out, dim=1)

            mean_probs = mc_outputs.mean(dim=0) # (batch, classes)
            
            total_unc = compute_entropy(mean_probs)
            entropies = compute_entropy(mc_outputs, dim=2) 
            aleatoric_unc = entropies.mean(dim=0)
            epistemic_unc = total_unc - aleatoric_unc

            epistemic_uncertainties.append(epistemic_unc.cpu().numpy())
            aleatoric_uncertainties.append(aleatoric_unc.cpu().numpy())
            total_uncertainties.append(total_unc.cpu().numpy())
            all_labels.append(labels.numpy())
            
            _, predicted = torch.max(mean_probs, 1)
            all_preds.append(predicted.cpu().numpy())
            
            if (i+1) % 10 == 0:
                print(f"Procesado batch {i+1}/{len(loader)}")

    return (np.concatenate(epistemic_uncertainties), 
            np.concatenate(aleatoric_uncertainties),
            np.concatenate(total_uncertainties),
            np.concatenate(all_labels),
            np.concatenate(all_preds))

def plot_uncertainty_dist(epistemic, aleatoric, save_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(epistemic, kde=True, color='purple', bins=30)
    plt.title('Distribución Incertidumbre Epistémica\n(Modelo / "No sé")')
    plt.xlabel('Incertidumbre (Nats)')
    plt.ylabel('Frecuencia')

    plt.subplot(1, 2, 2)
    sns.histplot(aleatoric, kde=True, color='orange', bins=30)
    plt.title('Distribución Incertidumbre Aleatoria\n(Datos / Ruido)')
    plt.xlabel('Incertidumbre (Nats)')
    plt.ylabel('Frecuencia')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'uncertainty_distributions.png'), dpi=300)
    print(f"Gráficas de incertidumbre guardadas en: {save_dir}")

'''------------------------ Training Loop -------------------------'''

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # --- Entrenamiento ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, aux_outputs = model(images)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
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
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Época [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}%")

    return history


'''------------------------ Ejecución Principal -------------------------'''
dataset_path = '/content/drive/MyDrive/galaxies_project3.h5' 
dataset = GalaxyH5Dataset(dataset_path)

# Transformaciones
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    InstanceWhitening()
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    InstanceWhitening()
])

# Splits
train_percent = 0.70
val_percent = 0.15
test_percent = 0.15

total_len = len(dataset)
train_len = int(total_len * train_percent)
val_len   = int(total_len * val_percent)
test_len  = total_len - train_len - val_len

train_subset, val_subset, test_subset = random_split(dataset, [train_len, val_len, test_len])

# Wrapper y Dataloaders
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

train_set = ApplyTransform(train_subset, transform=train_transforms)
val_set   = ApplyTransform(val_subset, transform=val_transforms)
test_set  = ApplyTransform(test_subset, transform=val_transforms)

train_sampler = get_sampler_for_subset(train_subset)

BATCH_SIZE = 128

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Modelo
model = GalaxyInceptionV3(num_classes=dataset.num_classes, freeze_features=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Entrenar
print(f"Iniciando entrenamiento. Resultados en: {EXP_DIR}")
history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)

# --- GUARDAR EL MODELO ENTRENADO ---
model_save_path = os.path.join(EXP_DIR, 'galaxy_inception_v3.pth')
torch.save(model.state_dict(), model_save_path)
print(f"--> Modelo guardado exitosamente en: {model_save_path}")

# --- Gráficas de Entrenamiento ---
def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Curvas de Pérdida')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Curvas de Precisión')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))

plot_training_history(history, EXP_DIR)

# --- Cálculo de Incertidumbre y Evaluación Final ---
epistemic, aleatoric, total_unc, y_true, y_pred = uncertainty_estimation(model, val_loader, n_samples=20)

plot_uncertainty_dist(epistemic, aleatoric, EXP_DIR)

# Guardar matriz de confusión (MC Dropout Mean)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (MC Dropout Mean)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(EXP_DIR, 'confusion_matrix.png'), dpi=300)

'''--- GUARDAR DATOS RAW PARA REPLICAR ---'''
print("--> Guardando datos crudos para replicación futura...")

# 1. Guardar historial de entrenamiento (diccionario)
history_path = os.path.join(EXP_DIR, 'training_history.npy')
np.save(history_path, history)
print(f"   Historial guardado en: {history_path}")

# 2. Guardar resultados de inferencia (arrays de numpy comprimidos)
inference_path = os.path.join(EXP_DIR, 'inference_results.npz')
np.savez(inference_path, 
         epistemic=epistemic,
         aleatoric=aleatoric,
         total_unc=total_unc,
         y_true=y_true,
         y_pred=y_pred)
print(f"   Datos de inferencia guardados en: {inference_path}")

print("Proceso finalizado correctamente.")
