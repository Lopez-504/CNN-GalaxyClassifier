# ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Bloque Residual Básico
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1  # ResNet básico

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# --------------------------------------
#     ResNet para Galaxy Classification
# --------------------------------------
class ResNetGalaxy(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetGalaxy, self).__init__()

        self.in_channels = 64

        # Entrada: 3 canales, tamaño 69x69
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- Etapas de ResNet ----
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # El tamaño depende de 69×69, calculamos:
        # 69 → 35 → 18 → 9 → 5 → 3 (aprox)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Clasificador final
        self.fc = nn.Linear(512, num_classes)

    # Crea una capa ResNet (varios bloques)
    def _make_layer(self, out_channels, blocks, stride):
        downsample = None

        # Cuando cambia # de canales o stride, ajustamos el "skip"
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)  # 64
        x = self.layer2(x)  # 128
        x = self.layer3(x)  # 256
        x = self.layer4(x)  # 512

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        
        
# ------------------------------------

# Transformations and training
# Train fo ResNet

# Crear el Dataset, aumentarlo y dividirlo
dataset = GalaxyH5Dataset('/content/drive/MyDrive/galaxies_project3.h5')

from torchvision import transforms

# --- TRAIN: con Data Augmentation ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),        
    transforms.RandomHorizontalFlip(),   
    transforms.RandomRotation(10),        
    transforms.ToTensor(),              
    transforms.Normalize(                 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- VALIDATION / TEST: sin augmentation ---
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(                 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Splits
train_percent = 0.70
val_percent = 0.15
test_percent = 0.15

total_len = len(dataset)
len_train = int(total_len * train_percent)
len_val   = int(total_len * val_percent)
len_test  = total_len - len_train - len_val

train_set, val_set, test_set = random_split(dataset, [len_train, len_val, len_test])

# Asignar transforms correctos
train_set.dataset.transform = train_transforms          # Esto no está haciendo las transformaciones
val_set.dataset.transform   = val_transforms
test_set.dataset.transform  = val_transforms

# Calcular longitudes absolutas
total_len = len(dataset)

longitud_train = int(total_len * train_percent)
longitud_val = int(total_len * val_percent)
longitud_test = total_len - longitud_train - longitud_val
train_set, val_set, test_set = random_split(dataset, [longitud_train, longitud_val, longitud_test])

# Loaders
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=256, shuffle=False)

# Inicializar Modelo 
model = ResNetGalaxy(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0008)

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100
)

# 100 epochs --> 15min

# -----------------------------------

# REsultados

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

'''
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
'''

# ------------------------

# Predictions

import numpy as np

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

print("True labels collected:", y_true.shape)
print("Predicted labels collected:", y_pred.shape)



# ------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# -----------------------------------

import torch.nn.functional as F

# 1. Set the model to evaluation mode
model.eval()

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

print("Shape of predicted probabilities:", y_pred_probabilities.shape)
print("Shape of true labels for ROC:", y_true_roc.shape)

# -----------------------------

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

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


# ---------------------------

# Macro averages

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def calculate_multiclass_metrics(cm):
    """
    Calculates Macro-Averaged Precision, Recall, F1-Score, and overall Accuracy
    from true labels and predicted labels for a multiclass classification problem.

    Args:
        y_true (array): Ground truth (correct) target values.
        y_pred (array): Estimated targets as returned by a classifier.
        labels (list, optional): List of labels to index the matrix. If None,
                                  labels are inferred from the data.

    Returns:
        dict: A dictionary containing 'accuracy', 'macro_precision',
              'macro_recall', and 'macro_f1'.
    """
    # 1. Generate the Confusion Matrix (C)
    #C = confusion_matrix(y_true, y_pred, labels=labels)
    C = cm
    num_classes = C.shape[0]

    # Calculate Overall Accuracy (sum of diagonal / sum of all elements)
    overall_accuracy = np.trace(C) / np.sum(C)

    # Initialize lists to store metrics for each class
    precision_list = []
    recall_list = []
    f1_list = []

    # 2. Iterate over each class (One-vs-Rest approach)
    for i in range(num_classes):
        # The confusion matrix uses: Rows = True Class, Columns = Predicted Class

        # True Positives (TP): Correctly predicted class i
        TP = C[i, i]

        # False Negatives (FN): True class i, but predicted as something else (Row sum excluding diagonal)
        FN = np.sum(C[i, :]) - TP

        # False Positives (FP): True class is NOT i, but predicted as i (Column sum excluding diagonal)
        FP = np.sum(C[:, i]) - TP

        # We don't need True Negatives (TN) for Macro-Averaging P, R, F1,
        # but for completeness: TN = total samples - (TP + FN + FP)
        # TN = np.sum(C) - (TP + FN + FP)

        # Calculate metrics for the current class i

        # Handle the case where the class might not have any true instances (TP + FN = 0)
        # or no predicted instances (TP + FP = 0) to prevent ZeroDivisionError.
        if (TP + FP) > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0.0

        if (TP + FN) > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0.0

        # F1-Score calculation
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)

    # 3. Calculate Macro Averages (unweighted mean of per-class metrics)
    macro_precision = np.mean(precision_list)
    macro_recall = np.mean(recall_list)
    macro_f1 = np.mean(f1_list)

    return {
        "accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": precision_list,
        "per_class_recall": recall_list,
        "per_class_f1": f1_list,
    }

# 2. Get the 3x3 Confusion Matrix
print(cm)
print("-" * 30)

# 3. Calculate metrics using the custom function
custom_metrics = calculate_multiclass_metrics(cm)

print("--- Macro Averages ---")
print(f"Overall Accuracy:   {custom_metrics['accuracy']:.4f}")
print(f"Macro Precision:    {custom_metrics['macro_precision']:.4f}")
print(f"Macro Recall:       {custom_metrics['macro_recall']:.4f}")
print(f"Macro F1-Score:     {custom_metrics['macro_f1']:.4f}")
print("\nPer-Class Metrics:")

class_labels = list(range(9))
for i, label in enumerate(class_labels):
    print(f"  Class {label} | P: {custom_metrics['per_class_precision'][i]:.4f} | R: {custom_metrics['per_class_recall'][i]:.4f} | F1: {custom_metrics['per_class_f1'][i]:.4f}")
print("-" * 30)


# 4. Verification using scikit-learn's built-in report (Recommended for real work)
print("\n--- scikit-learn Classification Report (Verification) ---")
# The report provides:
# - Precision, Recall, F1 for each class.
# - Micro, Macro, and Weighted Averages.
#print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))

# Note on Micro vs. Accuracy:
# Micro-Average is mathematically equivalent to overall Accuracy.
micro_f1 = accuracy_score(y_true, y_pred)
print(f"Micro F1-Score / Overall Accuracy (Verification): {micro_f1:.4f}")


# ---------------------------

def save_metrics_to_file(metrics_data, filepath="multiclass_summary.txt"):
    """
    Writes the calculated multiclass metrics to a text file in Markdown format.
    """
    output = []

    output.append("Multiclass Classification Metric Summary")
    output.append("\nDataset: Sample Data")
    output.append(f"Classes: {list(range(9))}")
    output.append(f"Total Samples: {len(y_true)}")
    output.append("\n\n--- Aggregated (Macro) Metrics ---")

    output.append(f"\nOverall Accuracy: {metrics_data['accuracy']:.4f}")
    output.append("(Micro-Average F1 is equal to Overall Accuracy)")

    output.append(f"\nMacro Precision: {metrics_data['macro_precision']:.4f}")
    output.append(f"Macro Recall: {metrics_data['macro_recall']:.4f}")
    output.append(f"Macro F1-Score: {metrics_data['macro_f1']:.4f}")

    output.append("\n\n--- Per-Class Metrics (One-vs-Rest) ---")
    output.append("\n| Class | Precision (P) | Recall (R) | F1-Score |")
    output.append("| ----- | ----- | ----- | ----- |")

    for i, label in enumerate(list(range(9))):
        line = f"| {label} | {metrics_data['per_class_precision'][i]:.4f} | {metrics_data['per_class_recall'][i]:.4f} | {metrics_data['per_class_f1'][i]:.4f} |"
        output.append(line)

    with open(filepath, "w") as f:
        f.write('\n'.join(output))

# Save metrics
output_filename = "multiclass_summary.txt"
save_metrics_to_file(custom_metrics, filepath=output_filename)

