import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import IUXrayDataset
from multimodal import MultiModalClassifier
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabulary from training
with open("vocab.pkl", "rb") as f:
    word2idx = pickle.load(f)
print("Vocabulary loaded successfully.")

# Dataset and DataLoader (Test Data)
test_dataset = IUXrayDataset(csv_file='data/test.csv', transform=None)
test_dataset.word2idx = word2idx
test_dataset.vocab_size = len(word2idx)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model
model = MultiModalClassifier(
    vocab_size=test_dataset.vocab_size,
    embed_dim=128,
    hidden_dim=256,
    num_classes=1,
    dropout=0.3
)

# Load the trained model
model.load_state_dict(torch.load('models/trained_model.pth', map_location=device))
model.to(device)
model.eval()

# Run inference
all_labels, all_preds, all_probs = [], [], []
with torch.no_grad():
    for images, texts, labels in test_loader:
        images, texts = images.to(device), texts.to(device)
        outputs = model(images, texts)

        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs >= 0.3).long()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Classification report
report = classification_report(all_labels, all_preds)
print("\nClassification Report:\n", report)

# Save metrics
with open("results/evaluation_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.show()

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)

plt.figure()
plt.plot(recall_vals, precision_vals, marker='.', label='Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig("results/precision_recall_curve.png")
plt.show()
