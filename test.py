import torch
from torch.utils.data import DataLoader
import pickle
from multimodal import MultiModalClassifier
from dataset import IUXrayDataset

def test_model():
    with open("vocab.pkl", "rb") as f:
        saved_vocab = pickle.load(f)

    test_csv = "test.csv"
    test_dataset = IUXrayDataset(test_csv, transform=None)

    test_dataset.word2idx = saved_vocab
    test_dataset.vocab_size = len(saved_vocab)
    
    # Instantiate the model with the same parameters as training
    model = MultiModalClassifier(
        vocab_size=test_dataset.vocab_size,
        embed_dim=100,
        hidden_dim=128,
        num_classes=1
    )
    # Load the saved model weights
    model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    correct = 0
    total = 0

    with torch.no_grad():
        for images, text_seq, labels in test_loader:
            images = images.to(device)
            text_seq = text_seq.to(device)
            labels = labels.to(device)
            
            logits = model(images, text_seq)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().squeeze(1)
            
            # Print raw logits, probabilities, predicted and expected labels for this batch.
            # print("Raw logits:\n", logits.cpu().numpy())
            # print("Probabilities:\n", probs.cpu().numpy())
            # print("Predicted labels:\n", preds.cpu().numpy())
            # print("Expected labels:\n", labels.cpu().numpy())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # break  
            
    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy on first batch: {accuracy:.4f}")

if __name__ == "__main__":
    test_model()
