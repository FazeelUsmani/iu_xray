import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import IUXrayDataset
from multimodal import MultiModalClassifier
import pickle
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def train():
    ###################
    # 1. Dataset Setup (Training Data)
    ###################
    csv_file = "data/train.csv" 
    batch_size = 8
    num_workers = 2

    dataset = IUXrayDataset(csv_file, transform=None)
    
    # Save the vocabulary (word2idx) for later use during testing
    with open("vocab.pkl", "wb") as f:
        pickle.dump(dataset.word2idx, f)
    print("Vocabulary saved to vocab.pkl")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ###################
    # 2. Model Setup
    ###################
    model = MultiModalClassifier(
        vocab_size=dataset.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_classes=1,
        dropout=0.3
    )

    df_train = pd.read_csv(csv_file)

    # Automatically create labels if not present
    if 'label' not in df_train.columns:
        print("'label' column not found. Creating labels from 'report' column...")
        df_train['label'] = df_train['report'].apply(lambda x: 1 if 'pneumonia' in x.lower() else 0)
        df_train.to_csv(csv_file, index=False)
        print("'label' column created and saved back to train.csv")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df_train['label']),
        y=df_train['label']
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

    # Using class-weighted loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights_tensor[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ###################
    # 3. Training Loop
    ###################
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, text_seq, labels) in enumerate(dataloader):
            images = images.to(device)
            text_seq = text_seq.to(device)
            labels = labels.to(device).unsqueeze(1) 

            optimizer.zero_grad()
            logits = model(images, text_seq)  
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss:.4f}")

    # Save the trained model after training completes
    torch.save(model.state_dict(), "models/trained_model.pth")
    print("Model saved to trained_model.pth")

if __name__ == "__main__":
    train()