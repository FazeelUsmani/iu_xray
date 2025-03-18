import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

# Maximum length for text tokens (adjust as needed)
MAX_TEXT_LEN = 50

def contains_pneumonia(text):
    """Return 1 if 'pneumonia' is found in the text, else 0."""
    return 1 if "pneumonia" in text.lower() else 0

class IUXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        :param csv_file: Path to CSV containing columns: [image_path, report]
                         Optionally, a 'label' column may exist.
        :param transform: Optional torchvision transforms for image augmentation/preprocessing.
        """
        self.data = pd.read_csv(csv_file)

        # Define augmentation transforms for minority class (label = 1)
        self.augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        # Default transform for majority class
        self.default_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        self.transform = transform

        # Build vocabulary from all words in the dataset
        all_text = []
        for txt in self.data["report"].astype(str):
            all_text.extend(txt.lower().split())
        vocab = sorted(set(all_text))
        self.word2idx = {word: i + 1 for i, word in enumerate(vocab)}
        self.word2idx["<PAD>"] = 0
        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["image_path"]
        report = str(row["report"])
        text_tokens = report.lower().split()

        if "label" in self.data.columns:
            label = row["label"]
        else:
            label = contains_pneumonia(report)

        image = Image.open(os.path.join("images", img_path)).convert("RGB")

        # Apply augmentation if positive label
        if label == 1:
            image = self.augment(image)
        else:
            image = self.default_transform(image)

        token_ids = [self.word2idx.get(word, 0) for word in text_tokens]
        if len(token_ids) < MAX_TEXT_LEN:
            token_ids += [0] * (MAX_TEXT_LEN - len(token_ids))
        else:
            token_ids = token_ids[:MAX_TEXT_LEN]

        text_tensor = torch.tensor(token_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return image, text_tensor, label_tensor
