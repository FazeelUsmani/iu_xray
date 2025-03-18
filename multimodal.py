import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class MultiModalClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=1, dropout=0.5):
        """
        :param vocab_size: Size of the vocabulary for text embedding.
        :param embed_dim: Dimensionality of the word embeddings.
        :param hidden_dim: Hidden dimension of the LSTM.
        :param num_classes: Number of output classes.
        :param dropout: Dropout rate for regularization.
        """
        super(MultiModalClassifier, self).__init__()

        # Image feature extraction using ResNet18
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()  # remove final classification layer
        self.img_feat_dim = 512

        # Text encoding using Embedding + LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # MLP classification layers
        self.fc1 = nn.Linear(self.img_feat_dim + hidden_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, image, text_seq):
        # Image features
        img_features = self.cnn(image)

        # Text features
        embedded_text = self.embedding(text_seq)
        _, (h_n, _) = self.lstm(embedded_text)
        text_features = h_n.squeeze(0)

        # Concatenate image and text features
        combined = torch.cat([img_features, text_features], dim=1)

        # Pass through classification layers
        x = self.dropout(self.relu(self.fc1(combined)))
        logits = self.fc2(x)
        return logits

if __name__ == "__main__":
    vocab_size = 2075
    model = MultiModalClassifier(vocab_size, embed_dim=128, hidden_dim=256, num_classes=1)

    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_text = torch.randint(1, vocab_size, (4, 50))

    logits = model(dummy_images, dummy_text)
    print("Model output shape:", logits.shape)