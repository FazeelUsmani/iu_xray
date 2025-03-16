import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class MultiModalClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=1):
        """
        :param vocab_size: Size of the vocabulary for text embedding.
        :param embed_dim: Dimensionality of the word embeddings.
        :param hidden_dim: Hidden dimension for the LSTM text encoder.
        :param num_classes: Number of output classes (1 for binary classification with BCEWithLogitsLoss).
        """
        super(MultiModalClassifier, self).__init__()
        
        # 1) Image feature extractor using pre-trained ResNet-18
        # self.cnn = models.resnet18(pretrained=True)
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final classification layer to get raw feature vectors.
        self.cnn.fc = nn.Identity()
        # ResNet-18 outputs a 512-dim vector from its final average pooling layer.
        self.img_feat_dim = 512
        
        # 2) Text encoder: Embedding layer + LSTM.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        # batch_first=True makes input and output tensors shaped (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.text_feat_dim = hidden_dim
        
        # 3) Fusion and classification (MLP)
        # The input to the first FC layer is the concatenation of image and text features.
        self.fc1 = nn.Linear(self.img_feat_dim + self.text_feat_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        # For binary classification, you can use BCEWithLogitsLoss which applies sigmoid internally.
    
    def forward(self, images, text_seq):
        """
        :param images: Tensor of images, shape: (batch_size, 3, H, W)
        :param text_seq: Tensor of tokenized text indices, shape: (batch_size, seq_len)
        :return: logits, a tensor of shape (batch_size, num_classes)
        """
        # Image branch: extract image features
        img_features = self.cnn(images)  # Shape: (batch_size, 512)
        
        # Text branch: embed and pass through LSTM
        embedded = self.embedding(text_seq)       # Shape: (batch_size, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(embedded)          # h_n shape: (num_layers, batch_size, hidden_dim)
        text_features = h_n.squeeze(0)             # Assuming 1-layer LSTM, shape becomes: (batch_size, hidden_dim)
        
        # Concatenate image and text features
        combined = torch.cat([img_features, text_features], dim=1)  # Shape: (batch_size, 512 + hidden_dim)
        
        # Pass through the MLP
        x = self.fc1(combined)
        x = self.relu(x)
        logits = self.fc2(x)  # Shape: (batch_size, num_classes)
        return logits

# Example usage:
if __name__ == "__main__":
    # For demonstration, assume:
    #   - vocab_size is computed from your dataset (e.g., 10000)
    #   - Batch size = 4 and sequence length = MAX_TEXT_LEN (e.g., 50)
    vocab_size = 10000
    model = MultiModalClassifier(vocab_size=vocab_size, embed_dim=100, hidden_dim=128, num_classes=1)
    
    # Create dummy input tensors:
    # Dummy image tensor: (batch_size, 3, 224, 224)
    dummy_images = torch.randn(4, 3, 224, 224)
    # Dummy text tensor: (batch_size, seq_len) where each value is a token id
    dummy_text = torch.randint(1, vocab_size, (4, 50))
    
    # Forward pass:
    logits = model(dummy_images, dummy_text)
    print("Output logits shape:", logits.shape)
