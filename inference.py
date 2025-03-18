import torch
from multimodal import MultiModalClassifier

# Make sure to set vocab_size to the same value used during training.
vocab_size = 5000  # Replace with your actual vocab_size
model = MultiModalClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_classes=1, dropout=0.3)

# Load the saved state dictionary; map_location ensures compatibility if you're on CPU
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device("cpu")))
model.eval()  # Set to evaluation mode

# Now you can use the model for inference:
# For example, prepare a dummy input or load your test data, then:
# output = model(image_tensor, text_tensor)
