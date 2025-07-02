import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

# Define the ResCNNEncoder class (copy-pasted from your provided code)
class ResCNNEncoder(nn.Module):
    def __init__(self, hidden1=512, hidden2=512, dropP=0.3, outputDim=25):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.hidden1, self.hidden2 = hidden1, hidden2
        self.dropP = dropP

        resnet = models.resnet152(pretrained=True)
        # Delete the last fc layer.
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1, momentum=0.01)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2, momentum=0.01)
        self.fc3 = nn.Linear(hidden2, outputDim)

    def forward(self, x):
        with torch.no_grad(): # Keep resnet layers frozen during inference of the full model
            x = self.resnet(x[:, :, :, :])
            x = x.view(x.size(0), -1)
        # However, the newly added FC layers are not frozen (they don't have no_grad)
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropP, training=self.training) # Ensure training=False for inference
        x = self.fc3(x)
        return x

# --- Configuration for your model ---
CNN_hidden1 = 128
CNN_hidden2 = 128
CNN_embed_dim = 20 # This is your outputDim
dropout_p = 0.15
res_size = 224 # Image resize dimension

# --- Set up device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Initialize the model ---
cnn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(device)

# Set model to evaluation mode (important for BatchNorm and Dropout)
cnn_encoder.eval()

# --- Load model weights ---
# Make sure 'rgb_letac_model.pth' is in the same directory as this script,
# or provide the full path to your saved model checkpoint.
# The checkpoint structure assumes 'cnn_encoder_state_dict' key.
model_path = 'single_letac_mpc_model.pth' # Or provide the full path to your .pth file

try:
    checkpoint = torch.load(model_path, map_location=device)
    cnn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
    print(f"Successfully loaded CNN encoder weights from {model_path}")
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {model_path}")
    print("Please ensure the model file exists and the path is correct.")
    exit()
except KeyError:
    print(f"Error: 'cnn_encoder_state_dict' not found in the checkpoint file. "
          "Please verify the checkpoint structure.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    exit()


# --- Define transformations (must match training) ---
# Your original transform:
# transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])
# This normalization is unusual (mean 0, std 0.2), usually it's ImageNet means/stds.
# Make sure this is exactly what was used for training.
transform = transforms.Compose([
    transforms.Resize([res_size, res_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])
])

# --- Create a dummy image for inference ---
# You can replace this with loading a real image:
dummy_image_path = 'image2.jpg'
dummy_pil_image = Image.open(dummy_image_path)#.convert('RGB')
# Or create a blank one:

# Apply transformations
input_tensor = transform(dummy_pil_image).unsqueeze(0).to(device) # Add batch dimension

print("\nPerforming inference...")
with torch.no_grad(): # Ensure no gradients are computed for inference
    embeddings = cnn_encoder(input_tensor)

print("\n--- Embeddings ---")
print(embeddings)
print(f"Shape of embeddings: {embeddings.shape}")
print(f"Type of embeddings: {embeddings.dtype}")