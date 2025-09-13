import torch

# Path to your .pth file
model_path = 'multi_agent_model.pth'

# Load the file
checkpoint = torch.load(model_path)

# Access the epoch count
epoch_count = checkpoint['epoch']

print(f"The model was saved after epoch: {epoch_count}")