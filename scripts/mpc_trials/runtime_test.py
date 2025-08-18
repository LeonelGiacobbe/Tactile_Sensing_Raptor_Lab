
import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
from multi_agent_functions import *

# Model architecture params
CNN_hidden1, CNN_hidden2 = 128, 128
CNN_embed_dim = 20
res_size = 224
eps = 1e-4
nStep = 15
del_t = 1/10
dropout_p = 0.15

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate models
nn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(device)
mpc_layer = MPClayer(nHidden = CNN_embed_dim, eps = eps, nStep = nStep, del_t = del_t).to(device)

# Load weights
model_path = os.path.join("/home/leo/Documents/Tactile_Sensing_Raptor_Lab/scripts/multi_agent_src/v2_checkpoint_epoch_20.pth")
checkpoint = torch.load(model_path, map_location=device)
nn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
mpc_layer.load_state_dict(checkpoint['mpc_layer_state_dict'])
nn_encoder.eval()
mpc_layer.eval()
print("Loaded model weights.")

# Load images
image_path_1 = "testing_images/sensor_1_wood.jpg"
image_path_2 = "testing_images/sensor_2_wood.jpg"

pil_image_1 = Image.open(image_path_1).convert("RGB")
pil_image_2 = Image.open(image_path_2).convert("RGB")

image_1 = transform(pil_image_1).unsqueeze(0).to(device)
image_2 = transform(pil_image_2).unsqueeze(0).to(device)

# Dummy gripper state
posi_1 = torch.tensor([25.0]).to(device)
vel_1 = torch.tensor([0.6]).to(device)
posi_2 = torch.tensor([25.0]).to(device)
vel_2 = torch.tensor([0.6]).to(device)

# Run inference
with torch.no_grad():
    output_1 = nn_encoder(image_1)
    output_2 = nn_encoder(image_2)
    
    pos_sequences_1, pos_sequences_2 = mpc_layer(output_1, output_2, posi_1, vel_1, posi_2, vel_2)

print("Output sequence 1:", pos_sequences_1)
print("Output sequence 2:", pos_sequences_2)
