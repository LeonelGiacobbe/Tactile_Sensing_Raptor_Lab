import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from functions import *
from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import LinearRegression
import torchvision.transforms as transforms



model = torch.load("../trained_model.pth")
cnn_encoder.load_state_dict(model["cnn_encoder_state_dict"])
MPC_layer.load_state_dict(model['MPC_layer_state_dict'])
optimizer.load_state_dict(model['optimizer_state_dict'])