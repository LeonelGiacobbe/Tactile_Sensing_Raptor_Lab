import torch, random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from multi_agent_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_image_data():
    input_size=(3, 224, 224)
    x1 = torch.randn(*input_size).to(device)
    x2 = torch.randn(*input_size).to(device)

    return x1, x2

def generate_pos():
    own_gripper_p = torch.tensor(random.randint(0, 140)).to(device)
    own_gripper_v = torch.tensor(random.randint(-5, 5)).to(device)

    other_gripper_p = torch.tensor(random.randint(0, 140)).to(device)
    other_gripper_v = torch.tensor(random.randint(-5, 5)).to(device)

    return own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v

cnn_encoder = ResCNNEncoder()
mpc_layer = MPClayer()

x1, x2 = generate_image_data()
encoding1, encoding2 = cnn_encoder(x1), cnn_encoder(x2)

own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v = generate_pos()

mpc_output = mpc_layer(x1, x2, own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v)

