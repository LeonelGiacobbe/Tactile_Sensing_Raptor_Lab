import torch, random, time
from multi_agent_functions import *
from single_agent_functions import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_image_data():
    input_size=(1, 3, 224, 224)
    x1 = torch.randn(*input_size).to(device)

    return x1

def generate_pos():
    own_gripper_p = torch.tensor(random.randint(0, 140)).to(device)
    own_gripper_v = torch.tensor(random.randint(-5, 5)).to(device)

    return own_gripper_p, own_gripper_v

ma_cnn_encoder = ResCNNEncoder().to(device)
ma_cnn_encoder.eval()
ma_mpc_layer = MPClayer().to(device)
ma_mpc_layer.eval()

samples = 10
for _ in range(samples):
    x = generate_image_data()
    #print("x1 size: ", x1.size())
    encoding = ma_cnn_encoder(x)
    gripper_p, gripper_v = generate_pos()

    # Forward pass with identical inputs
    out1, out2 = ma_mpc_layer(encoding, encoding, gripper_p, gripper_v, gripper_p, gripper_v)
    print("Symmetric test - Agent 1 output:", out1.mean().item())
    print("Symmetric test - Agent 2 output:", out2.mean().item())