import torch
import random
import time
from multi_agent_functions import *
from single_agent_functions import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_image_data(batch_size):
    input_size = (batch_size, 3, 224, 224)
    x1 = torch.randn(*input_size).to(device)
    x2 = torch.randn(*input_size).to(device)
    return x1, x2

def generate_pos(batch_size):
    own_gripper_p = torch.tensor([random.randint(0, 140) for _ in range(batch_size)]).to(device)
    own_gripper_v = torch.tensor([random.randint(-5, 5) for _ in range(batch_size)]).to(device)
    other_gripper_p = torch.tensor([random.randint(0, 140) for _ in range(batch_size)]).to(device)
    other_gripper_v = torch.tensor([random.randint(-5, 5) for _ in range(batch_size)]).to(device)
    return own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v

# Initialize models
ma_cnn_encoder = ResCNNEncoder().to(device)
ma_cnn_encoder.eval()
ma_mpc_layer = MPClayer().to(device)
ma_mpc_layer.eval()

sa_cnn_encoder = SingleResCNNEncoder().to(device)
sa_cnn_encoder.eval()
sa_mpc_layer = SingleMPClayer().to(device)
sa_mpc_layer.eval()

# Test different batch sizes
batch_sizes = [1]
samples = 10  # Number of runs per batch size

print(f"{'Batch Size':<12} | {'MA Time (s)':<12} | {'SA Time (s)':<12} | {'MA/SA Ratio':<12}")
print("-" * 60)

for batch_size in batch_sizes:
    ma_total_time = 0.
    sa_total_time = 0.
    
    for _ in range(samples):
        # Generate data with current batch size
        x1, x2 = generate_image_data(batch_size)
        own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v = generate_pos(batch_size)
        
        # Multi-agent timing
        ma_start = time.time()
        encoding1 = ma_cnn_encoder(x1)
        encoding2 = ma_cnn_encoder(x2)
        ma_mpc_output = ma_mpc_layer(encoding1, encoding2, own_gripper_p, own_gripper_v, 
                                    other_gripper_p, other_gripper_v)
        torch.cuda.synchronize()  # Ensure CUDA ops are complete
        ma_end = time.time()
        print(ma_end - ma_start)
        ma_total_time += (ma_end - ma_start)
        
        # Single-agent timing (using same inputs for fair comparison)
        sa_start = time.time()
        encoding1 = ma_cnn_encoder(x1)
        sa_output = sa_mpc_layer(encoding1, own_gripper_p, own_gripper_v)
        torch.cuda.synchronize()  # Ensure CUDA ops are complete
        sa_end = time.time()
        sa_total_time += (sa_end - sa_start)
    
    # Calculate averages
    ma_avg = ma_total_time / samples
    sa_avg = sa_total_time / samples
    ratio = (ma_avg / sa_avg) * 100
    
    print(f"{batch_size:<12} | {ma_avg:<12.6f} | {sa_avg:<12.6f} | {ratio:<12.2f}%")