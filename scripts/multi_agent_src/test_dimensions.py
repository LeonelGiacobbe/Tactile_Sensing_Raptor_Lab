import torch, random, time
from multi_agent_functions import *
from single_agent_functions import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_image_data():
    input_size=(1, 3, 224, 224)
    x1 = torch.randn(*input_size).to(device)
    x2 = torch.randn(*input_size).to(device)

    return x1, x2

def generate_pos():
    own_gripper_p = torch.tensor(random.randint(0, 140)).to(device)
    own_gripper_v = torch.tensor(random.randint(-5, 5)).to(device)

    other_gripper_p = torch.tensor(random.randint(0, 140)).to(device)
    other_gripper_v = torch.tensor(random.randint(-5, 5)).to(device)

    return own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v

ma_cnn_encoder = ResCNNEncoder().to(device)
ma_cnn_encoder.eval()
ma_mpc_layer = MPClayer().to(device)
ma_mpc_layer.eval()

sa_cnn_encoder = SingleResCNNEncoder().to(device)
sa_cnn_encoder.eval()
sa_mpc_layer = SingleMPClayer().to(device)
sa_mpc_layer.eval()


ma_overall_time = 0.
sa_overall_time = 0.
samples = 1000
for _ in range(samples):
    x1, x2 = generate_image_data()
    #print("x1 size: ", x1.size())
    encoding1, encoding2 = ma_cnn_encoder(x1), ma_cnn_encoder(x2)
    own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v = generate_pos()

    # print("encoding1 size: ", encoding1.size())
    # print("encoding2 size: ", encoding2.size())
    # print("own gripper pos size: ", own_gripper_p.size())
    # print("own gripper v size: ", own_gripper_v.size())
    # print("other gripper pos size: ", other_gripper_p.size())
    # print("other gripper v size: ", other_gripper_v.size())

    ma_start = time.time()
    ma_mpc_output = ma_mpc_layer(encoding1, encoding2, own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v)
    ma_end = time.time()
    ma_temp = ma_end - ma_start
    ma_overall_time += ma_temp

    sa_start = time.time()
    sa_output = sa_mpc_layer(encoding1, own_gripper_p, own_gripper_v)
    sa_end = time.time()
    sa_temp = sa_end - sa_start
    sa_overall_time += sa_temp

print(f"Overall inference time for {samples} multi agent inferences: {ma_overall_time} seconds")
print(f"Overall inference time for {samples} single agent inferences: {sa_overall_time} seconds")

percentage = ma_overall_time * 100 / sa_overall_time

print(f"Multi-agent inference took {percentage:.2f}% of the time that single-agent inference took")

