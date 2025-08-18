import os, glob, csv
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from multi_agent_functions import *
from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import LinearRegression
import torchvision.transforms as transforms

# EncoderCNN architecture
CNN_hidden1, CNN_hidden2 = 128, 128
CNN_embed_dim = 20
res_size = 224
dropout_p = 0.15

# Training parameters
epochs = 50
batch_size = 256
learning_rate = 1e-4
eps = 1e-4
nStep = 15
del_t = 1/10

data_path = "../data_collection/alternating_dataset"

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])])

def get_folder_info(folder_name):
    parts = folder_name.split('_')
    trial = int(parts[1])
    dp = int(parts[3])
    material = parts[4]
    y = float(parts[6])
    z = float(parts[8])
    gpown = float(parts[10])
    gpother = float(parts[12])
    return trial, dp, material, y, z, gpown, gpother

def read_and_pair_data(data_path):
    all_folders = []
    for material_folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, material_folder)):
            for trial_folder in os.listdir(os.path.join(data_path, material_folder)):
                all_folders.append(os.path.join(data_path, material_folder, trial_folder))

    grouped_folders = {}
    for folder in all_folders:
        trial, dp, _, _, _, _, _ = get_folder_info(os.path.basename(folder))
        key = (trial, dp)
        if key not in grouped_folders:
            grouped_folders[key] = []
        grouped_folders[key].append(folder)

    own_selected_all_names = []
    other_selected_all_names = []
    own_output_p = []
    other_output_p = []
    own_grip_posi_num = []
    other_grip_posi_num = []
    own_grip_vel_num = []
    other_grip_vel_num = []

    for key, folders in grouped_folders.items():
        if len(folders) == 2:
            folder1, folder2 = folders
            _, _, material1, y1, z1, gpown1, gpother1 = get_folder_info(os.path.basename(folder1))
            _, _, material2, y2, z2, gpown2, gpother2 = get_folder_info(os.path.basename(folder2))

            # Assuming 'empty' is always one of the materials
            if material1 == 'empty':
                empty_folder = folder1
                other_folder = folder2
                other_material = material2
            else:
                empty_folder = folder2
                other_folder = folder1
                other_material = material1
            
            empty_images = sorted(os.listdir(empty_folder))
            other_images = sorted(os.listdir(other_folder))

            for empty_img_name, other_img_name in zip(empty_images, other_images):
                # Assuming frame numbers match
                own_selected_all_names.append(os.path.join(empty_folder, empty_img_name))
                other_selected_all_names.append(os.path.join(other_folder, other_img_name))

                # Extract gripper positions from filenames
                own_gp = float(empty_img_name.split('_')[2])
                other_gp = float(other_img_name.split('_')[2])
                own_grip_posi_num.append(own_gp)
                other_grip_posi_num.append(other_gp)

                # Generate random velocity and output pressure
                rand_vel_1 = 2 * (random.random() - 0.5)
                rand_vel_2 = 2 * (random.random() - 0.5)
                own_grip_vel_num.append(rand_vel_1)
                other_grip_vel_num.append(rand_vel_2)
                
                own_output_p.append(gpown1)
                other_output_p.append(gpother1)


    return (own_selected_all_names, other_selected_all_names,
            own_output_p, other_output_p,
            own_grip_posi_num, other_grip_posi_num,
            own_grip_vel_num, other_grip_vel_num)


def train(model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, MPC_layer= model
    cnn_encoder.train()
    MPC_layer.train()
    losses = []
    scores = []

    N_count = 0
    epoch_count = 0
    # Train loader contains (own, other info)
    for (combined_X, combined_y) in train_loader:
        (own_X_image, own_pv_pair) = combined_X[0]
        (other_X_image, other_pv_pair) = combined_X[1]
        # distribute data to device
        own_gripper_p = own_pv_pair[0].to(device)
        own_gripper_v = own_pv_pair[1].to(device)

        other_gripper_p = other_pv_pair[0].to(device)
        other_gripper_v = other_pv_pair[1].to(device)

        X_own_image = own_X_image.to(device)
        X_other_image = other_X_image.to(device)

        y_own = combined_y[0].to(device).view(-1, )
        y_other = combined_y[1].to(device).view(-1, )
        
        X_own, y_own = X_own_image.to(device), y_own.to(device).view(-1, )
        X_other, y_other = X_other_image.to(device), y_other.to(device).view(-1, )
        N_count += X_own.size(0)
        optimizer.zero_grad()
        own_output = cnn_encoder(X_own)
        other_output = cnn_encoder(X_other)

        output_1, output_2 = MPC_layer(own_output, other_output, own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v)

        y_own= y_own.unsqueeze(1).expand(X_own.size(0), output_1.size(1))
        final_y_own = y_own[:,(output_1.size(1)-1)]*3
        final_output_own = output_1[:,(output_1.size(1)-1)]*3

        y_other= y_other.unsqueeze(1).expand(X_other.size(0), output_2.size(1))
        final_y_other = y_other[:,(output_2.size(1)-1)]*3
        final_output_other = output_2[:,(output_2.size(1)-1)]*3

        loss = F.mse_loss(output_1,y_own.float()) + F.mse_loss(output_2,y_other.float()) + F.mse_loss(final_y_own,final_output_own) + F.mse_loss(final_y_other,final_output_other)
        losses.append(loss.item())
        print("Agent 1 loss: ", F.mse_loss(output_1,y_own.float()).item() + F.mse_loss(final_y_own,final_output_own).item())
        print("Agent 2 loss: ", F.mse_loss(output_2,y_other.float()).item() + F.mse_loss(final_y_other,final_output_other).item())
        loss.backward()
        optimizer.step()
        epoch_count += 1

        print("\033[92mTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\033[0m".format(
            epoch + 1, N_count, len(train_loader.dataset), 100. * (epoch_count + 1) / len(train_loader), loss.item()))
    
    return np.mean(losses)

def validation(model, device, test_loader):
    cnn_encoder, MPC_layer= model
    cnn_encoder.eval()
    MPC_layer.eval()
    test_loss = 0
    loss_list = []
    new_loss_list = []
    all_y_own = []
    all_y_other = []
    all_y_pred_own = []
    all_y_pred_other = []
    with torch.no_grad():
        for (combined_X, combined_y) in test_loader:
            (own_X_image, own_pv_pair) = combined_X[0]
            (other_X_image, other_pv_pair) = combined_X[1]
            # distribute data to device
            own_gripper_p = own_pv_pair[0].to(device)
            own_gripper_v = own_pv_pair[1].to(device)

            other_gripper_p = other_pv_pair[0].to(device)
            other_gripper_v = other_pv_pair[1].to(device)

            X_own_image = own_X_image.to(device)
            X_other_image = other_X_image.to(device)

            y_own = combined_y[0].to(device).view(-1, )
            y_other = combined_y[1].to(device).view(-1, )
            
            X_own, y_own = X_own_image.to(device), y_own.to(device).view(-1, )
            X_other, y_other = X_other_image.to(device), y_other.to(device).view(-1, )

            own_output = cnn_encoder(X_own)
            other_output = cnn_encoder(X_other)

            output_1, output_2 = MPC_layer(own_output, other_output, own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v)

            y_own = y_own.unsqueeze(1).expand(X_own.size(0), output_1.size(1)) # size batchSize, nStep
            final_y_own = y_own[:,(output_1.size(1)-1)]*3
            final_output_own = output_1[:,(output_1.size(1)-1)]*3

            y_other = y_other.unsqueeze(1).expand(X_other.size(0), output_2.size(1))
            final_y_other = y_other[:,(output_2.size(1)-1)]*3
            final_output_other = output_2[:,(output_2.size(1)-1)]*3

            loss = F.mse_loss(output_1,y_own.float()) + F.mse_loss(output_2,y_other.float()) + F.mse_loss(final_y_own,final_output_own) + F.mse_loss(final_y_other,final_output_other)
            print("Agent 1 loss: ", F.mse_loss(output_1,y_own.float()).item() + F.mse_loss(final_y_own,final_output_own).item())
            print("Agent 2 loss: ", F.mse_loss(output_2,y_other.float()).item() + F.mse_loss(final_y_other,final_output_other).item())
            loss_list.append(loss.item())
            test_loss += F.mse_loss(output_1,y_own.float()).item() + F.mse_loss(output_2,y_other.float()).item()
            y_pred_own = output_1.max(1, keepdim=True)[1]
            y_pred_other = output_2.max(1, keepdim=True)[1]
            all_y_own.extend(y_own)
            all_y_other.extend(y_other)

            all_y_pred_own.extend(y_pred_own)
            all_y_pred_other.extend(y_pred_other)
    
    test_loss = np.mean(loss_list)
    all_y_own = torch.stack(all_y_own, dim=0)
    all_y_other = torch.stack(all_y_other, dim=0)
    all_y_pred_own = torch.stack(all_y_pred_own, dim=0)
    all_y_pred_other = torch.stack(all_y_pred_other, dim=0)
    print("\033[92m\nTest set ({:d} samples): Average loss: {:.4f}\n\033[0m".format(len(all_y_own + all_y_other), test_loss))
    return test_loss

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data loading parameters
params = {"batch_size": batch_size, "shuffle": True, "num_workers": 4, "pin_memory": True} if use_cuda else {}

(own_selected_all_names_, other_selected_all_names_,
 own_output_p_, other_output_p_,
 own_grip_posi_num_, other_grip_posi_num_,
 own_grip_vel_num_, other_grip_vel_num_) = read_and_pair_data(data_path)

own_pv_pair_list = zip(own_grip_posi_num_,own_grip_vel_num_)
own_frame_pair_list = zip(own_selected_all_names_, range(len(own_selected_all_names_)))
own_all_x_list = list(zip(own_frame_pair_list, own_pv_pair_list))
own_all_y_list = (own_output_p_)

other_pv_pair_list = zip(other_grip_posi_num_, other_grip_vel_num_)
other_frame_pair_list = zip(other_selected_all_names_, range(len(other_selected_all_names_)))
other_all_x_list = list(zip(other_frame_pair_list, other_pv_pair_list))
other_all_y_list = (other_output_p_)

# Combine own/other info for appropriate splitting
combined_all_x_list = []
for (own_fp, own_pv), (other_fp, other_pv) in zip(own_all_x_list, other_all_x_list):
    combined_all_x_list.append(((own_fp, own_pv), (other_fp, other_pv)))

combined_all_y_list = []
for (own, other) in zip(own_all_y_list, other_all_y_list):
    combined_all_y_list.append((own, other))

# train, test split
train_list, test_list, train_label, test_label = train_test_split(combined_all_x_list, combined_all_y_list, test_size=0.2, random_state=42)

train_set, valid_set = Dataset_LeTac(train_list, train_label, np.arange(1, 50, 1).tolist(), transform=transform), \
                       Dataset_LeTac(test_list, test_label, np.arange(1, 50, 1).tolist(), transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# Create model
cnn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(device)
MPC_layer = MPClayer(nHidden = CNN_embed_dim, eps = eps, nStep = nStep, del_t = del_t).to(device)
letac_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
            list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
            list(cnn_encoder.fc3.parameters()) + list(MPC_layer.parameters())

optimizer = torch.optim.Adam(letac_params, lr=learning_rate, weight_decay=1e-4) # L2 regularizer of 1e-4

# Load checkpoint if exists
start_epoch = 0
checkpoint_path = 'v2_checkpoint_epoch_15.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    cnn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
    MPC_layer.load_state_dict(checkpoint['mpc_layer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # +1 because we want to start from the next epoch
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    print("\033[91mWarning! Could not find checkpoint for this model\033[0m")

# start training - modify range to start from start_epoch
for epoch in range(start_epoch, epochs):
    with open("loss_log.csv", mode='a') as f:  # 'a' mode appends rather than overwrites
        valid_loss = validation([cnn_encoder, MPC_layer], device, valid_loader)
        training_loss = train([cnn_encoder, MPC_layer], device, train_loader, optimizer, epoch)
        writer = csv.writer(f)
        writer.writerow([epoch+1, valid_loss, training_loss])
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'./v2_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'cnn_encoder_state_dict': cnn_encoder.state_dict(),
                'mpc_layer_state_dict': MPC_layer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
