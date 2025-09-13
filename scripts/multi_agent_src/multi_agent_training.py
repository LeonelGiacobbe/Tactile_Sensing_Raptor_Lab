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
dropout_p = 0.35

# Training parameters
epochs = 400
batch_size = 256
learning_rate = 1e-5
eps = 1e-4
nStep = 15
del_t = 1/10

data_path = "../data_collection/dataset"

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])])
def read_empty_data(data_path):
    """
    For images when nothing is touching the sensor

    """
    all_names = []
    trials = []
    ys = []
    zs = []
    fnames = os.listdir(data_path)
    all_names = []
    own_grip_posi = []
    other_grip_posi = []
    path_list = []

    for f in fnames:
        loc1 = f.find('tr_')
        loc2 = f.find('_dp')
        trials.append(f[(loc1 + 3): loc2])
        loc3 = f.find('y_')
        loc4 = f.find('_z_')
        loc5 = f.find('_gpown_')
        loc6 = f.find('_gpother')
        ys.append(f[(loc3 + 4): loc4])
        zs.append(f[(loc4 + 3): loc5])
        own_grip_posi.append(50.) # Should match val used for target posi and vel
        other_grip_posi.append(50.) # Should match val used for target posi and vel
        path_list.append(data_path+'/'+f+'/')
        sub_fnames = os.listdir(data_path+'/'+f)
        all_names.append(sub_fnames)
        # print("appended to ys: ", f[(loc3 + 4): loc4])
        # print("appended to zs: ", f[(loc4 + 3): loc5])
        # print("appended to own grip posi: ", own_grip_posi[-1])
        # print("appended to other grip posi: ", other_grip_posi[-1])
    own_selected_all_names = []
    other_selected_all_names = []
    own_output_p = []
    other_output_p = []
    own_grip_posi_num = []
    other_grip_posi_num = []
    own_total = []
    other_total = []
    own_index = []
    other_index = []
    own_grip_vel_num = []
    other_grip_vel_num = []
    # Both ys and zs will be zeros. Not touching it just in case but I do not see the purpose
    for i in range(len(ys)):
        own_images = [img for img in all_names[i] if img.startswith('1_')]
        for j in range(len(own_images)): # Now, iterate through those files     
            img = own_images[j]
            loc1 = img.find('gp_')
            loc2 = img.find('_fr')
            img[(loc1 + 3): loc2]

            rand_num_1 = random.uniform(-0.5, 0.5)
            rand_num_2 = random.uniform(-0.5, 0.5)
            # Seems like every one of these evals in total would amount to 0?
            own_total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
            target_pos = 58.5 + rand_num_1
            own_output_p.append(target_pos)
            own_selected_all_names.append(path_list[i]+img)
            own_grip_posi_num.append(60.0)
            own_grip_vel_num.append((target_pos - 60.0) / 3.0)
            own_index.append(j)

            # Find matching frame to populate 'other' info
            fr_sloc = img.find('frame')
            fr_eloc = img.find('.jpg')
            frame_no = img[(fr_sloc + 5): fr_eloc]
            matching_img_path = glob.glob(os.path.join(path_list[i], f'2_*_frame{frame_no}.jpg'))
            filename_list = [os.path.basename(path) for path in matching_img_path]
            other_img = filename_list[0]
            other_idx_in_list = all_names[i].index(other_img)
            # Seems like every one of these evals in total would amount to 0?
            other_total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
            other_target_pos = 58.5 + rand_num_2
            other_output_p.append(other_target_pos)
            other_selected_all_names.append(path_list[i]+other_img)
            other_grip_posi_num.append(60.0)
            other_grip_vel_num.append((other_target_pos - 60.0) / 3.0)
            other_index.append(other_idx_in_list)

    return own_index,other_index,own_total,other_total,own_selected_all_names,other_selected_all_names,own_output_p,other_output_p,own_grip_posi_num,other_grip_posi_num, own_grip_vel_num, other_grip_vel_num

def read_data(data_path, label_path, up_limit=50, offset=0):
    # Load filenames
    fnames = os.listdir(data_path)
    label_dict = np.load(label_path, allow_pickle=True).item()

    own_index, other_index = [], []
    own_total, other_total = [], []
    own_selected_all_names, other_selected_all_names = [], []
    own_output_p, other_output_p = [], []
    own_grip_posi_num, other_grip_posi_num = [], []
    own_grip_vel_num, other_grip_vel_num = [], []

    # Preparse trial info
    trials_info = []
    for f in fnames:
        trial_id = f[f.find('tr_')+3 : f.find('_dp')]
        y_val = float(f[f.find('y_')+3 : f.find('_z_')])
        z_val = float(f[f.find('_z_')+3 : f.find('_gpown_')])
        own_gp = float(f[f.find('_gpown_')+7 : f.find('_gpother')])
        other_gp = float(f[f.find('_gpother')+9 :])
        trial_path = os.path.join(data_path, f)
        images = os.listdir(trial_path)
        trials_info.append({
            'trial': trial_id,
            'y': y_val,
            'z': z_val,
            'own_gp': own_gp,
            'other_gp': other_gp,
            'path': trial_path,
            'images': images
        })

    # Process trials with labels
    own_total_vals, other_total_vals = [], []
    own_output_vals, other_output_vals = [], []

    for trial in trials_info:
        tr_id = trial['trial']
        y, z = trial['y'], trial['z']
        total_distance = np.sqrt(y**2 + z**2)
        images = trial['images']

        if tr_id in label_dict:
            label_own, label_other = label_dict[tr_id]
            if label_own < up_limit and label_other < up_limit:
                # Separate own/other images once
                own_images = [img for img in images if img.startswith('1_')]
                img_map = {img[img.find('frame')+5:img.find('.jpg')]: img for img in images}

                for j, own_img in enumerate(own_images):
                    frame_no = own_img[own_img.find('frame')+5:own_img.find('.jpg')]
                    other_img = img_map.get(frame_no)
                    if other_img is None:
                        continue  # skip if matching other frame not found

                    # Own
                    own_gp_val = float(own_img[own_img.find('gp_')+3 : own_img.find('_fr')])
                    own_total.append(total_distance)
                    own_output_p.append(label_own + offset)
                    own_selected_all_names.append(os.path.join(trial['path'], own_img))
                    own_grip_posi_num.append(own_gp_val)
                    own_grip_vel_num.append(2*(random.random()-0.5))
                    other_grip_vel_num.append(2*(random.random()-0.5))
                    own_index.append(j)

                    # Other
                    other_gp_val = float(other_img[other_img.find('gp_')+3 : other_img.find('_fr')])
                    other_total.append(total_distance)
                    other_output_p.append(label_other + offset)
                    other_selected_all_names.append(os.path.join(trial['path'], other_img))
                    other_grip_posi_num.append(other_gp_val)
                    other_index.append(images.index(other_img))

                    # Save for regression
                    own_total_vals.append(total_distance)
                    other_total_vals.append(total_distance)
                    own_output_vals.append(label_own + offset)
                    other_output_vals.append(label_other + offset)
        else:
            # Trials without labels
            for j, img in enumerate(images):
                gp_val = float(img[img.find('gp_')+3 : img.find('_fr')])
                total_distance = np.sqrt(y**2 + z**2)
                rand_vel_1, rand_vel_2 = 2*(random.random()-0.5), 2*(random.random()-0.5)

                if img.startswith('1_'):
                    own_total.append(total_distance)
                    own_selected_all_names.append(os.path.join(trial['path'], img))
                    own_grip_posi_num.append(gp_val)
                    own_grip_vel_num.append(rand_vel_1)
                    other_grip_vel_num.append(rand_vel_2)
                    own_index.append(j)
                    own_output_p.append(0)  # placeholder, will update
                else:
                    other_total.append(total_distance)
                    other_selected_all_names.append(os.path.join(trial['path'], img))
                    other_grip_posi_num.append(gp_val)
                    other_index.append(j)
                    other_output_p.append(0)  # placeholder, will update

    # Fit linear regressors
    if own_total_vals:
        own_linear_regressor = LinearRegression().fit(np.array(own_total_vals).reshape(-1, 1), np.array(own_output_vals).reshape(-1, 1))
        other_linear_regressor = LinearRegression().fit(np.array(other_total_vals).reshape(-1, 1), np.array(other_output_vals).reshape(-1, 1))
        # Update placeholders for trials without labels
        for i in range(len(own_output_p)):
            if own_output_p[i] == 0:
                own_output_p[i] = own_linear_regressor.predict(np.array([[own_total[i]]]))[0,0]
        for i in range(len(other_output_p)):
            if other_output_p[i] == 0:
                other_output_p[i] = other_linear_regressor.predict(np.array([[other_total[i]]]))[0,0]

    return (own_index, other_index, own_total, other_total,
            own_selected_all_names, other_selected_all_names,
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
        # print("x own size: ", X_combined[0].size())
        # raise KeyError
        #X_combined[0] has size torch.Size([256, 3, 224, 224])
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

        # if ((epoch + 1) % 5 == 0):
        

        loss = F.mse_loss(output_1,y_own.float()) + F.mse_loss(output_2,y_other.float()) + F.mse_loss(final_y_own,final_output_own) + F.mse_loss(final_y_other,final_output_other)
        losses.append(loss.item())
        print("Agent 1 loss: ", F.mse_loss(output_1,y_own.float()).item() + F.mse_loss(final_y_own,final_output_own).item())
        print("Agent 2 loss: ", F.mse_loss(output_2,y_other.float()).item() + F.mse_loss(final_y_other,final_output_other).item())
        loss.backward()
        optimizer.step()
        epoch_count += 1

        # show information
        print("\033[92mTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\033[0m".format(
            epoch + 1, N_count, len(train_loader.dataset), 100. * (epoch_count + 1) / len(train_loader), loss.item()))
    # Return mean of losses in epoch (to match loss amount in validation)
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
            # print("own info 0 len: ", len(own_info[0]))
            # print("own info 1 len: ", len(own_info[1]))
            # print("other info len: ", len(other_info))
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
            # print("x own size: ", X_combined[0].size())
            # raise KeyError
            #X_combined[0] has size torch.Size([256, 3, 224, 224])
            X_own, y_own = X_own_image.to(device), y_own.to(device).view(-1, )
            X_other, y_other = X_other_image.to(device), y_other.to(device).view(-1, )
            #y_own and y_other are the p_slip of the trials

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
    #new_test_loss = np.mean(new_loss_list)
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
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

dataset_list = os.listdir(data_path)

own_index_ = []
other_index_ =[]
own_total_ = []
other_total_ = []
own_selected_all_names_ = []
other_selected_all_names_ = []
own_output_p_ = []
other_output_p_ = []
own_grip_posi_num_ = []
other_grip_posi_num_ = []
own_grip_vel_num_ = []
other_grip_vel_num_ = []

for i, val in enumerate(dataset_list):
    if 'npy' not in val:
        if val == 'empty':
            own_index,other_index,own_total,other_total,own_selected_all_names,other_selected_all_names,own_output_p,other_output_p,own_grip_posi_num,other_grip_posi_num,own_grip_vel_num, other_grip_vel_num = read_empty_data(data_path+'/'+val)
            own_index_.extend(own_index)
            other_index_.extend(other_index)
            own_total_.extend(own_total)
            other_total_.extend(other_total)
            own_selected_all_names_.extend(own_selected_all_names)
            other_selected_all_names_.extend(other_selected_all_names)
            own_output_p_.extend(own_output_p)
            other_output_p_.extend(other_output_p)
            own_grip_posi_num_.extend(own_grip_posi_num)
            other_grip_posi_num_.extend(other_grip_posi_num)
            own_grip_vel_num_.extend(own_grip_vel_num)
            other_grip_vel_num_.extend(other_grip_vel_num)
        else:
            own_index,other_index,own_total,other_total,own_selected_all_names,other_selected_all_names,own_output_p,other_output_p,own_grip_posi_num,other_grip_posi_num,own_grip_vel_num,other_grip_vel_num = read_data(data_path+'/'+val,data_path+'/'+val+'.npy')
            own_index_.extend(own_index)
            other_index_.extend(other_index)
            own_total_.extend(own_total)
            other_total_.extend(other_total)
            own_selected_all_names_.extend(own_selected_all_names)
            other_selected_all_names_.extend(other_selected_all_names)
            own_output_p_.extend(own_output_p)
            other_output_p_.extend(other_output_p)
            own_grip_posi_num_.extend(own_grip_posi_num)
            other_grip_posi_num_.extend(other_grip_posi_num)
            own_grip_vel_num_.extend(own_grip_vel_num)
            other_grip_vel_num_.extend(other_grip_vel_num)

# print("own_index length:", len(own_index_))
# print("other_index length:", len(other_index_))
# print("own_total length:", len(own_total_))
# print("other_total length:", len(other_total_))
# print("own_selected_all_names length:", len(own_selected_all_names_))
# print("other_selected_all_names length:", len(other_selected_all_names_))
# print("own_output_p length:", len(own_output_p_))
# print("other_output_p length:", len(other_output_p_))
# print("own_grip_posi_num length:", len(own_grip_posi_num_))
# print("other_grip_posi_num length:", len(other_grip_posi_num_))
# print("own_grip_vel_num length:", len(own_grip_vel_num_))
# print("other_grip_vel_num length:", len(other_grip_vel_num_))
# raise KeyError
own_pv_pair_list = zip(own_grip_posi_num_,own_grip_vel_num_)
own_frame_pair_list = zip(own_selected_all_names_,own_index_)
own_all_x_list = list(zip(own_frame_pair_list, own_pv_pair_list))
own_all_y_list = (own_output_p_)

other_pv_pair_list = zip(other_grip_posi_num_, other_grip_vel_num_)
other_frame_pair_list = zip(other_selected_all_names_,other_index_)
other_all_x_list = list(zip(other_frame_pair_list, other_pv_pair_list))
other_all_y_list = (other_output_p_)

# Combine own/other info for appropriate splitting
combined_all_x_list = []
for (own_fp, own_pv), (other_fp, other_pv) in zip(own_all_x_list, other_all_x_list):
    combined_all_x_list.append(((own_fp, own_pv), (other_fp, other_pv)))
# Now, combined_all_x_list contains [own_frame_pair, own_posi_vel, other_frame_pair, other_posi_vel]

combined_all_y_list = []
for (own, other) in zip(own_all_y_list, other_all_y_list):
    combined_all_y_list.append((own, other))
# Now, combined_all_y_list contains [own_output_p, other_output_p]

# train, test split
train_list, test_list, train_label, test_label = train_test_split(combined_all_x_list, combined_all_y_list, test_size=0.2, random_state=42)
# other_train_list, other_test_list, other_train_label, other_test_label = train_test_split(other_all_x_list, other_all_y_list, test_size=0.2, random_state=42)

train_set, valid_set = Dataset_LeTac(train_list, train_label, np.arange(1, 50, 1).tolist(), transform=transform), \
                       Dataset_LeTac(test_list, test_label, np.arange(1, 50, 1).tolist(), transform=transform)
# other_train_set, other_valid_setrain_list = Dataset_LeTac(other_train_list, other_train_label, np.arange(1, 25, 1).tolist(), transform=transform), \
#                        Dataset_LeTac(other_test_list, other_test_label, np.arange(1, 25, 1).tolist(), transform=transform)

# Changed 10 to 50 in Dataset_LeTac, although that list is never used?
train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)
# other_train_loader = data.DataLoader(other_train_set, **params)
# other_valid_loader = data.DataLoader(other_valid_set, **params)

# Create model
cnn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(device)
MPC_layer = MPClayer(nHidden = CNN_embed_dim, eps = eps, nStep = nStep, del_t = del_t).to(device)
letac_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
            list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
            list(cnn_encoder.fc3.parameters()) + list(MPC_layer.parameters())

optimizer = torch.optim.Adam(letac_params, lr=learning_rate, weight_decay=1e-4) # L2 regularizer of 1e-4

# Load checkpoint if exists
start_epoch = 0
checkpoint_path = 'multi_agent_model_epoch_100.pth'
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
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f'./multi_agent_model_epoch_{epoch+1}.pth'
            torch.save({
                'cnn_encoder_state_dict': cnn_encoder.state_dict(),
                'mpc_layer_state_dict': MPC_layer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
