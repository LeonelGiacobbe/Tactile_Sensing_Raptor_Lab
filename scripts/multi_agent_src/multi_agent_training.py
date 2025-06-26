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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# EncoderCNN architecture
CNN_hidden1, CNN_hidden2 = 128, 128 
CNN_embed_dim = 20  
res_size = 224       
dropout_p = 0.15  

# Training parameters
epochs = 100
batch_size = 256
learning_rate = 1e-4
eps = 1e-4
nStep = 15
del_t = 1/25

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
        own_grip_posi.append(f[(loc5 + 7): loc6])
        other_grip_posi.append(f[(loc6+9):len(f)])
        path_list.append(data_path+'/'+f+'/')
        sub_fnames = os.listdir(data_path+'/'+f)
        all_names.append(sub_fnames)
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
        for j in range(50): # because there's 50 images per subtrial (25 pairs)
            img = all_names[i][j]
            loc1 = img.find('gp_')
            loc2 = img.find('_fr')
            img[(loc1 + 3): loc2]
            
            if img.startswith('1_'): # Own gripper image
                rand_num = random.uniform(-0.5, 0.5)
                # Seems like every one of these evals in total would amount to 0?
                own_total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
                own_output_p.append(28.5+rand_num)
                own_selected_all_names.append(path_list[i]+img) # I don't think there's a need to differentiate between  
                own_grip_posi_num.append(eval(own_grip_posi[i]))
                own_grip_vel_num.append((28.5+rand_num-30)/3)
                # Velocities of both grippers should be related since they're
                # Acting on the same object, right? Maybe ask Dr. Sun?
                own_index.append(j)

                # Find matching frame to populate 'other' info
                fr_sloc = img.find('frame')
                fr_eloc = img.find('.jpg')
                frame_no = img[(fr_sloc + 5): fr_eloc]
                matching_img_path = glob.glob(os.path.join(path_list[i], f'2_*_frame{frame_no}.jpg'))
                filename_list = [os.path.basename(path) for path in matching_img_path]
                other_img = filename_list[0]
                # Seems like every one of these evals in total would amount to 0?
                other_total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
                other_output_p.append(28.5+rand_num)
                other_selected_all_names.append(path_list[i]+other_img)
                other_grip_posi_num.append(eval(other_grip_posi[i]))
                other_grip_vel_num.append((28.5+rand_num-30)/3)
                other_index.append(j)
    
    return own_index,other_index,own_total,other_total,own_selected_all_names,other_selected_all_names,own_output_p,other_output_p,own_grip_posi_num,other_grip_posi_num, own_grip_vel_num, other_grip_vel_num

def read_data(data_path,label_path,up_limit = 50,offset=0):
    # up_limit acts as a filter for trials where final gripper opening
    # was greater than up_limit. Those trials are ignored
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
        ys.append(f[(loc3 + 2): loc4])
        zs.append(f[(loc4 + 3): loc5])
        own_grip_posi.append(f[(loc5 + 3): loc6])
        other_grip_posi.append(f[(loc6+4):len(f)])
        path_list.append(data_path+'/'+f+'/')
        sub_fnames = os.listdir(data_path+'/'+f) # Contains list of all images in trial
        all_names.append(sub_fnames)
    label_dict = np.load(label_path,allow_pickle=True)
    label_dict = label_dict[()]
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
    
    for i in range(len(ys)):
        if trials[i] in label_dict.keys():
            # print("label_dict[trials[i]][0] : ", label_dict[trials[i]][0])
            # print("label_dict[trials[i]][0] type: ", type(label_dict[trials[i]][0]))
            if label_dict[trials[i]][0] < up_limit and label_dict[trials[i]][1] < up_limit:
                for j in range(50): # because there's 50 images per subtrial (25 pairs)
                    
                    # print("own_output_p: ", own_output_p)
                    # print("other_output_p: ",other_output_p)
                    img = all_names[i][j]
                    loc1 = img.find('gp_')
                    loc2 = img.find('_fr')
                    img[(loc1 + 3): loc2]
                    
                    if img.startswith('1_'): # Own gripper image
                        # Populate 'own' info
                        own_total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
                        own_output_p.append(label_dict[trials[i]][0]+offset)
                        
                        own_selected_all_names.append(path_list[i]+img)
                        own_grip_posi_num.append(eval(img[(loc1 + 3): loc2]))
                        rand_vel = 2*(random.random()-0.5)
                        # Velocities of both grippers should be related since they're
                        # Acting on the same object, right? Maybe ask Dr. Sun?
                        own_grip_vel_num.append(rand_vel)
                        other_grip_vel_num.append(rand_vel)
                        own_index.append(j)

                        # Find matching frame to populate 'other' info
                        fr_sloc = img.find('frame')
                        fr_eloc = img.find('.jpg')
                        frame_no = img[(fr_sloc + 5): fr_eloc]
                        matching_img_path = glob.glob(os.path.join(path_list[i], f'2_*_frame{frame_no}.jpg'))
                        filename_list = [os.path.basename(path) for path in matching_img_path]
                        other_img = filename_list[0]
                        other_loc1 = other_img.find('gp_')
                        other_loc2 = other_img.find('_fr')
                        other_total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
                        other_output_p.append(label_dict[trials[i]][1]+offset)
                        other_selected_all_names.append(path_list[i]+other_img)
                        other_grip_posi_num.append(eval(other_img[(other_loc1 + 3): other_loc2]))
                        other_index.append(j)
    own_linear_regressor = LinearRegression()
    own_linear_regressor.fit(np.array(own_total).reshape(-1, 1),np.array(own_output_p).reshape(-1, 1))
    # Same regressor but for 'other' gripper
    other_linear_regressor = LinearRegression()
    other_linear_regressor.fit(np.array(other_total).reshape(-1, 1),np.array(other_output_p).reshape(-1, 1))
    for i in range(len(ys)):
        if not(trials[i] in label_dict.keys()):
            for j in range(50): # because there's 50 images per subtrial (25 pairs)
                img = all_names[i][j]
                loc1 = img.find('gp_')
                loc2 = img.find('_fr')
                img[(loc1 + 3): loc2]
                
                if img.startswith('1_'): # Own gripper image
                    own_total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
                    own_selected_all_names.append(path_list[i]+img)
                    own_grip_posi_num.append(eval(img[(loc1 + 3): loc2]))
                    rand_vel = 2*(random.random()-0.5)
                    # Velocities of both grippers should be related since they're
                    # Acting on the same object, right? Maybe ask Dr. Sun?
                    own_grip_vel_num.append(rand_vel)
                    other_grip_vel_num.append(rand_vel)
                    own_index.append(j)
                else:
                    other_total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
                    other_selected_all_names.append(path_list[i]+img)
                    other_grip_posi_num.append(eval(img[(loc1 + 3): loc2]))
                    other_index.append(j)
                
                
                own_output_p.append(own_linear_regressor.predict((np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i]))).reshape(-1, 1))[0,0])
                other_output_p.append(other_linear_regressor.predict((np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i]))).reshape(-1, 1))[0,0])
    
    return own_index,other_index,own_total,other_total,own_selected_all_names,other_selected_all_names, own_output_p,other_output_p, own_grip_posi_num, other_grip_posi_num, own_grip_vel_num, other_grip_vel_num
        
def train(model, device, own_train_loader, other_train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, MPC_layer= model
    cnn_encoder.train()
    MPC_layer.train()
    losses = []
    scores = []

    N_count = 0 
    epoch_count = 0
    for (X_own, y_own), (X_other, y_other) in zip(own_train_loader, other_train_loader):
        # distribute data to device
        own_gripper_p = X_own[1][0].to(device)
        own_gripper_v = X_own[1][1].to(device)
        other_gripper_p = X_other[1][0].to(device)
        other_gripper_v = X_other[1][1].to(device) 
        # other_gripper_v = X[1][2].to(device)
        X_own, y_own = X_own[0].to(device), y_own.to(device).view(-1, )
        X_other, y_other = X_other[0].to(device), y_other.to(device).view(-1, )
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

        if ((epoch + 1) % 5 == 0):
            print(f"Agent 1 loss per time step:")
            for t in range(output_1.size(1)):
                step_loss = F.mse_loss(output_1[:, t], y_own[:, t].float()).item()
                print(f"  Step {t}: {step_loss:.6f}")

            print(f"Agent 2 loss per time step:")
            for t in range(output_2.size(1)):
                step_loss = F.mse_loss(output_2[:, t], y_other[:, t].float()).item()
                print(f"  Step {t}: {step_loss:.6f}")
        
        loss = F.mse_loss(output_1,y_own.float()) + F.mse_loss(output_2,y_other.float()) + F.mse_loss(final_y_own,final_output_own) + F.mse_loss(final_y_other,final_output_other)
        losses.append(loss.item())
        print("Agent 1 loss: ", F.mse_loss(output_1,y_own.float()).item() + F.mse_loss(final_y_own,final_output_own).item())
        print("Agent 2 loss: ", F.mse_loss(output_2,y_other.float()).item() + F.mse_loss(final_y_other,final_output_other).item())
        # print("Agent 1 only final loss: ", F.mse_loss(final_y_own,final_output_own).item())
        # print("Agent 2 only final loss: ",  F.mse_loss(final_y_other,final_output_other).item())
        loss.backward()

        for name, param in MPC_layer.named_parameters():
            if param.grad is not None:
                pass
            else:
                print(f"{name} is not being learned")
        
        optimizer.step()
        epoch_count += 1

        # show information
        # print("Agent new 1 loss: ", F.mse_loss(output_1,y_own.float()).item())
        # print("Agent nw 2 loss: ", F.mse_loss(output_2,y_other.float()).item())
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch + 1, N_count, len(own_train_loader.dataset), 100. * (epoch_count + 1) / len(own_train_loader), loss.item()))
    # Return mean of losses in epoch (to match loss amount in validation)
    return np.mean(losses)

def validation(model, device, own_test_loader, other_test_loader):
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
        for (X_own, y_own), (X_other, y_other) in zip(own_test_loader, other_test_loader):
            # distribute data to device
            own_gripper_p = X_own[1][0].to(device)
            own_gripper_v = X_own[1][1].to(device)

            other_gripper_p = X_other[1][0].to(device)
            other_gripper_v = X_other[1][1].to(device)
            
            X_own, y_own = X_own[0].to(device), y_own.to(device).view(-1, )
            X_other, y_other = X_other[0].to(device), y_other.to(device).view(-1, )

            #y_own and y_other are the p_slip of the trials
            own_output = cnn_encoder(X_own)
            other_output = cnn_encoder(X_other) 
            
            # print("own_output size: ", own_output.size())
            # print("other_output size: ", other_output.size())
            # print("own gripper pos size: ", own_gripper_p.size())
            # print("own gripper v size: ", own_gripper_v.size())
            # print("other gripper pos size: ", other_gripper_p.size())
            # print("other gripper v size: ", other_gripper_v.size())
            
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
            print("Agent 1 only final loss: ", F.mse_loss(final_y_own,final_output_own).item())
            print("Agent 2 only final loss: ",  F.mse_loss(final_y_other,final_output_other).item())
            loss_list.append(loss.item())
            test_loss += F.mse_loss(output_1,y_own.float()).item() + F.mse_loss(output_2,y_other.float()).item()     
            y_pred_own = output_1.max(1, keepdim=True)[1] 
            y_pred_other = output_2.max(1, keepdim=True)[1] 
            all_y_own.extend(y_own)
            all_y_other.extend(y_other)

            all_y_pred_own.extend(y_pred_own)
            all_y_pred_other.extend(y_pred_other)
    new_test_loss = np.mean(new_loss_list)
    test_loss = np.mean(loss_list)
    all_y_own = torch.stack(all_y_own, dim=0)
    all_y_other = torch.stack(all_y_other, dim=0)
    all_y_pred_own = torch.stack(all_y_pred_own, dim=0)
    all_y_pred_other = torch.stack(all_y_pred_other, dim=0)
    print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(len(all_y_own), test_loss))
    # print('\nTest set ({:d} samples): Average NEW loss: {:.4f}\n'.format(len(all_y_own), new_test_loss))
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


# train, test split
own_train_list, own_test_list, own_train_label, own_test_label = train_test_split(own_all_x_list, own_all_y_list, test_size=0.2, random_state=42)
other_train_list, other_test_list, other_train_label, other_test_label = train_test_split(other_all_x_list, other_all_y_list, test_size=0.2, random_state=42)

own_train_set, own_valid_set = Dataset_LeTac(own_train_list, own_train_label, np.arange(1, 25, 1).tolist(), transform=transform), \
                       Dataset_LeTac(own_test_list, own_test_label, np.arange(1, 25, 1).tolist(), transform=transform)
other_train_set, other_valid_set = Dataset_LeTac(other_train_list, other_train_label, np.arange(1, 25, 1).tolist(), transform=transform), \
                       Dataset_LeTac(other_test_list, other_test_label, np.arange(1, 25, 1).tolist(), transform=transform)

# Changed 10 to 50 in Dataset_LeTac, although that list is never used?

own_train_loader = data.DataLoader(own_train_set, **params)
own_valid_loader = data.DataLoader(own_valid_set, **params)
other_train_loader = data.DataLoader(other_train_set, **params)
other_valid_loader = data.DataLoader(other_valid_set, **params)

# Create model
cnn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(device)
MPC_layer = MPClayer(nHidden = CNN_embed_dim, eps = eps, nStep = nStep, del_t = del_t).to(device)
letac_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
            list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
            list(cnn_encoder.fc3.parameters()) + list(MPC_layer.parameters())

optimizer = torch.optim.Adam(letac_params, lr=learning_rate, weight_decay=1e-4) # L2 regularizer of 1e-4

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',          # because you're minimizing loss
    factor=0.1,          # reduce LR by 10x
    patience=5,          # wait 5 epochs with no improvement
    threshold=.5,      # minimal change to be considered an improvement
    #verbose=True         # prints when LR is reduced
)

# Load checkpoint if exists
start_epoch = 0
checkpoint_path = '40_soft_60_hard_v3_checkpoint_epoch_30.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    cnn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
    MPC_layer.load_state_dict(checkpoint['mpc_layer_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # +1 because we want to start from the next epoch
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    print("Warning, could not find checkpoint model")

# start training - modify range to start from start_epoch
for epoch in range(start_epoch, epochs):
    with open("loss_log.csv", mode='a') as f:  # 'a' mode appends rather than overwrites
        valid_loss = validation([cnn_encoder, MPC_layer], device, own_valid_loader, other_valid_loader)
        scheduler.step(valid_loss)
        training_loss = train([cnn_encoder, MPC_layer], device, own_train_loader, other_train_loader, optimizer, epoch)
        writer = csv.writer(f)
        writer.writerow([epoch+1, valid_loss, training_loss])
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'./40_soft_60_hard_v3_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'cnn_encoder_state_dict': cnn_encoder.state_dict(),
                'mpc_layer_state_dict': MPC_layer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

