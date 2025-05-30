import os
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
CNN_embed_dim = 25  
res_size = 224       
dropout_p = 0.15  

# Training parameters
epochs = 50 
batch_size = 256
learning_rate = 1e-4
eps = 1e-4
nStep = 20
del_t = 1/25

data_path = "../data_collection/dataset"

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])])
def read_empty_data(data_path):
    """
    For images when nothing is touching the sensor
    pn should be 30 (gp_* in dir and images)
    
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
        ys.append(f[(loc3 + 2): loc4])
        zs.append(f[(loc4 + 3): loc5])
        own_grip_posi.append(f[(loc5 + 3): loc6])
        other_grip_posi.append(f[(loc6+4):len(f)])
        path_list.append(data_path+'/'+f+'/')
        sub_fnames = os.listdir(data_path+'/'+f)
        all_names.append(sub_fnames)
    selected_all_names = []
    own_output_p = []
    own_grip_posi_num = []
    other_grip_posi_num = []
    total = []
    index = []
    own_grip_vel_num = []
    other_grip_vel_num = []
    for i in range(len(ys)):
        for j in range(10):
            own_grip_posi_num.append(eval(own_grip_posi[i]))
            other_grip_posi_num.append(eval(other_grip_posi[i]))
            total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
            img = all_names[i][j]
            selected_all_names.append(path_list[i]+img) # I don't think there's a need to differentiate between
            # Gripper 1 and gripper 2 images? Not sure
            index.append(j)
            rand_num = 1*(random.random()-0.5)
            own_output_p.append(28.5+rand_num)
            own_grip_vel_num.append((28.5+rand_num-30)/3)
            other_grip_vel_num.append((28.5+rand_num-30)/3)
    # For empty data, I think it's a good idea to keep own and other output_p equal. Maybe talk to Dr. Sun about this?
    other_output_p = own_output_p
    return index,total,selected_all_names,own_output_p,other_output_p,own_grip_posi_num,other_grip_posi_num, own_grip_vel_num, other_grip_vel_num

def read_data(data_path,label_path,up_limit = 30,offset=0):
    # up_limit acts as a filer for trials where final gripper opening
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
    selected_all_names = []
    own_output_p = []
    other_output_p = []
    own_grip_posi_num = []
    other_grip_posi_num = []
    total = []
    index = []
    own_grip_vel_num = []
    other_grip_vel_num = []
    for i in range(len(ys)):
        if trials[i] in label_dict.keys():
            # print("label_dict[trials[i]][0] : ", label_dict[trials[i]][0])
            # print("label_dict[trials[i]][0] type: ", type(label_dict[trials[i]][0]))
            if label_dict[trials[i]][0] < up_limit and label_dict[trials[i]][1] < up_limit:
                for j in range(50): # because there's 50 images per subtrial (25 pairs)
                    own_output_p.append(label_dict[trials[i]][0]+offset)
                    other_output_p.append(label_dict[trials[i]][1]+offset)
                    # print("own_output_p: ", own_output_p)
                    # print("other_output_p: ",other_output_p)
                    total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
                    img = all_names[i][j]
                    loc1 = img.find('gp_')
                    loc2 = img.find('_fr')
                    img[(loc1 + 3): loc2]
                    selected_all_names.append(path_list[i]+img)
                    if img.startswith('1_'): # Own gripper image
                        own_grip_posi_num.append(eval(img[(loc1 + 3): loc2]))
                        rand_vel = 2*(random.random()-0.5)
                        # Velocities of both grippers should be related since they're
                        # Acting on the same object, right? Maybe ask Dr. Sun?
                        own_grip_vel_num.append(rand_vel)
                        other_grip_vel_num.append(rand_vel)
                    else:
                        other_grip_posi_num.append(eval(img[(loc1 + 3): loc2]))
                        
                    
                    index.append(j)
                    
    own_linear_regressor = LinearRegression()
    own_linear_regressor.fit(np.array(total).reshape(-1, 1),np.array(own_output_p).reshape(-1, 1))
    # Same regressor but for 'other' gripper
    other_linear_regressor = LinearRegression()
    other_linear_regressor.fit(np.array(total).reshape(-1, 1),np.array(other_output_p).reshape(-1, 1))
    for i in range(len(ys)):
        if not(trials[i] in label_dict.keys()):
            for j in range(10):
                img = all_names[i][j]
                loc1 = img.find('gp_')
                loc2 = img.find('_fr')
                img[(loc1 + 3): loc2]
                selected_all_names.append(path_list[i]+img)
                if img.startswith('1_'): # Own gripper image
                    own_grip_posi_num.append(eval(img[(loc1 + 3): loc2]))
                    rand_vel = 2*(random.random()-0.5)
                    # Velocities of both grippers should be related since they're
                    # Acting on the same object, right? Maybe ask Dr. Sun?
                    own_grip_vel_num.append(rand_vel)
                    other_grip_vel_num.append(rand_vel)
                else:
                    other_grip_posi_num.append(eval(img[(loc1 + 3): loc2]))
                total.append(np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i])))
                index.append(j)
                own_output_p.append(own_linear_regressor.predict((np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i]))).reshape(-1, 1))[0,0])
                other_output_p.append(other_linear_regressor.predict((np.sqrt(eval(ys[i])*eval(ys[i])+eval(zs[i])*eval(zs[i]))).reshape(-1, 1))[0,0])
    return index,total,selected_all_names,own_output_p,other_output_p, own_grip_posi_num, other_grip_posi_num, own_grip_vel_num, other_grip_vel_num

def train(model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, MPC_layer= model
    cnn_encoder.train()
    MPC_layer.train()
    losses = []
    scores = []

    N_count = 0 
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        own_gripper_p = X[1][0].to(device)
        own_gripper_v = X[1][1].to(device)
        other_gripper_p = X[1][0].to(device) # NEEDS MODIFYING, PLACEHOLDER
        other_gripper_v = X[1][1].to(device) # NEEDS MODIFYING, PLACEHOLDER
        # other_gripper_v = X[1][2].to(device)
        X, y = X[0].to(device), y.to(device).view(-1, )
        N_count += X.size(0)
        optimizer.zero_grad()
        own_output = cnn_encoder(X)
        other_output = cnn_encoder(X) # NEEDS MODIFYING, PLACEHOLDER
        output = MPC_layer(own_output, other_output, own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v) # Need to add other agent's velocity 
        y= y.unsqueeze(1).expand(X.size(0), output.size(1))
        final_y = y[:,(output.size(1)-1)]*3
        final_output = output[:,(output.size(1)-1)]*3
        loss = F.mse_loss(output,y.float()) + F.mse_loss(final_y,final_output)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))

def validation(model, device, optimizer, test_loader):
    cnn_encoder, MPC_layer= model
    cnn_encoder.eval()
    MPC_layer.eval()
    test_loss = 0
    loss_list = []
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            own_gripper_p = X[1][0].to(device)
            own_gripper_v = X[1][1].to(device)
            other_gripper_p = X[1][0].to(device) # NEEDS MODIFYING, PLACEHOLDER
            other_gripper_v = X[1][1].to(device) # NEEDS MODIFYING, PLACEHOLDER
            # other_gripper_v = X[1][2].to(device)
            X, y = X[0].to(device), y.to(device).view(-1, )
            own_output = cnn_encoder(X)
            other_output = cnn_encoder(X) # NEEDS MODIFYING, PLACEHOLDER
            # print("own_output size: ", own_output.size())
            # print("other_output size: ", other_output.size())
            # print("own gripper pos size: ", own_gripper_p.size())
            # print("own gripper v size: ", own_gripper_v.size())
            # print("other gripper pos size: ", other_gripper_p.size())
            # print("other gripper v size: ", other_gripper_v.size())
            output = MPC_layer(own_output, other_output, own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v) # need to add other_gripper_v
            y= y.unsqueeze(1).expand(X.size(0), output.size(1))
            final_y = y[:,(output.size(1)-1)]*3
            final_output = output[:,(output.size(1)-1)]*3
            loss = F.mse_loss(output,y.float()) + F.mse_loss(final_y,final_output)
            loss_list.append(loss.item())
            test_loss += F.mse_loss(output,y.float()).item()      
            y_pred = output.max(1, keepdim=True)[1] 
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss = np.mean(loss_list)
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(len(all_y), test_loss))


use_cuda = torch.cuda.is_available()                  
device = torch.device("cuda" if use_cuda else "cpu")   

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

dataset_list = os.listdir(data_path)

index_ = []
total_ = []
selected_all_names_ = []
own_output_p_ = []
other_output_p_ = []
own_grip_posi_num = []
other_grip_posi_num = []
own_grip_vel_num = []
other_grip_vel_num = []

for i, val in enumerate(dataset_list):
    if 'npy' not in val:
        if val == 'empty':
            index,total,selected_all_names,own_output_p,other_output_p,own_grip_posi_num,other_grip_posi_num, own_grip_vel_num, other_grip_vel_num = read_empty_data(data_path+'/'+val)
            index_.extend(index)
            total_.extend(total)
            selected_all_names_.extend(selected_all_names)
            own_output_p_.extend(own_output_p)
            other_output_p_.extend(other_output_p)
            own_grip_posi_num.extend(own_grip_posi_num)
            other_grip_posi_num.extend(other_grip_posi_num)
            own_grip_vel_num.extend(own_grip_vel_num)
            other_grip_vel_num.extend(other_grip_vel_num)
        else:
            if 'gel' in val or 'hard_rubber' in val:
                index,total,selected_all_names,own_output_p,other_output_p,own_grip_posi_num, other_grip_posi_num, own_grip_vel_num, other_grip_vel_num = read_data(data_path+'/'+val,data_path+'/'+val+'.npy',up_limit = 11.5)
            else:
                index,total,selected_all_names,own_output_p,other_output_p,own_grip_posi_num, other_grip_posi_num, own_grip_vel_num, other_grip_vel_num = read_data(data_path+'/'+val,data_path+'/'+val+'.npy')
            index_.extend(index)
            total_.extend(total)
            selected_all_names_.extend(selected_all_names)
            own_output_p_.extend(own_output_p)
            other_output_p_.extend(other_output_p)
            own_grip_posi_num.extend(own_grip_posi_num)
            other_grip_posi_num.extend(other_grip_posi_num)
            own_grip_vel_num.extend(own_grip_vel_num)
            other_grip_vel_num.extend(other_grip_vel_num)
'''
Now, selected_all_names contains all images (of both sensors), and index_
contains the total amount of pairs (0 to 49) * num of subtrials (3 * num of trials)
Since we're separating own and other posi and vel, it might be smart to separate
selected_all_names and maybe index_(?) and have two versions, one for each gripper?
We'll need the info because of the new MPC Layer architecture

For now, I'll try keeping the data together (for own and other) and we'll see

'''

# Create a combined structure that includes both agents' data
pv_pair_list = list(zip(
    zip(own_grip_posi_num, own_grip_vel_num),  # Own agent's (posi, vel)
    zip(other_grip_posi_num, other_grip_vel_num)  # Other agent's (posi, vel)
))
frame_pair_list = list(zip(selected_all_names_, index_))
# Now each element contains (frame_info, (own_pv, other_pv))
all_x_list = list(zip(frame_pair_list, pv_pair_list))       
all_y_list = list(zip(own_output_p_, other_output_p_))

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_x_list, all_y_list, test_size=0.2, random_state=42)
train_set, valid_set = Dataset_LeTac(train_list, train_label, np.arange(1, 10, 1).tolist(), transform=transform), \
                       Dataset_LeTac(test_list, test_label, np.arange(1, 10, 1).tolist(), transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# Create model
cnn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(device)
MPC_layer = MPClayer(nHidden = CNN_embed_dim, eps = eps, nStep = nStep, del_t = del_t).to(device)
letac_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
            list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
            list(cnn_encoder.fc3.parameters()) + list(MPC_layer.parameters())

optimizer = torch.optim.Adam(letac_params, lr=learning_rate)

# start training
for epoch in range(epochs):
    validation([cnn_encoder, MPC_layer], device, optimizer, valid_loader)
    train([cnn_encoder, MPC_layer], device, train_loader, optimizer, epoch)
    
