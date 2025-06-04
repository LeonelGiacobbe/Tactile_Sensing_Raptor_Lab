import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import  Variable
from torch.nn.parameter import Parameter
from qpth.qp import QPFunction

# Helper function to stack zeros
def zeors_hstack_help(vec, n, size_row, size_col):
    combo = vec
    single = Variable(torch.zeros(size_row,size_col).cuda())
    for i in range(n-1):
        combo = torch.hstack((combo,single))
    return combo

#Dataloader
class Dataset_LeTac(data.Dataset):
    def __init__(self, folders_pv_pair, labels, frames, transform=None):
        self.labels = labels
        self.folders = list(np.array(tuple(folders_pv_pair),dtype=object)[:,0])
        self.pv_pairs = list(np.array(tuple(folders_pv_pair),dtype=object)[:,1])
        self.transform = transform
        self.frames = frames 

    def __len__(self):
        return len(self.folders)

    def read_images(self, selected_folder, use_transform):
        image = Image.open(selected_folder[0])
        if use_transform is not None:
            image = use_transform(image)
        return image

    def __getitem__(self, index):
        folder = self.folders[index]
        pv_pair = self.pv_pairs[index]
        x = self.read_images(folder, self.transform)    
        x = (x,tuple(pv_pair))
        y = torch.FloatTensor([self.labels[index]])            
        return x, y

# 2D CNN encoder using ResNet-152 pretrained.
class ResCNNEncoder(nn.Module):
    def __init__(self, hidden1=512, hidden2=512, dropP=0.3, outputDim=25):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.hidden1, self.hidden2 = hidden1, hidden2
        self.dropP = dropP

        resnet = models.resnet152(pretrained=True)
        # Delete the last fc layer.
        modules = list(resnet.children())[:-1]      
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1, momentum=0.01)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2, momentum=0.01)
        self.fc3 = nn.Linear(hidden2, outputDim)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x[:, :, :, :])  
            x = x.view(x.size(0), -1)            
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropP, training=self.training)
        x = self.fc3(x)
        return x


# Differentiable MPC layer
class MPClayer(nn.Module):
    def __init__(self, nHidden = 25, eps = 1e-4, nStep = 20, del_t = 1/60):
        super(MPClayer, self).__init__()

        self.Pq = 5
        self.Qv = 200 # Increase for quicker convergence
        self.Qa = 1 # Increase for quicker convergence
        self.nHidden = nHidden
        self.eps = eps
        self.nStep = nStep
        self.del_t = del_t
    
        # A matrix
        self.A_eye = Variable(torch.eye(self.nHidden).cuda()) # Identity matrix
        self.Af_zero = Variable(torch.zeros(self.nHidden,1).cuda())
        self.Af= Parameter(torch.rand(self.nHidden,1).cuda()) # Learned parameter
        Ap_right_temp = Variable(torch.from_numpy(np.array([ # Referred to as Ag in the paper
        [1,     self.del_t],
        [0,     1]
        ])).float().cuda())
        self.Ap_right = torch.hstack((Variable(0*torch.ones(2, self.nHidden).cuda()),Ap_right_temp)) # (0 ^ 2xM, page 4 of paper) and Ag in the paper

        # B matrix (own control input)
        Bg = Variable(torch.from_numpy(np.array([ # Also called Bg in the paper
        [0.5*self.del_t*self.del_t],
        [self.del_t]
        ])).float().cuda())
        self.B_zero = Variable(torch.zeros(self.nHidden, 1).cuda())
        self.B0 = torch.vstack((self.B_zero,Bg))

        # Need to expand B0 for coupling of agents
        self.B0 = torch.block_diag(self.B0, self.B0)

        # These are the matrices related to the "other" agent
        # Define Cf_zero here, same shape as Af_zero above.
        self.Cf_zero = Variable(torch.zeros(self.nHidden,1).cuda())
        # Define Cf here, the learned parameter for the other agent
        self.Cf = Parameter(torch.rand(self.nHidden, 1).cuda()) # Learned parameter
        # The other agent's velocity affects the hidden state (through Cf) and the velocity (through Cp)
        Cp_temp = Variable(torch.from_numpy(np.array([
            [0.5 * self.del_t],  # Other agent's velocity affects position with dampening
            # Or change to self.del_t**2, just as in Ap_right_temp
            [0]          # Does not directly affect velocity
        ])).float().cuda())
        # Not sure if the decisions for Cp_temp above are right, but they're easily changeable
        self.Cp = torch.vstack((self.Cf_zero, Cp_temp))
        

        # Weights
        self.Lq = Parameter(torch.tril(torch.rand(self.nHidden, self.nHidden).cuda()))
        self.Lq_coupling = Parameter(torch.tril(torch.rand(self.nHidden, self.nHidden).cuda()))
        self.R0 = self.Qa*Variable(torch.eye(1).cuda())
        self.Q0_right_down = Variable(torch.from_numpy(np.array([
        [0,     0],
        [0,     self.Qv]
        ])).float().cuda())
        self.Q0_down = Variable(torch.zeros(2,self.nHidden).cuda())
        self.Q0_right = Variable(torch.zeros(self.nHidden,2).cuda())

        # No constraints during training
        self.G = Variable(torch.zeros(2 * self.nStep,2 * self.nStep).cuda())
        self.h = Variable(torch.zeros(2 * self.nStep,1).cuda())


    def forward(self, x1, x2, own_gripper_p, own_gripper_v, other_gripper_p, other_gripper_v):
        """
            In the multi-agent approach, an agent uses its own tactile embeddings,
            position and velocity, and also uses the same information from the other gripper

        """
        nBatch = x1.size(0)
        nHiddenExpand = 2 * (self.nHidden + 2)

        # Single Q in cost function
        Q0 = self.Lq.mm(self.Lq.t()) + self.eps*Variable(torch.eye(self.nHidden)).cuda()
        Q0 = torch.hstack((Q0,self.Q0_right))
        Q0 = torch.vstack((Q0,torch.hstack((self.Q0_down,self.Q0_right_down))))
        # Q0 is equivalent to Qf in the paper
        # According to the paper, Q0 must be positive semi-definite
        
        Q_coupling = self.Lq_coupling.mm(self.Lq_coupling.t()) + self.eps*Variable(torch.eye(self.nHidden)).cuda()
        Q_coupling = torch.hstack((Q_coupling, self.Q0_right))
        # If Q_coupling is not scaled, then the off-diagonal coupling is too strong and the matrix is not SPD
        Q_coupling = torch.vstack((Q_coupling,torch.hstack((self.Q0_down,self.Q0_right_down))))  * 0.25

        # Add coupling terms on off-diagonal sections, combine for bigger Q matrix
        Top_Q0 = torch.hstack((Q0, Q_coupling)) # Top right
        Bottom_Q0 = torch.hstack((Q_coupling.t(), Q0)) # Bottom left
        Q0_combined = torch.vstack((Top_Q0, Bottom_Q0))

        # Now Q0 is guaranteed to be symmetric because we add Q_coupling to top right and its transpose to bottom left
        # Stacked Q
        Q0_stack = Q0_combined.unsqueeze(0).expand(self.nStep-1, nHiddenExpand, nHiddenExpand)
        Q0_final = self.Pq*Q0_combined.unsqueeze(0).expand(1, nHiddenExpand, nHiddenExpand)
        Q0_stack = torch.vstack((Q0_stack,Q0_final))
        Q_dia =  torch.block_diag(*Q0_stack).cuda() # Contains Q0_combined (or Qf in paper) for each time step
        
        # Stacked R
        R0_stack = self.R0.unsqueeze(0).expand(2 * self.nStep, 1, 1) # Qa stack
        R_dia =  torch.block_diag(*R0_stack).cuda() #Qa diagonal

        # Model computing of own dynamics 
        A0 = torch.vstack((torch.hstack((torch.hstack((self.A_eye,self.Af_zero)),self.Af)),self.Ap_right))
        
        # Expaning dynamics matrix for both agents
        coupling_matrix = torch.zeros_like(A0).cuda()  # (M+2) x (M+2) zeros

        coupling_matrix[:self.nHidden, -1] = self.Cf.squeeze(-1) # Add Cf in last column (velocity coupling) like Dr. Sun's graph
        
        # print("cf: ", self.Cf.squeeze())
        # print("cf shape: ", self.Cf.size())
        # raise KeyError
        # Printing coupling matrix looks fine

        # Insert coupling (top-right and bottom-left corners of A0)
        Top_A0 = torch.hstack((A0, coupling_matrix))
        Bottom_A0 = torch.hstack((coupling_matrix, A0))
        A0 = torch.vstack((Top_A0, Bottom_A0))
        
        # T_ is the state transition matrix, (how the initial state evolves if there are no control inputs)
        T_ = A0
        
        temp = A0
        for i in range(self.nStep-1):
            temp = torch.mm(temp,A0)
            T_ = torch.vstack((T_,temp))
        I = Variable(torch.eye(nHiddenExpand).cuda())
        row_single = zeors_hstack_help(I, self.nStep, nHiddenExpand, nHiddenExpand)
        AN_ = row_single
        for i in range(self.nStep-1):
            AN = I
            row_single = I
            for j in range(i+1):
                AN = torch.mm(A0,AN)
                row_single = torch.hstack((AN,row_single))
            row_single = zeors_hstack_help(row_single, self.nStep-i-1, nHiddenExpand, nHiddenExpand)
            AN_=torch.vstack((AN_, row_single))
        B0_stack = self.B0.unsqueeze(0).expand(self.nStep, nHiddenExpand, 2)
        B_dia =  torch.block_diag(*B0_stack)
        S_ = torch.mm(AN_,B_dia)
        
        
        Q_final = 2*(R_dia+(torch.mm(S_.t(),Q_dia)).mm(S_))+ self.eps*Variable(torch.eye(2 * self.nStep)).cuda() # f_k^T @ Qf @ f_k
        Q_batch = Q_final.unsqueeze(0).expand(nBatch, 2 * self.nStep, 2 * self.nStep)
        p_final = 2*torch.mm(T_.t(),torch.mm(Q_dia,S_)) # f_n^T @ Q_f @ 
        p_batch = p_final.unsqueeze(0).expand(nBatch, nHiddenExpand, 2 * self.nStep)

        # Prepare input state
        own_gripper_p = own_gripper_p.reshape([nBatch,1]).float()
        own_gripper_v = own_gripper_v.reshape([nBatch,1]).float()
        gripper_state_1 = torch.hstack((own_gripper_p,own_gripper_v))
        
        x1 = torch.hstack((x1,gripper_state_1))
        x1 = x1.reshape([nBatch,1,self.nHidden + 2])
        

        # Prepare input state for other agent
        other_gripper_p = other_gripper_p.reshape([nBatch,1]).float()
        other_gripper_v = other_gripper_v.reshape([nBatch,1]).float()
        gripper_state_2 = torch.hstack((other_gripper_p,other_gripper_v))
        
        x2 = torch.hstack((x2,gripper_state_2))
        x2 = x2.reshape([nBatch,1,self.nHidden + 2])

        # Now combine state inputs
        # print("x1 size: ", x1.size())
        # print("x2 size: ", x2.size())
        x_combined = torch.cat((x1, x2), dim=2) # Not sure about this part. dim=2 is the only one that causes no error, but might still be conceptually wrong        
        
        # Calculate other gripper's effect on hidden and own velocity
        # Initialize the batch effect tensor directly
        other_effect_batch = torch.zeros(nBatch, nHiddenExpand, 1).cuda()

        # Process batch elements
        hidden_effect = (other_gripper_v @ self.Cf.t()).unsqueeze(-1)  # (nBatch, 1) @ (1, nHidden) -> (nBatch, nHidden, 1)
        state_effect = (other_gripper_v @ self.Cp.t()).unsqueeze(-1)    # (nBatch, 1) @ (1, nHidden) -> (nBatch, nHidden, 1)

        other_effect_batch = torch.zeros(nBatch, nHiddenExpand, 1).cuda()
        other_effect_batch[:, :self.nHidden, :] = hidden_effect
        other_effect_batch[:, self.nHidden + 2:, :] = state_effect

        # Now use other_effect_batch directly
        
        x_with_effect = x_combined + other_effect_batch.transpose(1, 2)
        # print(x1[1])
        # print(x2[1])
        # print(x_combined[1])
        # print(x_with_effect[1])
        
        # print("x combined size: ", x_combined.size())
        # print("x with effect size: ", x_with_effect.size()) # size is nBatch, nBatch, nHiddenExpand? // in working version, its nBatch,1,self.nHidden+2
       
        # print("p_batch size: ", p_batch.size()) # size is nBatch, nHiddenExpand, 2 * self.nStep // in working version, its nBatch, self.nHidden+2, self.nStep
    
        p_x0_batch = torch.bmm(x_with_effect, p_batch) # In og paper, the bmm is x and p_batch

        e = Variable(torch.Tensor())
        G = self.G.unsqueeze(0).expand(nBatch, 2 * self.nStep, 2 * self.nStep)
        h = self.h.unsqueeze(0).expand(nBatch, 2 * self.nStep, 1)

        # print("p_x0_batch size: ", p_x0_batch.size())
        p_x0_batch = p_x0_batch.reshape([nBatch,2 * self.nStep])
        h = h.reshape([nBatch, 2 * self.nStep])

        u = QPFunction(verbose=-1)(Q_batch, p_x0_batch, G, h, e, e)
        

        S_batch = S_.unsqueeze(0).expand(nBatch, self.nStep*(nHiddenExpand), 2 * self.nStep)
        
        T_batch = T_.unsqueeze(0).expand(nBatch, self.nStep*(nHiddenExpand), nHiddenExpand)

        # Include other agent's velocity effect in prediction for each step
        other_effect_expanded = other_effect_batch.repeat(1, self.nStep, 1)
        x_predict = torch.bmm(S_batch, u.reshape(nBatch, 2 * self.nStep, 1)) + torch.bmm(T_batch, x_combined.reshape(nBatch, nHiddenExpand, 1)) + other_effect_expanded
        
        """
        After all of the math, x_predict will have size of [nBatch, self.nStep * nHiddenExpand, 1]
        this contains the predictions of both "own" (0-440) and "other" (441-880)

        We do all the math keeping both agent's information in mind, but after the math, we only care about
        correctly outputting the sequences of the "own" gripper.
        
        So we resize x_predict to only include the predictions for "own" and complete the mpc pass using
        the same block of code as the single-agent version.

        Right now, this approach (with resizing) has losses of ~28 at first.
        Without resizing, the losses are ~6000!
        
        """
        x_predict = x_predict[:, : (self.nStep * (self.nHidden + 2)), :]
        print("x_predict size: ", x_predict.size())
        
        embb_output = Variable(torch.zeros(1,self.nHidden).cuda())
        state_output = Variable(torch.eye(1).cuda())
        output_single = torch.hstack((embb_output,state_output))
        output_single = torch.hstack((output_single,torch.zeros(1,1).cuda()))
        output_stack = output_single.unsqueeze(0).expand(self.nStep, 1, self.nHidden+2)
        output_dia =  torch.block_diag(*output_stack).cuda()
        output_batch = output_dia.unsqueeze(0).expand(nBatch, 1*self.nStep, self.nStep*(self.nHidden+2))
        posi_predict = torch.bmm(output_batch,x_predict).resize(nBatch,self.nStep)
        
        x = posi_predict    
        return x

