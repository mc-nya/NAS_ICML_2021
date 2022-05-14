import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn.init as init
import numpy as np
class OneHidden(nn.Module):
    def __init__(self,epsilon,normalize=False,width=256,num_classes=10):
        super(OneHidden, self).__init__()
        self.normalize=normalize
        self.fc1=nn.Linear(784,width)
        self.fc2=nn.Linear(width,num_classes)
        if normalize:
            init.normal_(self.fc1.weight,0,1/np.sqrt(784))
            init.normal_(self.fc2.weight,0,0.1/np.sqrt(width))
        self.epsilon=epsilon
        

    def forward(self, x):
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        #x = self.fc1(x.view(-1,x.shape[0]))
        x = (1-self.epsilon)* F.relu(x) + self.epsilon * torch.sigmoid(x)
        x = self.fc2(x)
        return x
    