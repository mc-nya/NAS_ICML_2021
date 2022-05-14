import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn.init as init
import numpy as np
class OneHidden(nn.Module):
    # def __init__(self,epsilon,normalize=False,width=256,num_classes=10):
    #     super(OneHidden, self).__init__()
    #     self.normalize=normalize
    #     self.fc1=nn.Linear(784,width)
    #     self.fc3=nn.Linear(width,width//2)
    #     self.fc4=nn.Linear(width//2,width//4)
    #     if num_classes==2:
    #         self.fc2=nn.Linear(width//4,1)
    #     else:
    #         self.fc2=nn.Linear(width//4,num_classes)
    #     # if normalize:
    #     #     init.normal_(self.fc1.weight,0,1)
    #     #     init.normal_(self.fc2.weight,0,1)
    #     #     init.zeros_(self.fc1.bias)
    #     #     init.zeros_(self.fc2.bias)
    #     #     with torch.no_grad():
    #     #         self.fc2.weight-=torch.median(self.fc2.weight)+0.0000001
    #     #         self.fc2.weight*=torch.abs(1/self.fc2.weight)
    #     #         self.fc2.weight*=(0.5)/np.sqrt(width)
    #     # self.fc2.weight.requires_grad=False
    #     # self.fc2.bias.requires_grad=False
    #     # self.fc2.weight.requires_grad=False
    #     # self.fc2.bias.requires_grad=False
    #     self.epsilon=epsilon
        

    # def forward(self, x):
    #     x = torch.flatten(x,start_dim=1)
    #     x = self.fc1(x)
    #     #x = self.fc1(x.view(-1,x.shape[0]))
    #     x = (1-self.epsilon)* F.relu(x) + self.epsilon * torch.sigmoid(x)
    #     x = self.fc3(x)
    #     x = (1-self.epsilon)* F.relu(x) + self.epsilon * torch.sigmoid(x)
    #     x = self.fc4(x)
    #     x = (1-self.epsilon)* F.relu(x) + self.epsilon * torch.sigmoid(x)
    #     x = self.fc2(x)
    #     return x
    def __init__(self,epsilon,normalize=False,width=256,num_classes=10):
        super(OneHidden, self).__init__()
        self.normalize=normalize
        self.fc1=nn.Linear(784,width)

        if num_classes==2:
            self.fc2=nn.Linear(width,1)
        else:
            self.fc2=nn.Linear(width,num_classes)
        # if normalize:
        #     init.normal_(self.fc1.weight,0,1)
        #     init.normal_(self.fc2.weight,0,1)
        #     init.zeros_(self.fc1.bias)
        #     init.zeros_(self.fc2.bias)
        #     with torch.no_grad():
        #         self.fc2.weight-=torch.median(self.fc2.weight)+0.0000001
        #         self.fc2.weight*=torch.abs(1/self.fc2.weight)
        #         self.fc2.weight*=(0.5)/np.sqrt(width)
        self.fc2.weight.requires_grad=False
        self.fc2.bias.requires_grad=False
        self.epsilon=epsilon
        

    def forward(self, x):
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        x = (1-self.epsilon)* F.relu(x) + self.epsilon * torch.sigmoid(x)
        x = self.fc2(x)
        return x
#OneHidden(0.0,True,256)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn.init as init
import numpy as np
class Deeper(nn.Module):
    def __init__(self,epsilon,normalize=False,width=256,num_classes=10):
        super(Deeper, self).__init__()
        self.normalize=normalize
        self.fc1=nn.Linear(784,width)
        self.fc3=nn.Linear(width,width//2)
        self.fc4=nn.Linear(width//2,width//4)
        if num_classes==2:
            self.fc2=nn.Linear(width//4,1)
        else:
            self.fc2=nn.Linear(width//4,num_classes)
        # if normalize:
        #     init.normal_(self.fc1.weight,0,1)
        #     init.normal_(self.fc2.weight,0,1)
        #     init.zeros_(self.fc1.bias)
        #     init.zeros_(self.fc2.bias)
        #     with torch.no_grad():
        #         self.fc2.weight-=torch.median(self.fc2.weight)+0.0000001
        #         self.fc2.weight*=torch.abs(1/self.fc2.weight)
        #         self.fc2.weight*=(0.5)/np.sqrt(width)
        # self.fc2.weight.requires_grad=False
        # self.fc2.bias.requires_grad=False
        # self.fc2.weight.requires_grad=False
        # self.fc2.bias.requires_grad=False
        self.epsilon=epsilon
        

    def forward(self, x):
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        #x = self.fc1(x.view(-1,x.shape[0]))
        x = (1-self.epsilon)* F.relu(x) + self.epsilon * torch.sigmoid(x)
        x = self.fc3(x)
        x = (1-self.epsilon)* F.relu(x) + self.epsilon * torch.sigmoid(x)
        x = self.fc4(x)
        x = (1-self.epsilon)* F.relu(x) + self.epsilon * torch.sigmoid(x)
        x = self.fc2(x)
        return x
    # def __init__(self,epsilon,normalize=False,width=256,num_classes=10):
    #     super(OneHidden, self).__init__()
    #     self.normalize=normalize
    #     self.fc1=nn.Linear(784,width)

    #     if num_classes==2:
    #         self.fc2=nn.Linear(width,1)
    #     else:
    #         self.fc2=nn.Linear(width,num_classes)
    #     # if normalize:
    #     #     init.normal_(self.fc1.weight,0,1)
    #     #     init.normal_(self.fc2.weight,0,1)
    #     #     init.zeros_(self.fc1.bias)
    #     #     init.zeros_(self.fc2.bias)
    #     #     with torch.no_grad():
    #     #         self.fc2.weight-=torch.median(self.fc2.weight)+0.0000001
    #     #         self.fc2.weight*=torch.abs(1/self.fc2.weight)
    #     #         self.fc2.weight*=(0.5)/np.sqrt(width)
    #     # self.fc2.weight.requires_grad=False
    #     # self.fc2.bias.requires_grad=False
    #     # self.fc2.weight.requires_grad=False
    #     # self.fc2.bias.requires_grad=False
    #     self.epsilon=epsilon
        

    # def forward(self, x):
    #     x = torch.flatten(x,start_dim=1)
    #     x = self.fc1(x)
    #     x = (1-self.epsilon)* F.relu(x) + self.epsilon * torch.sigmoid(x)
    #     x = self.fc2(x)
    #     return x
#OneHidden(0.0,True,256)