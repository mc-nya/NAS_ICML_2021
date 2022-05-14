import os,sys
sys.path.append('.')
import torch
import torchvision
from dataset.MNIST import get_MNIST
from model.MLP import OneHidden
from TrainFunction import test_epoch,train_epoch
import torch.optim as optim
import numpy as np
from torch.nn import CrossEntropyLoss,MSELoss,BCELoss,BCEWithLogitsLoss
torch.backends.cudnn.enabled = True

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--epsilon', dest='epsilon', default=0.0, type=float)
parser.add_argument('--epoch', dest='epoch', default=100, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=128, type=int)
parser.add_argument('--lr', dest='lr', default=0.01, type=float)
parser.add_argument('--log_interval', dest='log_interval', default=20, type=int)
parser.add_argument('--checkpoint_interval', dest='checkpoint_interval', default=50, type=int)
parser.add_argument('--model_file', dest='model_file', default=None, type=str)
parser.add_argument('--save_path', dest='save_path', default=None, type=str)
parser.add_argument('--width', dest='width', default=256, type=int)
parser.add_argument('--normalize', dest='normalize', default=True, type=bool)
args=parser.parse_args()

n_epochs = args.epoch
batch_size_train = args.batch_size
batch_size_test = 1000
learning_rate = args.lr
log_interval = args.log_interval
epsilon=args.epsilon
checkpoint_interval=args.checkpoint_interval
model_file=args.model_file
save_path=args.save_path
net_width=args.width
normalize=args.normalize
num_classes=2

if not save_path:
    save_path=f'result/{net_width}/{epsilon}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path,exist_ok=True)

train_loader,test_loader=get_MNIST(batch_size_train,batch_size_test,num_classes)
model=OneHidden(epsilon,normalize=normalize,width=net_width,num_classes=num_classes)
# for name,param in model.named_parameters():
#     if param.requires_grad:
#         print (name, param.data)

if model_file:
    model.load_state_dict(torch.load(model_file,map_location='cpu'))
model=model.cuda()
optimizer=optim.SGD(model.parameters(),lr=learning_rate)
#loss_function=CrossEntropyLoss().cuda()
loss_function=MSELoss().cuda()
train_result=[]
test_result=[]
for epoch in range(n_epochs+1):
    train_result.append(test_epoch(model,epoch,train_loader,loss_function))
    test_result.append(test_epoch(model,epoch,test_loader,loss_function))
    if epoch % checkpoint_interval == 0:
        torch.save(model.state_dict(), f'{save_path}/checkpoint_model_{epoch}.pth')
        torch.save(optimizer.state_dict(), f'{save_path}/checkpoint_optimizer_{epoch}.pth')
    print(f'Epoch : {epoch}.    Train result:  {train_result[-1]}')
    print(f'Epoch : {epoch}.    Test result:  {test_result[-1]}')
    if train_result[-1]['acc']>99.9:
        break
    train_epoch(model,epoch,train_loader,loss_function,optimizer,log_interval=log_interval)
    
torch.save(model.state_dict(), f'{save_path}/model_final.pth')
torch.save(optimizer.state_dict(), f'{save_path}/optimizer_final.pth')
np.save(f'{save_path}/train_result.npy',train_result)
np.save(f'{save_path}/test_result.npy',test_result)

