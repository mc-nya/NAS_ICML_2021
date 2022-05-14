import torch
import numpy as np
import torch.nn.functional as F
def test_epoch(model,epoch,test_loader,loss_func):
    test_losses=[]
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data,target=data.cuda(),target.cuda()
            digits = model(data)[:,0]
            #print(torch.mean(data),torch.std(data),torch.mean(digits))
            test_loss += loss_func(digits, target).item()
            # print(digits.shape,target.shape)
            # print(digits,target)
            correct += (((digits-0.5)*(target-0.5))>0).sum()
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    # test_loss, correct, len(test_loader.dataset),
    # 100. * correct / len(test_loader.dataset)))
    ret={}
    ret['loss']=test_loss
    ret['acc']=(100. * correct / len(test_loader.dataset)).item()
    return ret

def test_epoch_cpu(model,epoch,test_loader,loss_func):
    test_losses=[]
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            digits = model(data)[:,0]
            #print(torch.mean(data),torch.std(data),torch.mean(digits))
            test_loss += loss_func(digits, target).item()
            # print(digits.shape,target.shape)
            # print(digits,target)
            correct += (((digits-0.5)*(target-0.5))>0).sum()
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    # test_loss, correct, len(test_loader.dataset),
    # 100. * correct / len(test_loader.dataset)))
    ret={}
    ret['loss']=test_loss
    ret['acc']=(100. * correct / len(test_loader.dataset)).item()
    return ret
    
def train_epoch(model,epoch,train_loader,loss_func,optimizer,log_interval=50):
    model.train()
    train_losses=[]
    train_counter=[]
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target=data.cuda(),target.cuda()
        optimizer.zero_grad()
        digits=model(data)[:,0]
        loss=loss_func(digits,target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct += (((digits-0.5)*(target-0.5))>0).sum()
            train_losses.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    ret={}
    ret['loss']=np.mean(train_losses)
    ret['acc']=(100. * correct / len(train_loader.dataset)).item()
    return ret

def get_weights(model):
    weight=[]
    for param in model:
        if 'weight' in param:
            weight.extend(model[param].data.view(-1).numpy())
    weight=np.array(weight)
    return weight