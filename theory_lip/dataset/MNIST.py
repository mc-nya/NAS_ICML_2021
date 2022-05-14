import torch
from torch.utils.data.dataset import TensorDataset
import torchvision
import numpy as np
import numpy.linalg as npl
def get_MNIST(batch_size_train,batch_size_test,classes=10):
    train_data=torchvision.datasets.MNIST('./dataset/download/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    test_data=torchvision.datasets.MNIST('./dataset/download/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    train_label=train_data.targets.numpy()
    test_label=test_data.targets.numpy()
    train_data=train_data.data.numpy()/255.
    test_data=test_data.data.numpy()/255.
    train_index=np.where(train_label<classes)[0]
    test_index=np.where(test_label<classes)[0]

    #train_norm=npl.norm(train_data,axis=(1,2))
    #train_data=np.array([train_data[i]/train_norm[i] for i in range(train_data.shape[0])])[train_index]
    train_data=np.array([(train_data[i]-0.1307)/0.3081 for i in range(train_data.shape[0])])[train_index]
    train_label=train_label[train_index]

    #test_norm=npl.norm(test_data,axis=(1,2))
    #test_data=np.array([test_data[i]/test_norm[i] for i in range(test_data.shape[0])])[test_index]
    test_data=np.array([(test_data[i]-0.1307)/0.3081 for i in range(test_data.shape[0])])[test_index]
    test_label=test_label[test_index]
    
    # if(classes==2):
    #     test_label=test_label*2-1
    #     train_label=train_label*2-1
    tensor_train=torch.Tensor(train_data)
    train_label=torch.Tensor(train_label)
    tensor_test=torch.Tensor(test_data)
    test_label=torch.Tensor(test_label)

    train_data=TensorDataset(tensor_train,train_label)
    test_data=TensorDataset(tensor_test,test_label)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size_test, shuffle=True)
    return train_loader,test_loader

#get_MNIST(128,128,2)