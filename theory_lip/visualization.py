import torch
import numpy as np
import numpy.linalg as npl
from TrainFunction import get_weights
import seaborn as sns
import matplotlib.pyplot as plt
from model.MLP import Deeper as Model
from TrainFunction import test_epoch_cpu as test_epoch
from dataset.MNIST import get_MNIST
from torch.nn import CrossEntropyLoss,MSELoss,BCELoss,BCEWithLogitsLoss
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#width_list=np.arange(64,2048,64)
width_list=np.arange(64,1000,64)
width_list=[64,128,256,512,1024,2048,4096]
delta_eps_list=[0.005,0.01,0.03,0.05,0.07]
eps_list=np.arange(0.1,0.8,0.1)
result_path='result_deep'
train_loader,test_loader=get_MNIST(128,1000,2)
loss_function=MSELoss()#.cuda()
legend=[]
for delta_eps in delta_eps_list:
    means=[]
    stds=[]
    for width in width_list:
        init_model_path=f'{result_path}/{width}/{eps_list[0]}/checkpoint_model_0.pth'
        init_model=torch.load(init_model_path,map_location='cpu')
        init_weight=get_weights(init_model)

        datapoints=[]
        #model=Model(0,width=width,num_classes=2)
        for eps in eps_list:
            #model.epsilon=eps

            final_model_path=f'{result_path}/{width}/{eps}/model_final.pth'
            final_model=torch.load(final_model_path,map_location='cpu')
            final_weight=get_weights(final_model)

            #model.load_state_dict(final_model)
            #model=model
            #result=test_epoch(model,0,test_loader,loss_function)
            #if result['acc']<90:
            #    continue

            delta_model_path=f'{result_path}/{width}/{eps+delta_eps}/model_final.pth'
            delta_model=torch.load(delta_model_path,map_location='cpu')
            #model.load_state_dict(delta_model)
            #model=model#.cuda()
            #result=test_epoch(model,0,test_loader,loss_function)
            delta_weight=get_weights(delta_model)
            #print(eps,eps+delta_eps,result,npl.norm(final_weight-delta_weight)/np.sqrt((final_weight-init_weight).dot(delta_weight-init_weight)))
            #if result['acc']>90:
                #distance=npl.norm(final_weight-delta_weight)/np.sqrt((final_weight-init_weight).dot(delta_weight-init_weight))/delta_eps
            distance=npl.norm(final_weight-delta_weight)/delta_eps
            if not np.isnan(distance):
                #print(npl.norm(final_weight-delta_weight),np.sqrt((final_weight-init_weight).dot(delta_weight-init_weight)), npl.norm(final_weight-init_weight),npl.norm(delta_weight-init_weight))
                    datapoints.append(distance)

            delta_model_path=f'{result_path}/{width}/{eps-delta_eps}/model_final.pth'
            delta_model=torch.load(delta_model_path,map_location='cpu')
            #model.load_state_dict(delta_model)
            #model=model#.cuda()
            #result=test_epoch(model,0,test_loader,loss_function)
            delta_weight=get_weights(delta_model)
            #print(eps,eps+delta_eps,result,npl.norm(final_weight-delta_weight)/np.sqrt((final_weight-init_weight).dot(delta_weight-init_weight)))
            #if result['acc']>90:
                #distance=npl.norm(final_weight-delta_weight)/np.sqrt((final_weight-init_weight).dot(delta_weight-init_weight))/delta_eps
            distance=npl.norm(final_weight-delta_weight)/delta_eps
            if not np.isnan(distance):
                datapoints.append(distance)

        means.append(np.mean(datapoints))
        stds.append(np.std(datapoints)/np.sqrt(len(datapoints)))
        print(delta_eps,width,datapoints)
    print(means,stds)
    plt.errorbar(width_list,means,yerr=stds,linewidth=3, elinewidth=2,capsize=4)#, capsize=1, capthick=1)
    legend.append('$\Delta\\alpha$='+str(delta_eps))
plt.xscale('log')
#plt.legend(legend,fontsize=18,loc=1)
plt.legend(legend,fontsize=18,loc=3)
plt.xticks(width_list,fontsize=19)
plt.yticks(fontsize=19)
plt.grid('--')
plt.ylim(bottom=0,top=3)

ax1=plt.axes()
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig(f'{result_path}/figure.pdf')