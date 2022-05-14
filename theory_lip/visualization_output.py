import torch
import numpy as np
import numpy.linalg as npl
from TrainFunction import get_weights
import seaborn as sns
import matplotlib.pyplot as plt
#from model.MLP import OneHidden as Model
from model.MLP import Deeper as Model
from TrainFunction import test_epoch_cpu as test_epoch
from dataset.MNIST import get_MNIST
from torch.nn import CrossEntropyLoss,MSELoss,BCELoss,BCEWithLogitsLoss
import matplotlib
#width_list=np.arange(64,2048,64)
width_list=np.arange(64,1000,64)
width_list=[64,128,256,512,1024,2048,4096]
delta_eps_list=[0.005,0.01,0.03,0.05,0.07]
eps_list=np.arange(0.1,0.8,0.1)
result_path='result_deep'
train_loader,test_loader=get_MNIST(512,1000,2)
loss_function=MSELoss()#.cuda()
legend=[]

def test_epoch_distance(model_1,model_2,test_loader):
    model_1.eval()
    model_2.eval()
    accumulate_difference=0.
    max_difference=0.
    total_num=0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.cuda()
            total_num+=data.shape[0]
            digits_1 = model_1(data).cpu().numpy()[:,0]
            digits_2 = model_2(data).cpu().numpy()[:,0]
            #print(digits_1.shape,digits_2.shape)
            difference=(digits_1-digits_2)**2
            accumulate_difference+=np.sum(difference)
            max_difference=max(max_difference,max(difference))
    return np.sqrt(accumulate_difference/total_num), np.sqrt(max_difference)

avg_all={}
max_all={}
for delta_eps in delta_eps_list:
    means_avg_output=[]
    stds_avg_output=[]
    means_max_output=[]
    stds_max_output=[]
    for width in width_list:
        init_model_path=f'{result_path}/{width}/{eps_list[0]}/checkpoint_model_0.pth'
        init_model=torch.load(init_model_path,map_location='cpu')
        init_weight=get_weights(init_model)

        datapoints_avg=[]
        datapoints_max=[]
        model_final=Model(0,width=width,num_classes=2)
        model_delta=Model(0,width=width,num_classes=2)
        for eps in eps_list:
            model_final.epsilon=eps
            model_delta.epsilon=eps

            final_model_path=f'{result_path}/{width}/{eps}/model_final.pth'
            final_model=torch.load(final_model_path,map_location='cpu')
            model_final.load_state_dict(final_model)
            model_final=model_final.cuda()

            delta_model_path=f'{result_path}/{width}/{eps+delta_eps}/model_final.pth'
            delta_model=torch.load(delta_model_path,map_location='cpu')
            model_delta.load_state_dict(delta_model)
            model_delta=model_delta.cuda()
            avg_distance,max_distance=test_epoch_distance(model_final,model_delta,test_loader)
            avg_distance/=delta_eps
            max_distance/=delta_eps
            if not np.isnan(avg_distance):
                datapoints_avg.append(avg_distance)
            if not np.isnan(max_distance):
                datapoints_max.append(max_distance)

            delta_model_path=f'{result_path}/{width}/{eps-delta_eps}/model_final.pth'
            delta_model=torch.load(delta_model_path,map_location='cpu')
            model_delta.load_state_dict(delta_model)
            model_delta=model_delta.cuda()
            avg_distance,max_distance=test_epoch_distance(model_final,model_delta,test_loader)
            avg_distance/=delta_eps
            max_distance/=delta_eps
            if not np.isnan(avg_distance):
                datapoints_avg.append(avg_distance)
            if not np.isnan(max_distance):
                datapoints_max.append(max_distance)

        means_avg_output.append(np.mean(datapoints_avg))
        stds_avg_output.append(np.std(datapoints_avg)/np.sqrt(len(datapoints_avg)))
        means_max_output.append(np.mean(datapoints_max))
        stds_max_output.append(np.std(datapoints_max)/np.sqrt(len(datapoints_max)))
        print(delta_eps,width)
    #print(means,stds)
    avg_all[delta_eps]=[means_avg_output,stds_avg_output]
    max_all[delta_eps]=[means_max_output,stds_max_output]
    #plt.errorbar(width_list,means_avg_output,yerr=stds_avg_output,linewidth=3, elinewidth=2,capsize=4)#, capsize=1, capthick=1)
    #legend.append('$\Delta\\alpha$='+str(delta_eps))

plt.cla()
for delta_eps in delta_eps_list:
    #print(means,stds)
    plt.errorbar(width_list,avg_all[delta_eps][0],yerr=avg_all[delta_eps][1],linewidth=3, elinewidth=2,capsize=4)#, capsize=1, capthick=1)
    legend.append('$\Delta\\alpha$='+str(delta_eps))

plt.xscale('log')
plt.legend(legend,fontsize=18)
plt.xticks(width_list,fontsize=19)
plt.yticks(fontsize=19)
plt.grid('--')
plt.ylim(bottom=0)#,top=2)
#plt.ylim(bottom=0,top=1.4)

ax1=plt.axes()
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig(f'{result_path}/figure_avg_output_test.pdf')

plt.cla()
for delta_eps in delta_eps_list:
    #print(means,stds)
    plt.errorbar(width_list,max_all[delta_eps][0],yerr=max_all[delta_eps][1],linewidth=3, elinewidth=2,capsize=4)#, capsize=1, capthick=1)
    legend.append('$\Delta\\alpha$='+str(delta_eps))
    
plt.xscale('log')
plt.legend(legend,fontsize=18)
plt.xticks(width_list,fontsize=19)
plt.yticks(fontsize=19)
plt.grid('--')
plt.ylim(bottom=0)#,top=3)

ax1=plt.axes()
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig(f'{result_path}/figure_max_output_test.pdf')