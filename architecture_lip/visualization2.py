import torch
import numpy as np
import numpy.linalg as npl
from TrainFunction import get_weights
import seaborn as sns
import matplotlib.pyplot as plt
from model.MLP import OneHidden
from TrainFunction import test_epoch_cpu as test_epoch
from dataset.MNIST import get_MNIST
from torch.nn import CrossEntropyLoss,MSELoss,BCELoss,BCEWithLogitsLoss

#width_list=np.arange(64,2048,64)
width_list=[64,128,256,512,1024]#,2048,4096]
delta_eps_list=[0.01,0.03,0.05]#,0.07]
eps_list=np.arange(0.1,1.0,0.1)
#eps_list=np.arange(0.1,1.0,0.1)
legend=[]
for delta_eps in delta_eps_list:
    means=[]
    stds=[]
    for width in width_list:
        init_model_path=f'result/{width}/{eps_list[0]}/checkpoint_model_0.pth'
        init_model=torch.load(init_model_path,map_location='cpu')
        init_weight=get_weights(init_model)

        datapoints=[]
        for eps in eps_list:


            final_model_path=f'result/{width}/{eps}/model_final.pth'
            final_model=torch.load(final_model_path,map_location='cpu')
            final_weight=get_weights(final_model)

            delta_model_path=f'result/{width}/{eps+delta_eps}/model_final.pth'
            delta_model=torch.load(delta_model_path,map_location='cpu')
            delta_weight=get_weights(delta_model)
            distance=npl.norm(final_weight-delta_weight)/np.sqrt((final_weight-init_weight).dot(delta_weight-init_weight))
            datapoints.append(distance)

            delta_model_path=f'result/{width}/{eps-delta_eps}/model_final.pth'
            delta_model=torch.load(delta_model_path,map_location='cpu')
            delta_weight=get_weights(delta_model)
            distance=npl.norm(final_weight-delta_weight)/np.sqrt((final_weight-init_weight).dot(delta_weight-init_weight))
            datapoints.append(distance)

        means.append(np.mean(datapoints))
        stds.append(np.std(datapoints)/np.sqrt(len(datapoints)))
    print(means,stds)
    plt.errorbar(width_list,means,yerr=stds, uplims=True, lolims=True,linewidth=3)#, capsize=1, capthick=1)
    legend.append('$\Delta\\alpha$='+str(delta_eps))
plt.legend(legend,fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('result/figure.pdf')