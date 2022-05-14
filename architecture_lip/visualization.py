import torch
import numpy as np
import numpy.linalg as npl
from TrainFunction import get_weights
import seaborn as sns
import matplotlib.pyplot as plt

epsilons=np.arange(0,1.01,0.05)
net_type='normalized_1024'
init_model_path=f'result/{net_type}/model_0.pth'

init_model=torch.load(init_model_path,map_location='cpu')

weights=[]


for epsilon in epsilons:
    output_path=f'result/{net_type}/{epsilon}/'
    model_path=f'{output_path}/model_final.pth'
    model=torch.load(model_path,map_location='cpu')
    weights.append(get_weights(model))

#weights.append(get_weights(init_model))
init_weight=get_weights(init_model)
l=len(weights)
distance=np.zeros((l,l))
for i in range(l):
    for j in range(l):
        distance[i,j]=npl.norm(weights[i]-weights[j])/np.sqrt((weights[i]-init_weight).dot(weights[j]-init_weight))
        #print(distance[i,j])
        
epsilons=['%.2f' %e for e in epsilons]


#epsilons.append('init')
print(epsilons)
sns.heatmap(distance,annot=True, fmt='.2f',annot_kws=dict(fontsize=5),vmin=0,vmax=2)
plt.xticks(range(21),epsilons,rotation=90)
plt.yticks(range(21),epsilons)
plt.savefig(f'result/{net_type}/heatmap.pdf')