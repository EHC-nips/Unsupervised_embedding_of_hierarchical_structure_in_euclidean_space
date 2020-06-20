import numpy as np
import scipy
import numpy as np
import torch

from model import *
from util import *
from sklearn import datasets
import os
from sklearn.decomposition import PCA


device = "cuda"

def transformation(model, data, rate = 2, cla = None):
    mean, _ = model.encoder(torch.from_numpy(data).float())
    pred = model.predict(torch.from_numpy(data).float())
    cluster_means = model.mu_c[pred]
    scaled_cluster_means = cluster_means * rate
    scaled_mean = (mean - cluster_means) + scaled_cluster_means
    return scaled_mean.detach()


print("###################EXPERIMENTS ON DIGIT###########################")

with open("data/optdigits.tra") as f:
    raw_data = f.readlines()
data = []
cla = []
for i in range(len(raw_data)):
    line = raw_data[i].split(',')
    data.append(line[:-1])
    cla.append(int(line[-1]))
with open("data/optdigits.tes") as f:
    raw_data = f.readlines()
for i in range(len(raw_data)):
    line = raw_data[i].split(',')
    data.append(line[:-1])
    cla.append(int(line[-1]))    
data = np.array(data).astype(np.float)

train_loader = []
for i in range(data.shape[0]//100):  
    train_loader.append(torch.from_numpy(data[i*100:(i+1)*100]).float())


cla = np.array(cla)

model = VaDE(10, 10, 64, [50,50,100])
pca = PCA(n_components = 10)


if  os.path.exists('./pretrained_parameters/parameters_digits.pth'):
    model.load_state_dict(torch.load('./pretrained_parameters/parameters_digits.pth'))
else:
    
    model.pre_train(train_loader, 50)
    train(model, train_loader, 200)
    torch.save(model.state_dict(), "pretrained_parameters/parameters_digits.pth")

#train(model, train_loader, 100, lr = 5e-4)    

mean, _ = model.encoder(torch.from_numpy(data).float())
scaled_mean = transformation(model, data, 3)
projection = pca.fit_transform(data)
methods_list = ["average", "centroid", "complete", "single", "ward"]
NUM = 1000
for method in methods_list:
    print(method:)
    #print("VAE:", compute_purity_average(vae_mean.detach().numpy(), cla, 8, 200, 50, method = method))
    print("Transform DP:", compute_purity_average(scaled_mean.detach().numpy(), cla, 8, NUM, 50, method = method))
    print("VaDE DP:", compute_purity_average(mean.detach().numpy(), cla, 8, NUM, 50, method = method))
    print("PCA DP:", compute_purity_average(projection, cla, 8, NUM, 50, method = method))
    print("Origin DP:", compute_purity_average(data, cla, 8, NUM, 50, method = method))
    
    #print(compute_MW_objective_average(8, vae_mean.detach().numpy(), cla, 200, 50, method = method))
    print("Transform MW:", compute_MW_objective_average(8, scaled_mean.detach().numpy(), cla, NUM, 50, method = method))
    print("VaDE MW:", compute_MW_objective_average(8, mean.detach().numpy(), cla, NUM, 50, method = method))
    print("PCA MW:", compute_MW_objective_average(8, projection, cla, NUM, 50, method = method))
    print("Origin MW:",compute_MW_objective_average(8, data, cla, NUM, 50, method = method))

    '''
    Z_vade = linkage(mean.detach().numpy()[:200], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_vade, rd=True)
    print("VaDE Dendrogram Purity " + method + ":", compute_purity(Z_vade, cla[:200], 8))
    
    Z_pca = linkage(projection[:200], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_pca, rd=True)
    print("PCA Dendrogram Purity " + method + ":", compute_purity(Z_pca, cla[:200], 8))
    
    Z_pca = linkage(scaled_mean[:200], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z_pca, rd=True)
    print("Trans Dendrogram Purity " + method + ":", compute_purity(Z_pca, cla[:200], 8))

    Z = linkage(cla[:200].reshape(-1,1), method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    max = compute_objective_gt(200, rootnode, cla[:200])
    
    Z = linkage(mean.detach().numpy()[:200], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("VaDE MW:", compute_objective_gt(200, rootnode, cla[:200]) / max)
    
    Z = linkage(projection[:200], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("PCA MW:", compute_objective_gt(200, rootnode, cla[:200]) / max)
    
    Z = linkage(scaled_mean[:200], method)
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    print("Trans MW:", compute_objective_gt(200, rootnode, cla[:200]) / max)
'''