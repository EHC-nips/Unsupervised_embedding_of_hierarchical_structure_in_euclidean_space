from util import *
import argparse
import numpy as np
import scipy
from scipy import cluster
import numpy as np
import torch
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm
from sklearn.decomposition import PCA
import sklearn
from dataloader import *
from model import *

parse=argparse.ArgumentParser(description='VaDE')
parse.add_argument('--batch_size',type=int,default=800)
parse.add_argument('--datadir',type=str,default='./data/mnist')
parse.add_argument('--nClusters',type=int,default=10)
parse.add_argument('--num', type=int, default = 2000)
parse.add_argument('--hid_dim',type=int,default=10)
parse.add_argument('--cuda',type=bool,default=False)


args=parse.parse_args()

labels = ['rec.motorcycles', 'misc.forsale', 'talk.politics.guns',
       'comp.sys.mac.hardware', 'soc.religion.christian',
       'talk.politics.misc', 'comp.graphics', 'sci.electronics',
       'talk.religion.misc', 'comp.os.ms-windows.misc', 'alt.atheism',
       'comp.sys.ibm.pc.hardware', 'rec.sport.baseball', 'sci.med',
       'sci.space', 'rec.autos', 'rec.sport.hockey', 'comp.windows.x',
       'talk.politics.mideast', 'sci.crypt']

similarity_matrix = np.zeros((20,20))
for i in range(20):
    for j in range(20):
        label_i = labels[i].split(".")
        label_j = labels[j].split(".")
        #print(label_i,label_j)
        if labels[i] == labels[j]:
            similarity_matrix[i,j] = len(label_i)
        for k in range(min(len(label_i), len(label_j))):
            if label_i[k] != label_j[k]:
                similarity_matrix[i,j] = k
                break

def compute_objective_gt(n, root, cla):
    obj = 0
    if root.is_leaf():
        return 0
    else:
        right_leaves  = list_leaves(root.right)
        left_leaves = list_leaves(root.left)
        for i, index1 in enumerate(right_leaves):
            for j, index2 in enumerate(left_leaves):
                xi = cla[index1]
                xj = cla[index2]
                #if xi == xj:
                if similarity_matrix[xi,xj] != 0:
                    obj += (n - len(left_leaves) - len(right_leaves)) * similarity_matrix[xi,xj]
                    
        obj_right = compute_objective_gt(n, root.right, cla)
        obj_left = compute_objective_gt(n, root.left, cla)
        #print(obj, obj_right, obj_left)
        return obj + obj_right + obj_left     


def predict(model,x):
    z_mu, z_sigma2_log = model.encoder(x)
    z = z_mu
    pi = model.pi_
    log_sigma2_c = model.log_sigma2_c
    mu_c = model.mu_c
    yita_c = torch.exp(torch.log(pi.unsqueeze(0))+model.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

    yita=yita_c.detach().cpu().numpy()
    return np.argmax(yita,axis=1)


def transformation(model, data, rate = 2):
    mean, _ = model.encoder(torch.from_numpy(data).float())
    pred = predict(model, torch.from_numpy(data).float())
    cluster_means = model.mu_c[pred]
    scaled_cluster_means = cluster_means * rate
    scaled_mean = (mean - cluster_means) + scaled_cluster_means
    return scaled_mean.detach()

#DL,_=get_mnist(args.datadir,args.batch_size)
DL,_=get_20newsgroup("tfidf_embedding.pk",batch_size = 128)

with open("tfidf_embedding.pk", "rb") as f:
    dic = pickle.load(f)
    X = dic["X"].float()
    y = dic["y"]

vade=VaDE(args)
#vade=nn.DataParallel(vade,device_ids=range(1))

vade.pre_train(DL,pre_epoch=50)

mean, _ = vade.encoder(X)
mean = mean.detach().numpy()
transformed_mean = transformation(vade, X.numpy())

pca = PCA(n_components = 10)
pca_data = pca.fit_transform(X[:args.num])
origin_data = X[:args.num]

cluster_target = []
for i in range(args.num):
    cluster_target.append(similarity_matrix[target[i].numpy()])
cluster_target = np.array(cluster_target)
Z = linkage(cluster_target, "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
print("max MW objective:", compute_objective_gt(args.num, rootnode, y[:args.num]))


print("-" * 25 + "Computing Dendrogram Purity" + "-" * 25)

from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=20, random_state=0).fit(origin_data)
print("Origin KMeans NMI score:", normalized_mutual_info_score(kmeans.labels_, y[:args.num]))

kmeans = KMeans(n_clusters=20, random_state=0).fit(pca_data)
print("PCA KMeans NMI score:", normalized_mutual_info_score(kmeans.labels_, y[:args.num]))

kmeans = KMeans(n_clusters=20, random_state=0).fit(transformed_mean[:args.num])
print("Trans KMeans NMI score:", normalized_mutual_info_score(kmeans.labels_, y[:args.num]))

kmeans = KMeans(n_clusters=20, random_state=0).fit(mean[:args.num])
print("VaDE KMeans NMI score:", normalized_mutual_info_score(kmeans.labels_, y[:args.num]))
"""
Z = linkage(transformed_mean[:args.num], "ward")
print("Trans VaDE:", compute_purity(Z, y[:args.num]), args.nClusters)

Z = linkage(mean[:args.num], "ward")
print("VaDE:", compute_purity(Z, y[:args.num]), args.nClusters)

Z = linkage(pca_data, "ward")
print("PCA:", compute_purity(Z, y[:args.num]), args.nClusters)

Z = linkage(origin_data, "ward")
print("Origin:", compute_purity(Z, y[:args.num]), args.nClusters)
"""


print("-" * 25 + "Computing MW objective" + "-" * 25)
max = 1


"""
Z = linkage(y[:2000].reshape(-1,1), "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
max = compute_objective_gt(2000, rootnode, y[:2000]).numpy()
"""
Z = linkage(transformed_mean[:args.num], "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
print("Trans VaDE:", compute_objective_gt(args.num, rootnode, y[:args.num]) / max)

"""
Z = linkage(y[:2000].reshape(-1,1), "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
max = compute_objective_gt(2000, rootnode, y[:2000]).numpy()
"""
Z = linkage(mean[:args.num], "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
print("VaDE:", compute_objective_gt(args.num, rootnode, y[:args.num]) / max)


"""
Z = linkage(y[:2000].reshape(-1,1), "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
max = compute_objective_gt(2000, rootnode, y[:2000]).numpy()
"""
Z = linkage(pca_data, "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
print("PCA:", compute_objective_gt(args.num, rootnode, y[:args.num]) / max)

"""
Z = linkage(y[:2000].reshape(-1,1), "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
max = compute_objective_gt(2000, rootnode, y[:2000]).numpy()
"""
Z = linkage(origin_data, "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
print("origin:", compute_objective_gt(args.num, rootnode, y[:args.num]) / max)


