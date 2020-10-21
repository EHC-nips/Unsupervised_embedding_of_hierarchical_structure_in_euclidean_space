import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset
import pickle


def get_mnist(data_dir='./data/mnist/',batch_size=128):
    train=MNIST(root=data_dir,train=True,download=True)
    test=MNIST(root=data_dir,train=False,download=True)

    X=torch.cat([train.data.float().view(-1,784)/255.,test.data.float().view(-1,784)/255.],0)
    Y=torch.cat([train.targets,test.targets],0)

    dataset=dict()
    dataset['X']=X
    dataset['Y']=Y

    dataloader=DataLoader(TensorDataset(X,Y),batch_size=batch_size,shuffle=True,num_workers=4)

    return dataloader,dataset



def get_20newsgroup(data_dir, batch_size=128, device = "cuda"):
    with open(data_dir, "rb") as f:
        dic = pickle.load(f)
        X = dic["X"]
        y = dic["y"]
    train_loader = []
    #X -= X.min(1, keepdim = True)[0]
    #X /= X.max(1, keepdim = True)[0]
    for i in range(len(X) // batch_size):
        train_loader.append([X[i*batch_size: (i+1) * batch_size].float(), y[i*batch_size:(i+1)*batch_size]])
    return train_loader, batch_size





