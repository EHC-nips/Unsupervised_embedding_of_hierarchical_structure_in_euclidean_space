import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np
import os

class VAE(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.fc1 = nn.Linear(self.input_dim, 500)
        self.fc2 = nn.Linear(500,200)
        self.fc21 = nn.Linear(200, self.hid_dim)
        self.fc22 = nn.Linear(200, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, 200)
        self.fc4 = nn.Linear(200,500)
        self.fc5 = nn.Linear(500, self.input_dim)
    def encode(self, x):
        h1 = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc4(F.relu(self.fc3(z))))
        return self.fc5(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VaDE(nn.Module):
    def __init__(self,nClusters, hid_dim, input_dim = 100, inter_dims = [50,50,100]):
        super(VaDE,self).__init__()
        self.encoder=Encoder(input_dim, hid_dim=hid_dim, inter_dims = [50,50,100])
        self.decoder=Decoder(input_dim, hid_dim=hid_dim, inter_dims = [50,50,100])

        self.pi_=nn.Parameter(torch.FloatTensor(nClusters,).fill_(1)/nClusters,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(nClusters,hid_dim).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(nClusters,hid_dim).fill_(0),requires_grad=True)
        self.nClusters = nClusters


    def pre_train(self,dataloader,pre_epoch=10):
        if  not os.path.exists('./pretrain_model.pk'):
            Loss=nn.MSELoss()
            opti=Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))

            print('Pretraining......')
            epoch_bar=tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L=0
                for x in dataloader:
                    z,_=self.encoder(x)
                    x_=self.decoder(z)
                    loss=Loss(x,x_)

                    L+=loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                epoch_bar.write('L2={:.4f}'.format(L/len(dataloader)))

            self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

            Z = []
            with torch.no_grad():
                for x in dataloader:
                    z1, z2 = self.encoder(x)
                    assert F.mse_loss(z1, z2) == 0
                    Z.append(z1)


            Z = torch.cat(Z, 0).detach().cpu().numpy()

            gmm = GaussianMixture(n_components=self.nClusters, covariance_type='diag')

            pre = gmm.fit_predict(Z)


            self.pi_.data = torch.from_numpy(gmm.weights_).float()
            self.mu_c.data = torch.from_numpy(gmm.means_).float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).float())
        '''
            torch.save(self.state_dict(), './pretrain_model.pk')
        else:
            self.load_state_dict(torch.load('./pretrain_model.pk'))
        '''



    def predict(self,x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)


    def ELBO_Loss(self,x,L=1):
        det=1e-10

        L_rec=0

        z_mu, z_sigma2_log = self.encoder(x)
        #print("z_mu:", torch.sum(torch.isnan(z_mu)))
        #print("z_sigma2_log:", torch.sum(torch.isnan(z_sigma2_log)))
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_pro=self.decoder(z)
            #L_rec+=F.binary_cross_entropy(x_pro,x)
            L_rec+=F.mse_loss(x_pro, x)
        #print("L_rec:", torch.isnan(L_rec))
        L_rec/=L
        Loss=L_rec*x.size(1)
        pi=self.pi_ * (self.pi_ > 0) + 1e-5
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c
    
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        #print("pi", pi.unsqueeze(0))
        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters
        
        Loss+=2 * 0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))
        #print("loss1:", torch.isnan(Loss))
        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))
        return Loss






    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)




    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
        nn.ReLU(True)
    ]
    return layers

class Encoder(nn.Module):
    def __init__(self,input_dim=100,inter_dims=[50,50,200],hid_dim=10):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            *block(input_dim,inter_dims[0]),
            #*block(inter_dims[0],inter_dims[1]),
            *block(inter_dims[1],inter_dims[2]),
        )

        self.mu_l=nn.Linear(inter_dims[-1],hid_dim)
        self.log_sigma2_l=nn.Linear(inter_dims[-1],hid_dim)

    def forward(self, x):
        e=self.encoder(x)
        mu=self.mu_l(e)
        log_sigma2=self.log_sigma2_l(e)

        return mu,log_sigma2


class Decoder(nn.Module):
    def __init__(self,input_dim=100,inter_dims=[50,50,200],hid_dim=10):
        super(Decoder,self).__init__()

        self.decoder=nn.Sequential(
            *block(hid_dim,inter_dims[-1]),
            *block(inter_dims[-1],inter_dims[-2]),
            #*block(inter_dims[-2],inter_dims[-3]),
            nn.Linear(inter_dims[-3],input_dim),
            #nn.Sigmoid()
        )

    def forward(self, z):
        x_pro=self.decoder(z)

        return x_pro
