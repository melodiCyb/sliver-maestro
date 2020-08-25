#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,transforms
import torch.utils
import torch.nn.utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from configparser import ConfigParser
from utils.model_utils import *
from utils import *
import argparse
import os 

config = ConfigParser()
config.read('config.cfg')



class DRAW(nn.Module):
    def __init__(self, T, A, B, batch_size, z_size, N, dec_size, enc_size, path):
        super(DRAW, self).__init__()
        self.T = T
        self.batch_size = batch_size
        self.A = A
        self.B = B
        self.z_size = z_size
        self.N = N
        self.dec_size = dec_size
        self.enc_size = enc_size
        self.cs = [0] * T
        self.logsigmas, self.sigmas, self.mus = [0] * T,[0] * T,[0] * T

        self.encoder = nn.LSTMCell(2 * N * N + dec_size, enc_size)
        self.mu_linear = nn.Linear(dec_size, z_size)
        self.sigma_linear = nn.Linear(dec_size, z_size)

        self.decoder = nn.LSTMCell(z_size, dec_size)
        self.dec_linear = nn.Linear(dec_size, 5)
        self.dec_w_linear = nn.Linear(dec_size, N * N)

        self.sigmoid = nn.Sigmoid()
        
        self.path = path
        self.phases = ["train", "test"]
        self.dataloaders = {
            phase: provider(
                path=self.path,
                phase=phase
            )
            for phase in self.phases
        }
        self.final_output_path = 'save/weights_final.tar'

    def normalSample(self):
        return Variable(torch.randn(self.batch_size, self.z_size))


    def compute_mu(self, g, rng, delta):
        rng_t, delta_t = align(rng, delta)
        tmp = (rng_t.float() - self.N / 2 - 0.5) * delta_t 
        tmp_t, g_t = align(tmp,g)
        mu = tmp_t + g_t
        return mu

    def filterbank(self, gx, gy, sigma2, delta):

        rng = Variable(torch.arange(0, self.N).view(1, -1)) 
        mu_x = self.compute_mu(gx,rng,delta)
        mu_y = self.compute_mu(gy, rng, delta)

        a = Variable(torch.arange(0, self.A).view(1, 1, -1)) 
        b = Variable(torch.arange(0, self.B).view(1, 1, -1)) 

        mu_x = mu_x.view(-1, self.N, 1)
        mu_y = mu_y.view(-1, self.N, 1)
        sigma2 = sigma2.view(-1, 1, 1)
        Fx = self.filterbank_matrices(a.float(), mu_x, sigma2)
        Fy = self.filterbank_matrices(b.float(), mu_y, sigma2)
        return Fx,Fy

    def forward(self, x):
        #self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size)) 
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size)) 

        enc_state = Variable(torch.zeros(self.batch_size,self.enc_size)) 
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size)) 
        for t in range(self.T):
            if t == 0:
                c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) 
            else:
                c_prev = self.cs[t-1]
            x_hat = x - self.sigmoid(c_prev)     
            r_t = self.read(x, x_hat, h_dec_prev)
            h_enc_prev, enc_state = self.encoder(torch.cat((r_t, h_dec_prev), 1),(h_enc_prev, enc_state))
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc_prev)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

    def loss(self, x):
        self.forward(x)
        criterion = nn.BCELoss()
        x_recons = self.sigmoid(self.cs[-1])
        Lx = criterion(x_recons, x) * self.A * self.B
        Lz = 0
        kl_terms = [0] * T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))   
            kl_terms[t] = 0.5 * torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - self.T * 0.5
            Lz += kl_terms[t]

        Lz = torch.mean(Lz)   
        loss = Lz + Lx    
        return loss


    def filterbank_matrices(self, a, mu_x, sigma2, epsilon=1e-9):
        t_a,t_mu_x = align(a, mu_x)
        temp = t_a - t_mu_x
        temp, t_sigma = align(temp, sigma2)
        temp = temp / (t_sigma * 2)
        F = torch.exp(-torch.pow(temp, 2))
        F = F / (F.sum(2, True).expand_as(F) + epsilon)
        return F


    def attn_window(self, h_dec):
        params = self.dec_linear(h_dec)
        gx_, gy_, log_sigma_2, log_delta, log_gamma = params.split(1, 1) 

        gx = (self.A + 1) / 2 * (gx_ + 1)  
        gy = (self.B + 1) / 2 * (gy_ + 1) 
        delta = (max(self.A, self.B) - 1) / (self.N - 1) * torch.exp(log_delta) 
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma2, delta), gamma

    def read(self, x, x_hat, h_dec_prev):
        (Fx,Fy),gamma = self.attn_window(h_dec_prev)
        def filter_img(img, Fx, Fy, gamma, A, B, N):
            Fxt = Fx.transpose(2, 1)
            img = img.view(-1, B, A)
            glimpse = Fy.bmm(img.bmm(Fxt))
            glimpse = glimpse.view(-1, N * N)
            return glimpse * gamma.view(-1, 1).expand_as(glimpse)
        x = filter_img(x, Fx, Fy, gamma, self.A, self.B, self.N)
        x_hat = filter_img(x_hat, Fx, Fy, gamma, self.A, self.B, self.N)
        return torch.cat((x, x_hat), 1)


    def write(self, h_dec=0):
        w = self.dec_w_linear(h_dec)
        w = w.view(self.batch_size, self.N, self.N)
        (Fx,Fy),gamma = self.attn_window(h_dec)
        Fyt = Fy.transpose(2, 1)
        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size, self.A * self.B)
        return wr / gamma.view(-1, 1).expand_as(wr)
 
    def sampleQ(self, h_enc):
        e = self.normalSample()
        mu = self.mu_linear(h_enc)          
        log_sigma = self.sigma_linear(h_enc) 
        sigma = torch.exp(log_sigma)
        return mu + sigma * e , mu , log_sigma, sigma

    def generate(self):
        with torch.no_grad():
            h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size))
            dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))

        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            z = self.normalSample()
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
        imgs = []
        for img in self.cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return imgs
    
   
    
    def start(self, epoch_num, phase):

        dataloader = self.dataloaders[phase]
        avg_loss = 0
        count = 0
        for epoch in range(epoch_num):
            for i in range(int(len(dataloader.data)/batch_size)):
                data = dataloader.next_batch(batch_size)
                data = np.reshape(data, (data.shape[0], 1, A, B))
                data = torch.Tensor(data)

                bs = data.size()[0]
                data = Variable(data).view(bs, -1)
                loss = model.loss(data)
                avg_loss += loss.cpu().data.numpy()
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                    optimizer.step()
                    
                count += 1
                a, b = (100, 3000) if phase == 'train' else (25, 750)
                if count % a == 0:
                    print('Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / 100))
                    print("Phase: %s | Epoch: %s | Start Time: %s" % (phase, epoch, time.strftime("%H:%M:%S")))
                    if count % b == 0:
                        if phase == 'train':
                            torch.save(model.state_dict(),'save/weights_%d.tar'%(count))
                        generate_image(count)
                    avg_loss = 0
        if phase == 'train':
            torch.save(model.state_dict(), self.final_output_path)
        generate_image(count)

def provider(path, phase):
    x_train, x_test, y_train, y_test = split_data(path)
    data = Dataset(x_train) if phase == "train" else Dataset(x_test)
    return data        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='bankster')
    parser.add_argument('-rp', '--rootpath')
    parser.add_argument('-phase', '--phase') # 'train', 'test'
    
    args = parser.parse_args()
    root_path = args.rootpath
    if not root_path:
        root_path = os.getcwd()
    phase = args.phase
    if not phase:
        phase = 'test'
 

    path = os.path.join(root_path, config['DRAW']['path'])

    T = int(config['DRAW']['T'])
    batch_size = int(config['DRAW']['batch_size'])
    A = int(config['DRAW']['A'])
    B = int(config['DRAW']['B'])
    z_size = int(config['DRAW']['z_size'])
    N = int(config['DRAW']['N'])
    dec_size = int(config['DRAW']['dec_size'])
    enc_size = int(config['DRAW']['enc_size'])
    epoch_num = int(config['DRAW']['epoch_num'])
    learning_rate = float(config['DRAW']['learning_rate'])
    beta1 = float(config['DRAW']['beta1'])
    USE_CUDA = eval(config['DRAW']['USE_CUDA'])
    clip = float(config['DRAW']['clip'])
    
    model = DRAW(T, A, B, batch_size, z_size, N, dec_size, enc_size, path)
    
    if phase == 'train':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        #if USE_CUDA:
        #    model.cuda()

        model.start(epoch_num=epoch_num, phase='train')
    
    torch.set_default_tensor_type('torch.FloatTensor')
    model = DRAW(T, A, B, z_size, N, dec_size, enc_size, path)
    
    #if USE_CUDA:
    #    model.cuda()
    
    state_dict = torch.load('save/weights_final.tar', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.start(phase='test')

    
 
    
    
    





