import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils
import torch.nn.utils
from numpy import reshape
import time
import argparse
import os
import sys
from configparser import ConfigParser

base_path = os.getcwd().split('sliver-maestro')[0]
base_path = os.path.join(base_path, "sliver-maestro")
sys.path.insert(1, base_path)
from src.utils.model_utils import *

config = ConfigParser()
cfg_file = os.path.join(base_path, "src/config.cfg")
config.read(cfg_file)


class DRAW(nn.Module):
    def __init__(self, category, base_path=base_path):
        super(DRAW, self).__init__()
        self.T = int(config['DRAW']['T'])
        self.batch_size = int(config['DRAW']['batch_size'])
        self.A = int(config['DRAW']['A'])
        self.B = int(config['DRAW']['B'])
        self.z_size = int(config['DRAW']['z_size'])
        self.N = int(config['DRAW']['N'])
        self.dec_size = int(config['DRAW']['dec_size'])
        self.enc_size = int(config['DRAW']['enc_size'])
        self.epoch_num = int(config['DRAW']['epoch_num'])
        self.learning_rate = float(config['DRAW']['learning_rate'])
        self.beta1 = float(config['DRAW']['beta1'])
        self.USE_CUDA = eval(config['DRAW']['USE_CUDA'])
        self.clip = float(config['DRAW']['clip'])
        self.padsize = int(config['DRAW']['padsize'])
        self.padval = float(config['DRAW']['padval'])
        self.ph = self.B + 2 * self.padsize
        self.pw = self.A + 2 * self.padsize

        self.cs = [0] * self.T
        self.logsigmas, self.sigmas, self.mus = [0] * self.T, [0] * self.T, [0] * self.T

        self.encoder = nn.LSTMCell(2 * self.N * self.N + self.dec_size, self.enc_size)
        self.mu_linear = nn.Linear(self.dec_size, self.z_size)
        self.sigma_linear = nn.Linear(self.dec_size, self.z_size)

        self.decoder = nn.LSTMCell(self.z_size, self.dec_size)
        self.dec_linear = nn.Linear(self.dec_size, 5)
        self.dec_w_linear = nn.Linear(self.dec_size, self.N * self.N)

        self.sigmoid = nn.Sigmoid()

        self.path = os.path.join(base_path, "src", config['DRAW']['path'])
        self.category = category
        self.phases = ["train", "test"]
        self.dataloaders = {
            phase: self.provider(
                path=self.path,
                category=self.category,
                phase=phase
            )
            for phase in self.phases
        }

        base_path = os.getcwd().split('sliver-maestro')[0]
        self.final_output_path = os.path.join(base_path,
                                              "sliver-maestro",
                                              "src",
                                              "save",
                                              category,
                                              '%s_weights.tar' % category)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))

    def normalSample(self):
        torch.manual_seed(0)
        return Variable(torch.randn(self.batch_size, self.z_size))

    def compute_mu(self, g, rng, delta):
        rng_t, delta_t = align(rng, delta)
        tmp = (rng_t.float() - self.N / 2 - 0.5) * delta_t
        tmp_t, g_t = align(tmp, g)
        mu = tmp_t + g_t
        return mu

    def filterbank(self, gx, gy, sigma2, delta):

        rng = Variable(torch.arange(0, self.N).view(1, -1))
        mu_x = self.compute_mu(gx, rng, delta)
        mu_y = self.compute_mu(gy, rng, delta)

        a = Variable(torch.arange(0, self.A).view(1, 1, -1))
        b = Variable(torch.arange(0, self.B).view(1, 1, -1))

        mu_x = mu_x.view(-1, self.N, 1)
        mu_y = mu_y.view(-1, self.N, 1)
        sigma2 = sigma2.view(-1, 1, 1)
        Fx = self.filterbank_matrices(a.float(), mu_x, sigma2)
        Fy = self.filterbank_matrices(b.float(), mu_y, sigma2)
        return Fx, Fy

    def forward(self, x):
        # self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        enc_state = Variable(torch.zeros(self.batch_size, self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        for t in range(self.T):
            if t == 0:
                c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B))
            else:
                c_prev = self.cs[t - 1]
            x_hat = x - self.sigmoid(c_prev)
            r_t = self.read(x, x_hat, h_dec_prev)
            h_enc_prev, enc_state = self.encoder(torch.cat((r_t, h_dec_prev), 1), (h_enc_prev, enc_state))
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
        kl_terms = [0] * self.T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            kl_terms[t] = 0.5 * torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - self.T * 0.5
            Lz += kl_terms[t]

        Lz = torch.mean(Lz)
        loss = Lz + Lx
        return loss

    def filterbank_matrices(self, a, mu_x, sigma2, epsilon=1e-9):
        t_a, t_mu_x = align(a, mu_x)
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
        (Fx, Fy), gamma = self.attn_window(h_dec_prev)

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
        (Fx, Fy), gamma = self.attn_window(h_dec)
        Fyt = Fy.transpose(2, 1)
        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size, self.A * self.B)
        return wr / gamma.view(-1, 1).expand_as(wr)

    def sampleQ(self, h_enc):
        e = self.normalSample()
        mu = self.mu_linear(h_enc)
        log_sigma = self.sigma_linear(h_enc)
        sigma = torch.exp(log_sigma)
        return mu + sigma * e, mu, log_sigma, sigma


    def start(self, phase):

        dataloader = self.dataloaders[phase]
        avg_loss = 0
        count = 0
        for epoch in range(self.epoch_num):
            for i in range(int(len(dataloader.data) / self.batch_size)):
                data = dataloader.next_batch(self.batch_size)
                data = reshape(data, (data.shape[0], 1, self.A, self.B))
                data = torch.Tensor(data)

                bs = data.size()[0]
                data = Variable(data).view(bs, -1)
                loss = self.loss(data)
                avg_loss += loss.cpu().data.numpy()
                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), self.clip)
                    self.optimizer.step()

                count += 1
                a, b = (100, 3000) if phase == 'train' else (25, 750)
                if count % a == 0:
                    print("Phase: %s | Epoch: %s | Count: %d \ Start Time: %s | Loss: %d" % (
                        phase, epoch, count, time.strftime("%H:%M:%S"), avg_loss / 100))

                    avg_loss = 0
        if phase == 'train':
            torch.save(self.state_dict(), self.final_output_path)

    def generate(self, batch_size=False):
        """generates canvases"""
        if not batch_size:
            batch_size = self.batch_size
        with torch.no_grad():
            h_dec_prev = Variable(torch.zeros(batch_size, self.dec_size))
            dec_state = Variable(torch.zeros(batch_size, self.dec_size))

        for t in range(self.T):
            c_prev = Variable(torch.zeros(batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            z = self.normalSample()
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
            
        imgs = []
        
        for img in self.cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy())

        return imgs

    @staticmethod
    def provider(path, category, phase):
        x_train, x_test, y_train, y_test = split_data(path, category)
        data = Dataset(x_train) if phase == "train" else Dataset(x_test)
        return data

