import torch
import torch.utils.data
from torch import nn


class ReshapeLayer(nn.Module):
    def __init__(self, target_shape):
        super(ReshapeLayer, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.reshape(self.target_shape)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.after_conv_shape = 7*7*16
        self.internal_shape = 100
        self.convs = nn.Sequential(nn.Conv2d(1, 8, (3, 3), padding=1),
                                   nn.MaxPool2d((2, 2)),
                                   nn.BatchNorm2d(8),
                                   nn.LeakyReLU(0.1),
                                   nn.Conv2d(8, 16, (3, 3), padding=1),
                                   nn.MaxPool2d((2, 2)),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU(0.1))

        self.fc_mu = nn.Linear(self.after_conv_shape, self.internal_shape)
        self.fc_sigma = nn.Linear(self.after_conv_shape, self.internal_shape)

    def forward(self, x):
        out_conv = self.convs(x)
        return self.fc_mu(out_conv.reshape(-1, self.after_conv_shape)),\
               self.fc_sigma(out_conv.reshape(-1, self.after_conv_shape))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.after_conv_shape = 7*7*16
        self.internal_shape = 100
        self.up_convs = nn.Sequential(nn.Linear(self.internal_shape, self.after_conv_shape),
                                      nn.BatchNorm1d(self.after_conv_shape),
                                      nn.LeakyReLU(0.1),
                                      ReshapeLayer((-1, 16, 7, 7)),

                                      nn.Upsample((14,14)),
                                      nn.Conv2d(16, 8, (3, 3), padding=1),
                                      nn.LeakyReLU(0.1),
                                      nn.BatchNorm2d(8),

                                      nn.Upsample((28, 28)),
                                      nn.Conv2d(8, 1, (3, 3), padding=1),
                                      nn.Sigmoid())

    def simple_forward(self, x):
        return self.up_convs(x)

    def forward(self, mu, log_sigma):
        return self.simple_forward(self.reparametrize(mu, log_sigma))

    def sample_hidden(self, batch_size=1):
        z = torch.randn((batch_size, 100))
        return z

    def reparametrize(self, mu, log_sigma):
        batch_size=mu.shape[0]
        return mu + self.sample_hidden(batch_size) * torch.exp(0.5*log_sigma) #std = torch.exp(0.5 * logvar)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):
        pass