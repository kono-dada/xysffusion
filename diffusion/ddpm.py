import torch
from torch import nn
from .unet import Unet


class CondDenoisingDiffusion(nn.Module):
    def __init__(self, device, n_conditions, in_channels=1, out_channels=1, features=[64, 128, 256], n_steps=500):
        super(CondDenoisingDiffusion, self).__init__()
        self.device = device
        self.unet = Unet(in_channels, out_channels, features, n_conditions)
        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta
        
    def forward(self, x, c):  # c should start from 0
        batchsize = x.shape[0]
        t = torch.randint(self.n_steps, size=(batchsize,))
        alphas = self.alpha_bar[t].view(-1, 1, 1, 1)
        t = t.to(self.device)
        eps = torch.randn_like(x).to(self.device)
        zt = torch.sqrt(alphas) * x + torch.sqrt(1. - alphas) * eps
        predicted_noise = self.unet(zt, t, c)
        loss = nn.functional.mse_loss(predicted_noise, eps)
        return loss
    
    @torch.no_grad()
    def cond_sample(self, z, c):
        size = z.shape[0]
        for t in reversed(range(self.n_steps)):
            z_hat = z / (1 - self.beta[t]) ** 0.5 - self.beta[t] / ((1-self.alpha_bar[t]) ** 0.5 * (1-self.beta[t]) ** 0.5) * \
                self.unet(z, torch.full((size,), t, dtype=torch.long).to(self.device), c)
            if t > 0:
                eps = torch.randn_like(z)
                z = z_hat + self.sigma2[t] ** 0.5 * eps
        return z_hat
    
    
def ddpm_from_config(device, config):
    return CondDenoisingDiffusion(
        device=device,
        n_conditions=config['n_conditions'],
        features=config['features'],
        n_steps=config['n_steps'],
    )