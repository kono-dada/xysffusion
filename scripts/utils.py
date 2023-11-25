from torch.utils.data import DataLoader
from diffusion import CondDenoisingDiffusion
import torch
from torchvision.image import save_image


characters = [chr(i) for i in range(0x4E00, 0x9FFF+1)]


def train(dataloader: DataLoader, model: CondDenoisingDiffusion, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y= X.cuda(), y.cuda()
        loss = model(X, y)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print loss
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}, [{current:>5d}/{size:>5d}]')
            
    
@torch.no_grad()    
def sample(model: CondDenoisingDiffusion, epoch):
    model.eval()
    conditions = torch.arange(10).repeat(4).cuda()
    z = torch.randn((40, 1, 28, 28)).cuda()
    sample = model.cond_sample(z, conditions)
    save_image(sample, f'./samples_cond/sample_{epoch}.png', nrow=10)