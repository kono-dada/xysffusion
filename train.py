import torch
from torch.utils.data import DataLoader
from diffusion import CondDenoisingDiffusion
from torchvision.utils import save_image
import random
import os
from data import union_dataset
from data.font import valid_chars

# set cuda device 3
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

image_size = 64
batch_size = 128
font_path = 'font'
# list the fonts in a folder
font_paths = [os.path.join(font_path, font) for font in os.listdir(font_path)]
datasets, n_classes = union_dataset(font_paths, image_size=image_size)
print('Total number of the characters:', n_classes)


dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
print('Total number of the batches:', len(dataloader))


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
def sample(model: CondDenoisingDiffusion, epoch, n_classes):
    model.eval()
    # random sample 40 numbers from 0..n_classes
    conditions = torch.tensor(random.sample(list(range(n_classes)), 40))
    print([valid_chars[i] for i in conditions])
    conditions = conditions.cuda()
    z = torch.randn((40, 1, image_size, image_size)).cuda()
    sample = model.cond_sample(z, conditions)
    # create the folder if not exists
    if not os.path.exists('./samples_cond2'):
        os.makedirs('./samples_cond2')
    save_image(sample, f'./samples_cond2/sample_{epoch}.png', nrow=10)
    

model = CondDenoisingDiffusion(device='cuda', n_conditions=n_classes, features=[128, 256, 512]).cuda()
# load the checkpoint if exists
# if os.path.exists('./cond_checkpoints'):
#     model.load_state_dict(torch.load('./cond_checkpoints/checkpoint_90.pth'))
#     print('Load the checkpoint successfully!')

epochs = 180
learning_rate = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1)
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(dataloader, model, optimizer)
    scheduler.step()
    sample(model, t+1, n_classes)
    if (t+1) % 30 == 0:
        # create the folder if not exists
        if not os.path.exists('./cond_checkpoints2'):
            os.makedirs('./cond_checkpoints2')
        torch.save(model.state_dict(), f'./cond_checkpoints2/checkpoint_{t+1}.pth')

print('Done!')