import math
import torch
from torch import nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Contracting(nn.Module):
    def __init__(self, in_channels, out_channels, n_conditions, kernel_size=3):
        super(Contracting, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels,  out_channels, kernel_size=kernel_size, padding=1)
        self.activation = Swish()
        self.time_embedding = TimeEmbedding(out_channels)
        self.cond_embedding = ConditionEmbedding(out_channels, n_conditions)

    def forward(self, x, t, c):
        x = self.conv1(x)
        x = self.activation(x)
        x += self.time_embedding(t)[:, :, None, None] + self.cond_embedding(c)[:, :, None, None]
        x = self.conv2(x)
        x = self.activation(x)
        return x
    

class Bottleneck(nn.Module):
    def __init__(self, input_channels, n_conditions, kernel_size=3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=kernel_size, padding=1)
        self.time_embedding = TimeEmbedding(input_channels * 2)
        self.cond_embedding = ConditionEmbedding(input_channels * 2, n_conditions)
        self.activation = Swish()

    def forward(self, x, t, c):
        x = self.conv1(x)
        x = self.activation(x)
        x += self.time_embedding(t)[:, :, None, None] + self.cond_embedding(c)[:, :, None, None]
        x = self.conv2(x)
        x = self.activation(x)
        return x
    

class Expanding(nn.Module):
    def __init__(self, in_channels, out_channels, n_conditions, kernel_size=3):
        super(Expanding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.activation = Swish()
        self.time_embedding = TimeEmbedding(in_channels)
        self.cond_embedding = ConditionEmbedding(in_channels, n_conditions)

    def forward(self, x, skip_con_x, t, c):
        upsample = nn.Upsample(size=skip_con_x.shape[2:], mode='bilinear', align_corners=True)
        x = upsample(x)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv1(x)
        x = self.activation(x)
        x += self.time_embedding(t)[:, :, None, None] + self.cond_embedding(c)[:, :, None, None]
        x = self.conv2(x)
        x = self.activation(x)
        return x
    
    
class ConditionEmbedding(nn.Module):
    def __init__(self, embedding_dim, n_conditions):
        super().__init__()
        self.embedding = nn.Embedding(n_conditions, embedding_dim)
        self.lin1 = nn.Linear(embedding_dim, embedding_dim)
        self.act = Swish()
        self.lin2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, conditions):
        embed = self.embedding(conditions)
        embed = self.act(self.lin1(embed))
        embed = self.lin2(embed)
        return embed 
    
    
class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    This class is from labml.nn
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, features, n_conditions):
        super(Unet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cond_embedding = ConditionEmbedding(features[0], n_conditions)

        # Down part of the Unet
        for feature in features:
            self.downs.append(Contracting(in_channels, feature, n_conditions))
            in_channels = feature
            
        self.bottleneck = Bottleneck(features[-1], n_conditions)

        # Up part of the Unet
        for feature in reversed(features[:-1]):
            self.ups.append(Expanding(in_channels, feature, n_conditions))
            in_channels = feature
        self.ups.append(Expanding(in_channels, features[0], n_conditions))
        
        self.conv_final = nn.Conv2d(features[0], out_channels, kernel_size=1)
            
    def forward(self, x, t, c):
        skip_connections = []
        for down in self.downs:
            x = down(x, t, c)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x, t, c)
        skip_connections = skip_connections[::-1]
        
        for idx, up in enumerate(self.ups):
            skip_connection = skip_connections[idx]
            x = up(x, skip_connection, t, c)
            
        return self.conv_final(x + self.cond_embedding(c)[:, :, None, None])


