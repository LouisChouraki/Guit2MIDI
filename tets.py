import torch
from torch import nn

tensor = torch.empty((1,1,9,1))
result = nn.Conv2d(1, 48, (3, 3), padding=(0,1), dilation=(3,1))(tensor)

print()
