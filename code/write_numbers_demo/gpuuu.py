import torch
from torch import nn

print(torch.device('cpu'), torch.device('cuda'))

print(torch.cuda.device_count())