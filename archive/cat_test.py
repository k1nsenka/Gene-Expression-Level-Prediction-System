import torch


t1 = torch.randn(4, 96, 1024)
print(t1.dtype)
t2 = torch.zeros(4, 24, 1024)
t3 = torch.cat([t1, t2], 1)
print(t3.shape)