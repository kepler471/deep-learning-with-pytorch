import torch

a = torch.ones(3)
a[1]
float(a[1])

points = torch.tensor([4.,1.,5.,3,])

torch.randn(2,3, 3, 3)

batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columns]
batch_t.mean()
batch_t.mean(-2).shape