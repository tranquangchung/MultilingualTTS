import torch
from torch.nn.parameter import Parameter
import torch.nn as nn


model = [Parameter(torch.randn(2, 2, requires_grad=True))]
# optimizer = torch.optim.SGD(model, 0.1)
optimizer = torch.optim.AdamW(model, lr=0.1, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for epoch in range(100):
    optimizer.zero_grad()
    optimizer.step()
    if epoch % 10 == 0:
        scheduler.step()
    print(scheduler.get_lr())
