import torch
import numpy as np
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

grid_x, grid_y = torch.meshgrid(x, y)


print(torch.vstack([grid_x.ravel(), grid_y.ravel()]).reshape((x.size(0)*y.size(0),-1)))
print(torch.stack((grid_x, grid_x), dim=2).view(-1, 2))