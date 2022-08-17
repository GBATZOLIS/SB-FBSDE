import torch
import numpy as np

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

grid_x, grid_y = torch.meshgrid(x, y)

def create_mesh_points(x,y):
    z = []
    for i in range(x.size(0)):
        for j in range(y.size(0)):
            z.append([x[i], y[j]])
    return torch.tensor(z)

print(create_mesh_points(x,y))
