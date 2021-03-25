import numpy as np
import torch

a = np.zeros((4, 4))
print(a[1])
b = torch.Tensor(a)
print(b[1])
for i in range(0):
    print('a')
    print(i)

print(np.random.randint(4))