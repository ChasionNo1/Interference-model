"""
模拟干扰模型
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Circle


fig = plt.figure()
ax = fig.add_subplot(111)
ell = Ellipse(xy=(0.0, 0.0), width=15, height=20, facecolor='none', edgecolor='black', linestyle='solid', linewidth=2.0, angle=90, alpha=0.3)
ax.add_patch(ell)
x, y = 0, 0
ax.plot(x, y, 'ro')
# plt.axis('scaled')
plt.axis('equal')
plt.show()
print(ell.get_patch_transform())



