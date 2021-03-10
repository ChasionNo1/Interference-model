import math
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


# x = -1.2446699294613897
# a = 7.5
# b = 10
# print(pow(b, 2)*(1 - pow(x, 2)/pow(a, 2)))
#
#
# a = np.random.random(10)
# print(a)
# np.delete(a, 1)
# print(a)

# a = [[0, 1], [1, 1]]
# print(euclidean_distances(a, a))

a = np.array([1, 2, 3, 4])[:, np.newaxis]
b = np.array([3, 4, 5, 6])[:, np.newaxis]
print(np.hstack([a, b]))
