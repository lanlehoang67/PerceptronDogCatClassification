import numpy as np 

a = np.array([
    1,2,3
])
w = np.array([
    [3,5],
    [6,7],
    [9,6]
])
print(np.dot(a,w).shape)