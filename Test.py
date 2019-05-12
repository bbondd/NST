import numpy as np

a = np.array([[0, 1], [3, 5]])
print(np.expand_dims(a, axis=0).shape)