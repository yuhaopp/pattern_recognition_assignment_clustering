import numpy as np
import random

a = np.array([[2, 3], [2, 3, 12], [2, 3, 3, 4], [3]]).tolist()
print(random.sample(a, 2))
