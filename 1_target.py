import numpy as np

alpha_1 = np.array([8, 8, 8, 8, 8, 1, 1, 1, 1, 1])
alpha_2 = np.array([1, 1, 1, 1, 1, 8, 8, 8, 8, 8])

num_target = 10
num_half = num_target // 2

x1_a = np.random.dirichlet(alpha_1, size=num_half)
x1_b = np.random.dirichlet(alpha_2, size=num_target - num_half)

x1 = np.vstack([x1_a, x1_b])
np.random.shuffle(x1)

print(x1)