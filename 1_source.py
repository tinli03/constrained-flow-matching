import numpy as np

n = 10

num_samples = 10

alpha_source = np.ones(n)

x0 = np.random.dirichlet(alpha_source, size=num_samples)

print(x0)



#xt = (1 - t) * x0 + t * x1

# target_velocity = x1 - x0

# loss = torch.mean((predicted_velocity - target_velocity) ** 2)
# tar alla element i vektorn, summan genom antalet element