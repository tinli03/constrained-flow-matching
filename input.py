import numpy as np
import csv
import pandas as pd

x0 = np.array([1, 2, 3, 4])
x1 = np.array([1, 1, 1, 2])


def input_points(x0, x1):
    t = np.random.uniform(0, 1)
    xt = (1-t)*x0 + t*x1
    return xt, t

def velocity_data(x0, x1):
    v_d = x1 - x0
    return v_d

def sample_training_batch(batch_size, i):
    data = pd.read_csv("target_samples.csv", header=None).to_numpy()
    source = pd.read_csv("source_samples.csv", header=None).to_numpy()

    # pick batch_size random rows (with replacement, since the file is small)
    idx = np.random.randint(0, len(data), size=batch_size)
    x1 = data[idx]        # (batch_size, N_DIM)
    x0 = source[idx]      # (batch_size, N_DIM)

    # one t per sample in the batch
    t = np.random.uniform(0, 1, size=(batch_size, 1))  # (batch_size, 1)

    xt = (1 - t) * x0 + t * x1        # (batch_size, N_DIM)
    target = x1 - x0                  # (batch_size, N_DIM)

    return xt, t, target



#print(input_points(x0, x1))
#print(velocity_data(x0, x1))
#print(sample_training_batch(4, 0))



#xt, t = input_points(x0, x1)
#vektor = np.array([1, 2, 3, 4])
#lista = vektor.tolist()