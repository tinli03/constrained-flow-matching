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

def sample_training_batch(i):
    data = pd.read_csv("target_samples.csv", header=None)
    x1 = data.iloc[i]

    source = pd.read_csv("source_samples.csv", header=None)
    x0 = source.iloc[i]

    t = np.random.uniform(0, 1)
    xt = ((1-t)*x0 + t*x1).tolist()

    target = (x1 - x0).tolist()

    return xt, t, target



#print(input_points(x0, x1))
#print(velocity_data(x0, x1))
print(sample_training_batch(0))



#xt, t = input_points(x0, x1)
#vektor = np.array([1, 2, 3, 4])
#lista = vektor.tolist()