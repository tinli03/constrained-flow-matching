import csv
import numpy as np
import pandas as pd 



def source(number_of_dim): # one list/vector with 10 dimensions from dirichlet distribution
    alpha_source = np.ones(number_of_dim)
    x0 = np.random.dirichlet(alpha_source, size=1)
    return x0

def n_dims():
    data = pd.read_csv("source_samples.csv", header=None)
    x1 = data.iloc[0].tolist()
    n_dims = len(x1)
    return n_dims

def create_source(number_of_lists):
    with open("source_samples.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for _ in range(number_of_lists):
            one_list = source(10)
            writer.writerows(one_list)

