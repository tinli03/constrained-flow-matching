import csv
import numpy as np

N_DIM = 10

def source(number_of_dim): # one list/vector with 10 dimensions from dirichlet distribution
    alpha_source = np.ones(number_of_dim)
    x0 = np.random.dirichlet(alpha_source)
    return x0

def create_csv_source(number_of_sources):
    with open("data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for _ in range(number_of_sources):
            one_list = source(N_DIM)
            writer.writerows(one_list)

