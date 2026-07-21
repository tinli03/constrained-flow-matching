import csv
import numpy as np



def source(n): # one list/vector with 10 dimensions from dirichlet distribution
    alpha_source = np.ones(n)
    x0 = np.random.dirichlet(alpha_source, size=1)
    return x0



antal_listor = 12

with open("samples.csv", "w", newline="") as file:

    writer = csv.writer(file)

    for _ in range(antal_listor):

        one_list = source(10)

        writer.writerows(one_list)