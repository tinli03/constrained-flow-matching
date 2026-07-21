import csv
import numpy as np



def target_pair(): # one list/vector with 10 dimensions from dirichlet distribution
    alpha_1 = np.array([8, 8, 8, 8, 8, 1, 1, 1, 1, 1])
    alpha_2 = np.array([1, 1, 1, 1, 1, 8, 8, 8, 8, 8])
    x1_a = np.random.dirichlet(alpha_1, size=1)
    x1_b = np.random.dirichlet(alpha_2, size=1)

    target_pair = np.vstack([x1_a, x1_b])
    np.random.shuffle(target_pair)

    return target_pair


number_of_lists = 6 # half as many as for the source

with open("target_samples.csv", "w", newline="") as file:

    writer = csv.writer(file)

    for _ in range(number_of_lists):

        one_list = target_pair()

        writer.writerows(one_list)
