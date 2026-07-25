# creates the target points and into a csv file called target.csv

import csv
import numpy as np

from source_to_file import N_DIM 



def target1():
    alpha_1 = np.array([8, 8, 8, 8, 8, 1, 1, 1, 1, 1])
    x1_a = np.random.dirichlet(alpha_1)
    return x1_a

def target2():
    alpha_2 = np.array([1, 1, 1, 1, 1, 8, 8, 8, 8, 8])
    x1_b = np.random.dirichlet(alpha_2)
    return x1_b


def create_csv_target(number_of_targets): # skapar en csv med alla fasta, source punkter som används vid evaluation för alla metoder
    with open("target.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for n in range(number_of_targets): 
            if n % 2 != 0:
                one_list = (target1())
            else:
                one_list = (target2())
            writer.writerow(one_list)


