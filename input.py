import numpy as np
import csv
import pandas as pd
from source_to_file import source
from target_to_file import target1, target2

x0 = np.array([1, 2, 3, 4])
x1 = np.array([1, 1, 1, 2])


def input_points(x0, x1):
    t = np.random.uniform(0, 1)
    xt = (1-t)*x0 + t*x1
    return xt, t

def velocity_data(x0, x1):
    v_d = x1 - x0
    return v_d

def sample_training_batch(batch_size):
    xt_list = []
    target_list = []
    t_list = []

    for n in range(batch_size):
        x0 = np.array(source(10))
        if n % 2 != 0:
            x1 = np.array(target1())
        else:
            x1 = np.array(target2())
        
        target = x1 - x0
        target_list.append(target.tolist())

        t = np.random.uniform(0, 1)
        t_list.append(t)

        xt = (1 - t) * x0 + t * x1
        xt_list.append(xt.tolist())

    xt_stack = np.array(xt_list)
    t_stack = np.array(t_list)
    target_stack = np.array(target_list)

    return xt_stack, t_stack, target_stack



#print(input_points(x0, x1))
#print(velocity_data(x0, x1))
#print(sample_training_batch(4, 0))



#xt, t = input_points(x0, x1)
#vektor = np.array([1, 2, 3, 4])
#lista = vektor.tolist()