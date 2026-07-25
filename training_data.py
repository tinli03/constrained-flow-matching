# creates the input for the NN training, includes produced targets, xt, and t

import numpy as np
from source import source
from target import target1, target2




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


