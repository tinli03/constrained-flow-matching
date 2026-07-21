import numpy as np

x0 = np.array([1, 2, 3, 4])
x1 = np.array([1, 1, 1, 2])


def input_points(x0, x1):
    t = np.random.uniform(0, 1)
    xt = (1-t)*x0 + t*x1
    return xt, t

def velocity_data(x0, x1):
    v_d = x1 - x0
    return v_d
    

print(input_points(x0, x1))
print(velocity_data(x0, x1))



#xt, t = input_points(x0, x1)
#vektor = np.array([1, 2, 3, 4])
#lista = vektor.tolist()