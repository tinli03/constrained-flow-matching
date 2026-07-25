# inferring using trained model
import torch
import csv
import numpy as np
import time

from model import FlowNetwork
from source import N_DIM # dimension of both source and target vectors
from utils import to_tensor
from train import parse_args

args = parse_args()
model = FlowNetwork(args, N_DIM)
model.load_state_dict(torch.load(args.checkpoint_path))


### -----------------------------------------------

def projection(vector):
    vector = np.asarray(vector, dtype=float)
    sorted_vector = np.sort(vector)[::-1] # sorting from largest to smallest
    cumulative_sum = np.cumsum(sorted_vector) # cumulative_sum
    indices = np.arange(1, len(vector) + 1)
    condition = (
        sorted_vector
        - (cumulative_sum - 1) / indices
        > 0
    ) # s = (the first cumulative sum - 1) DIVIDED with 1 (first vector)
    # if the first sorted vector - s larger than 0 then TRUE
    rho = indices[condition][-1] # how many should remain positive
    theta = (
        cumulative_sum[rho - 1] - 1
    ) / rho
    projected = np.maximum(vector - theta, 0)

    return projected


def csv_to_tensor(filename): # läser av en csv och gör en tensor för att användas i genereringen av samples
    list = []
    data = np.loadtxt(filename, delimiter=",")
    for n in range(data.shape[0]):
        list.append(data[n])
    source_matrix = np.array(list)
    source_tensor = to_tensor(source_matrix)

    return source_tensor


def list_to_csv(list_of_all, number_of_steps, method_name): # ger ut i CSV alla slutpunkter från data.csv
    filename = f"{number_of_steps}steps_{method_name}_generated.csv"
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for n in range(len(list_of_all)):
            one_list = list_of_all[n]
            writer.writerow(one_list)


def csv_to_list(filename): 
    data = np.loadtxt(filename, delimiter=",")
    return data.tolist()

### -----------------------------------------------

### METHODS

def sample_unconstrained(model, n_steps, filename): # input tensor, output lists
    dt = 1.0 / n_steps 
    x = csv_to_tensor(filename)
    model.eval()

    start = time.perf_counter()

    with torch.no_grad():
        for k in range(n_steps):
            t = torch.full((x.shape[0],), k * dt, dtype=x.dtype,device=x.device) # hur långt tidsmässigt vi har kommit fram 
            v = model(x, t)
            x += dt * v
    list_of_all = x.tolist()

    sampling_time = time.perf_counter() - start

    return list_of_all, sampling_time


def sample_finalproj(model, n_steps, filename): # input tensor, output lists
    list_of_all = sample_unconstrained(model, n_steps, filename)[0]
    sampling_time = sample_unconstrained(model, n_steps, filename)[1]
    start = time.perf_counter()

    for row in range(len(list_of_all)):
        list = list_of_all[row]
        list_of_all[row] = projection(list)

    projection_time = time.perf_counter() - start

    return list_of_all, sampling_time, projection_time


def sample_stepbystepproj(model, n_steps, filename): # input tensor, output lists
    dt = 1.0 / n_steps 
    x = csv_to_tensor(filename)
    model.eval()
    sampling_time = 0.0
    projection_time = 0.0

    with torch.no_grad():
        for k in range(n_steps):
            t = torch.full((x.shape[0],), k * dt, dtype=x.dtype,device=x.device) # hur långt tidsmässigt vi har kommit fram 
            
            sampling_start = time.perf_counter()
            v = model(x, t)
            x += dt * v
            sampling_time += time.perf_counter() - sampling_start
            
            projection_start = time.perf_counter()
            list_of_all = x.tolist()
            for row in range(len(list_of_all)):
                sample = list_of_all[row]
                list_of_all[row] = projection(sample).tolist()
            x = torch.tensor(list_of_all,dtype=x.dtype,device=x.device)
            projection_time += time.perf_counter() - projection_start

    list_of_all = x.tolist()

    return list_of_all, sampling_time, projection_time
    
### -----------------------------------------------



filename = "data.csv" # created with create_csv_source(10 000)
number_of_steps = 100

generated = sample_unconstrained(model, number_of_steps, filename)[0]
list_to_csv(generated, number_of_steps, "unconstrained")
final_proj = sample_finalproj(model, number_of_steps, filename)[0]
list_to_csv(final_proj, number_of_steps, "finalprojection")
stepbystep_proj = sample_stepbystepproj(model, number_of_steps, filename)[0]
list_to_csv(stepbystep_proj, number_of_steps, "stepbystepprojection")


generated, sampling_time_un = sample_unconstrained(model, number_of_steps, filename)

generated, sampling_time_final, projection_time_final = sample_finalproj(model, number_of_steps, filename)

samples, sampling_time_sbs, projection_time_sbs = sample_stepbystepproj(model, number_of_steps, filename)
