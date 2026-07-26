# inferring using trained model
import torch
import csv
import numpy as np
import time

from model import FlowNetwork
from source import N_DIM # dimension of both source and target vectors
from utils import to_tensor
from train import parse_args
from paths import SEEDS, METHODS, source_path, generated_path, ensure_dirs, checkpoint_path


def load_model(distr="dirichlet", checkpoint=None):
    # Modellen hämtas alltid utifrån fördelningen, så dirichletmodellen aldrig
    # av misstag körs på gaussiska källpunkter (det kraschar inte, det blir bara fel).
    args = parse_args()
    model = FlowNetwork(args, N_DIM)
    model.load_state_dict(torch.load(checkpoint or checkpoint_path(distr)))
    model.eval()
    return model


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


def list_to_csv(list_of_all, filename): # ger ut i CSV alla slutpunkter, filename kommer från paths.generated_path
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

SAMPLERS = {
    "unc": sample_unconstrained,
    "fpr": sample_finalproj,
    "sbs": sample_stepbystepproj,
}

number_of_steps = 100


def run_sweep(seeds=SEEDS, methods=METHODS, n_steps=number_of_steps, distr="dirichlet", model=None):
    # Genererar alla seeds x metoder. Filnamnen kommer alltid från paths.py.
    ensure_dirs()
    if model is None:
        model = load_model(distr)
    for seed in seeds:
        src = source_path(seed, distr)
        for method in methods:
            out = generated_path(seed, method, distr)
            points = SAMPLERS[method](model, n_steps, src)[0]
            list_to_csv(points, out)
            print(f"wrote {out}")


if __name__ == "__main__":
    run_sweep(distr=parse_args().distr)
