# inferring using trained model
import torch
import csv
import numpy as np
from input import sample_training_batch
from model import FlowNetwork
from source_to_file import N_DIM 
from utils import to_tensor
import matplotlib.pyplot as plt
from train import parse_args
from input import sample_training_batch 
from source_to_file import source

args = parse_args()
model = FlowNetwork(args, N_DIM)
model.load_state_dict(torch.load(args.checkpoint_path))


def tensor_to_csv(tensor_matrix, number_of_steps, method_name): # ger ut i CSV alla slutpunkter från data.csv
    list = tensor_matrix.tolist()
    filename = f"{number_of_steps}steps_{method_name}_generated.csv"
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for n in range(len(list)):
            one_list = list[n]
            writer.writerow(one_list)


def tensor_from_source(filename): # läser av en csv och gör en tensor för att användas i genereringen av samples
    list = []
    data = np.loadtxt(filename, delimiter=",")
    for n in range(data.shape[0]):
        list.append(data[n])
    source_matrix = np.array(list)
    source_tensor = to_tensor(source_matrix)

    return source_tensor

def create_csv_source(number_of_sources): # skapar en csv med alla fasta, source punkter som används vid evaluation för alla metoder
    with open("data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for _ in range(number_of_sources):
            one_list = source(10)
            writer.writerow(one_list)




def sample_unconstrained(model, n_steps, filename): # ger ut i TERMINALEN alla slutpunkter från data.csv
    dt = 1.0 / n_steps 
    x = tensor_from_source(filename)
    model.eval()

    with torch.no_grad():
        for k in range(n_steps):
            t = torch.full((x.shape[0],), k * dt, dtype=x.dtype,device=x.device) # hur långt tidsmässigt vi har kommit fram 
            v = model(x, t)
            x += dt * v

    return x



filename = "data.csv"
method_name = "unconstrained"


number_of_steps = 100
generated = sample_unconstrained(model, number_of_steps, filename)
print(tensor_to_csv(generated, number_of_steps, method_name))




#print(create_csv_source(10000)) # number of sources