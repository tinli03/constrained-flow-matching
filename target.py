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
from target_to_file import target1, target2




def create_csv_target(number_of_targets): # skapar en csv med alla fasta, source punkter som används vid evaluation för alla metoder
    with open("target.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for n in range(number_of_targets): 
            if n % 2 != 0:
                one_list = (target1())
            else:
                one_list = (target2())
            writer.writerow(one_list)

create_csv_target(10000)


