import matplotlib.pyplot as plt
import torch
import csv
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from generate_samples import csv_to_list, sampling_time_un, sampling_time_final, projection_time_final, sampling_time_sbs, projection_time_sbs, number_of_steps



### -----------------------------------------------

### EVALUATION METRICS 
 

# hur mycket avviker från summa 1 i en vektor
def mass_error(u): # vill ha en lista
    x = 0
    for n in range(10):
        x += u[n]
    mass_error = abs(1-x)
    return mass_error

def mass_error_mean(list):
    x = 0
    for n in range(len(list)):
        x += mass_error(list[n])
    mass_error_mean = x / len(list)
    return mass_error_mean


# hur mycket negativa värden och dess summan i en vektor
def negativity_violation(u): # vill ha en lista
    x = 0
    for n in range(10):
        if u[n] < 0:
            x += u[n]
        else:
            continue
    negativity_violation = abs(x)
    return negativity_violation

def negativity_violation_mean(list):
    x = 0
    for n in range(len(list)):
        x += negativity_violation(list[n])
    negativity_violation_mean = x / len(list)
    return negativity_violation_mean


# antal vektorer som ej är feasible
def feasibility_rate2(list): # vill ha listor i listor av alla steg/vektorer
    x = 0
    p = 1e-5
    for rad in range(len(list)):
        sum = 0
        for kolumn in range(len(list[0])):
            sum += list[rad][kolumn]
            if list[rad][kolumn] < -p: 
                x += 1 # count if any x_i is less than 0
                break
            if kolumn == len(list[0]) - 1:
                if abs(sum - 1) > p:
                    x += 1
    return x


# hur många i en csv fylld av vektorer är större ena sidan och större andra sidan, målet är 50 50 som target
def mode_balance(dim, filename): # vill ha en csv med genererade 
    with open(filename, "r", newline="") as file: ##### BYT UT target samples
        reader = csv.reader(file)
        r = 0
        l = 0
        for row in reader:
            sample = [float(value) for value in row]
            if sum(sample[: dim // 2]) > sum(sample[dim // 2:]):
                l += 1
            else:
                r += 1
    left = l / (l+r)
    right = r / (l+r)
    return right, left


# measure how similarly the generated datasets behave compared to the target dataset
def covariance(filename): 
    target = pd.read_csv("target.csv", header=None)
    t_covariance_matrix = target.cov().to_numpy()
    target_cova_mat = t_covariance_matrix.tolist()

    generated = pd.read_csv(filename, header=None) ##### BYT TILL generated csv sen
    g_covariance_matrix = generated.cov().to_numpy()

    difference = (t_covariance_matrix - g_covariance_matrix).tolist()

    E_cov = 0
    for row in range(10):
        for column in range(10):
            E_cov += (difference[row][column])**2
    abs_cova_err = np.sqrt(E_cov)

    norm_target_cova_mat = 0
    for row in range(10):
        for column in range(10):
            norm_target_cova_mat += (target_cova_mat[row][column])**2
    norm_target = np.sqrt(norm_target_cova_mat)
    E_rel = abs_cova_err / norm_target
    
    return E_rel


# plot av hur lika target och genererade är, mål: punkter övertäcker varandra
def PCA_plot(unconstrained_filename,final_projection_filename,stepbystep_projection_filename, target_filename): # vill ha csv för target samt för genererade
    unconstrained_data = pd.read_csv(unconstrained_filename,header=None) ###### change source csv to the generated later # Read the data 
    target_data = pd.read_csv(target_filename, header=None)
    final_projection_data = pd.read_csv(final_projection_filename,header=None)
    sbs_projection_data = pd.read_csv(stepbystep_projection_filename ,header=None)
    combined_data = pd.concat([target_data, unconstrained_data, final_projection_data, sbs_projection_data],ignore_index=True) # Combine the datasets

    pca = PCA(n_components=2) # Fit PCA on the combined data
    pca.fit(combined_data)

    target_pca = pca.transform(target_data) # Transform both datasets using the same PCA
    unconstrained_pca = pca.transform(unconstrained_data)
    final_projection_pca = pca.transform(final_projection_data)
    sbs_projection_pca = pca.transform(sbs_projection_data)

    plt.scatter(target_pca[:, 0],target_pca[:, 1],label="Target",alpha=0.4, s = 10) # Plot the two-dimensional representations
    plt.scatter(unconstrained_pca[:, 0],unconstrained_pca[:, 1],label="Unconstrained",alpha=0.4,s=10)
    plt.scatter(final_projection_pca[:, 0], final_projection_pca[:, 1],label="Final projection",alpha=0.4 , s = 10)
    plt.scatter(sbs_projection_pca[:, 0], sbs_projection_pca[:, 1],label="Step-by-step projection",alpha=0.4 , s = 10)


    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("PCA comparison of target and generated samples")
    plt.legend()
    plt.show()


# ett värde för hur lika de är, ju lägre värde desto bättre
def swd(generated: torch.Tensor, target: torch.Tensor, num_projections: int = 100, seed: int | None = None,) -> torch.Tensor:
    if generated.ndim != 2 or target.ndim != 2:
        raise ValueError( "Both inputs must have shape (number_of_samples, dimension).")
    if generated.shape[1] != target.shape[1]:
        raise ValueError("Generated and target samples must have the same dimension." )
    if generated.shape[0] != target.shape[0]:
        raise ValueError("Generated and target must contain the same number of samples.")
    if generated.device != target.device:
        raise ValueError("Generated and target tensors must be on the same device.")
    if seed is not None:
        generator = torch.Generator(device=generated.device)
        generator.manual_seed(seed)
    else:
        generator = None

    dimension = generated.shape[1]
    directions = torch.randn(num_projections, dimension, device=generated.device, dtype=generated.dtype, generator=generator, ) # skapar slumpmässiga riktningar
    directions = directions / directions.norm(dim=1, keepdim=True, ).clamp_min(1e-12)  # normaliserar riktningarna 
    generated_proj = generated @ directions.T # projicerar på riktningarna
    target_proj = target @ directions.T
    generated_sorted = torch.sort(generated_proj, dim=0,).values   # sorterar projektionerna för att jämföra största med största, minsta med minsta
    target_sorted = torch.sort(target_proj, dim=0,).values
    distances = torch.mean(torch.abs(generated_sorted - target_sorted),dim=0,)  # skillnaden sedan medelvärdet

    return distances.mean()

def swd_value(generated_filename, target_filename): # output is swd for each method
    target_df = pd.read_csv(target_filename, header = None) ###### BYT UT # Läser targetdata från CSV-filen.
    target = torch.tensor(target_df.to_numpy(), dtype=torch.float32,)     # Omvandlar targetdata från en pandas DataFrame till en PyTorch-tensor.
    generated_df = pd.read_csv(generated_filename, header = None)
    generated = torch.tensor(generated_df.to_numpy(), dtype=torch.float32,)
    distance = swd(generated=generated,target=target,num_projections=100,seed=42,)
    
    return distance.item()


### -----------------------------------------------

def print_title(title):
    print()
    print("=" * 60)
    print(title.upper())
    print("=" * 60)

unconstrained_filename = f"100steps_unconstrained_generated.csv"
finalprojection_filename = f"100steps_finalprojection_generated.csv"
stepbystepprojection_filename = f"100steps_stepbystepprojection_generated.csv"
target_filename = f"target.csv" 
unconstrained_list = csv_to_list(unconstrained_filename)
finalprojection_list = csv_to_list(finalprojection_filename)
stepbystepprojection_list = csv_to_list(stepbystepprojection_filename)

print_title("Evaluation of unconstraint generated points:")
print("Mass error mean: ", mass_error_mean(unconstrained_list))
print("Negativity violation mean: ", negativity_violation_mean(unconstrained_list))
print("Feasibility rate: ", feasibility_rate2(unconstrained_list), "out of 10 000 are infeasible.")
print("The mode balance is: ", mode_balance(10, unconstrained_filename))
print("Swd_value: ", swd_value(unconstrained_filename, target_filename))
print("The covarience matrix difference: ", covariance(unconstrained_filename))
print(f"Sampling time: {sampling_time_un:.4f} s")

print_title("Evaluation of final projection generated points:")
print("Mass error mean: ", mass_error_mean(finalprojection_list))
print("Negativity violation mean: ", negativity_violation_mean(finalprojection_list))
print("Feasibility rate: ", feasibility_rate2(finalprojection_list), "out of 10 000 are infeasible.")
print("The mode balance is: ", mode_balance(10, finalprojection_filename))
print("Swd_value: ", swd_value(finalprojection_filename, target_filename))
print("The covarience matrix difference: ", covariance(finalprojection_filename))
print(f"Sampling time : {sampling_time_final:.4f} s")
print(f"Projection time - 1 projection: {projection_time_final:.4f} s")
print(f"Total time     : {sampling_time_final + projection_time_final:.4f} s")


print_title("Evaluation of step by step generated points:")
print("Mass error mean: ", mass_error_mean(stepbystepprojection_list))
print("Negativity violation mean: ", negativity_violation_mean(stepbystepprojection_list))
print("Feasibility rate: ", feasibility_rate2(stepbystepprojection_list), "out of 10 000 are infeasible.")
print("The mode balance is: ", mode_balance(10, stepbystepprojection_filename))
print("Swd_value: ", swd_value(stepbystepprojection_filename, target_filename))
print("The covarience matrix difference: ", covariance(stepbystepprojection_filename))
print(f"Sampling time:   {sampling_time_sbs:.6f} seconds")
print(f"Projection time - 100 projections: {projection_time_sbs:.6f} seconds")
print(f"Mean projection time - mean for 1 projection: {(projection_time_sbs / number_of_steps):.6f} seconds")
print(f"Total time:  {sampling_time_sbs + projection_time_sbs:.4f} s")

PCA_plot(unconstrained_filename, finalprojection_filename, stepbystepprojection_filename, target_filename)

results = {"unconstrained": {1: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},2: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},3: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},4: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},5: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},},
    "finalprojection": {1: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},2: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},3: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},4: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},5: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},},
    "stepbystepprojection": {1: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},2: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},3: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},4: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},5: {"mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,},},}
total_batches = 5
methods = ["unconstrained", "finalprojection", "stepbystepprojection"]
metrics = ["mass_error_mean", "negativity_violation_mean", "feasibility_rate2", "mode_balance", "swd_value", "covariance"]
filenames = ["100steps_unconstrained_generated.csv", "100steps_finalprojection_generated.csv", "100steps_stepbystepprojection_generated.csv"]

 # adds all the filenames
for i in range(1, total_batches + 1):
    for n in range(len(methods)):
        filenames.append(f"d_{i}_{methods[n]}_generated.csv")

print("Mass error mean: ", mass_error_mean(unconstrained_list))
print("Negativity violation mean: ", negativity_violation_mean(unconstrained_list))
print("Feasibility rate: ", feasibility_rate2(unconstrained_list), "out of 10 000 are infeasible.")
print("The mode balance is: ", mode_balance(10, unconstrained_filename))
print("Swd_value: ", swd_value(unconstrained_filename, target_filename))
print("The covarience matrix difference: ", covariance(unconstrained_filename))
unconstrained_filename = f"100steps_unconstrained_generated.csv"
finalprojection_filename = f"100steps_finalprojection_generated.csv"
stepbystepprojection_filename = f"100steps_stepbystepprojection_generated.csv"
target_filename = f"target.csv" 
unconstrained_list = csv_to_list(unconstrained_filename)
finalprojection_list = csv_to_list(finalprojection_filename)
stepbystepprojection_list = csv_to_list(stepbystepprojection_filename)

for number_of_method in range(len(methods)): # 3 st, 0
    for number_of_metric in range(len(metrics)): # 6 st, 0
        for number_of_batch in range(1, total_batches + 1): # 5 st, 1
            if number_of_metric == 1 or 2 or 3:
                function = csv_to_list(filenames[number_of_metric * 3])
                value = f"{metrics[number_of_metric]}(csv_to_list(filenames[number_of_metric * 3]))"
            if number_of_metric == 6:
                value = f"{metrics[number_of_metric]}{csv_to_list(filenames[number_of_metric * 3])}"
            results[methods[number_of_method]][number_of_batch][metrics[number_of_metric]] =  value


results["unc"][1].append(0.136)
    


# "mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,

#import numpy as np
#mean_swd = np.mean(unconstrained_swd)
#std_swd = np.std(unconstrained_swd, ddof=1)
#print(f"Unconstrained SWD: {mean_swd:.3f} ± {std_swd:.3f}")
