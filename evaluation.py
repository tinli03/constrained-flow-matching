import matplotlib.pyplot as plt
import torch
import csv
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def list_from_csv(filename): # läser av en csv och gör en tensor för att användas i genereringen av samples
    data = np.loadtxt(filename, delimiter=",")
    return data.tolist()



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

g_unconstraint_filename = f"100steps_unconstrained_generated.csv"
g_final_projection_filename = f"100steps_finalprojection_generated.csv"
g_stepbystep_projection_filename = f"100steps_stepbystepprojection_generated.csv"
target_filename = f"target.csv" 
unconstraint_list = list_from_csv(g_unconstraint_filename)
finalproj_list = list_from_csv(g_final_projection_filename)
stepbystep_list = list_from_csv(g_stepbystep_projection_filename)

print("Evaluation of unconstraint generated points:")
print("Mass error mean: ", mass_error_mean(unconstraint_list))
print("Negativity violation mean: ", negativity_violation_mean(unconstraint_list))
print("Feasibility rate: ", feasibility_rate2(unconstraint_list), "out of 10 000 are infeasible.")
print("The mode balance is: ", mode_balance(10, g_unconstraint_filename))
print("Swd_value - gen vs tar: ", swd_value(g_unconstraint_filename, target_filename))


print("Evaluation of final projection generated points:")
print("Mass error mean: ", mass_error_mean(finalproj_list))
print("Negativity violation mean: ", negativity_violation_mean(finalproj_list))
print("Feasibility rate: ", feasibility_rate2(finalproj_list), "out of 10 000 are infeasible.")
print("The mode balance is: ", mode_balance(10, g_final_projection_filename))
print("Swd_value - gen vs tar: ", swd_value(g_final_projection_filename, target_filename))

print("Evaluation of step by step generated points:")
print("Mass error mean: ", mass_error_mean(stepbystep_list))
print("Negativity violation mean: ", negativity_violation_mean(stepbystep_list))
print("Feasibility rate: ", feasibility_rate2(stepbystep_list), "out of 10 000 are infeasible.")
print("The mode balance is: ", mode_balance(10, g_stepbystep_projection_filename))
print("Swd_value - gen vs tar: ", swd_value(g_stepbystep_projection_filename, target_filename))


PCA_plot(g_unconstraint_filename, g_final_projection_filename, g_stepbystep_projection_filename, target_filename)
