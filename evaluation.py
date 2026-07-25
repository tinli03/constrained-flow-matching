import matplotlib.pyplot as plt
import torch
import csv
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
#from generate_samples import csv_to_list
# sampling_time_un, sampling_time_final, projection_time_final, sampling_time_sbs, projection_time_sbs, number_of_steps
from source import N_DIM


### -----------------------------------------------

### EVALUATION METRICS 
 
def csv_to_list(filename):

    data = np.loadtxt(filename, delimiter=",")

    return data.tolist() 

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
def covariance(generated_filename, target_filename): 
    target = pd.read_csv(target_filename, header=None)
    t_covariance_matrix = target.cov().to_numpy()
    target_cova_mat = t_covariance_matrix.tolist()

    generated = pd.read_csv(generated_filename, header=None) ##### BYT TILL generated csv sen
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

all_lists = []
def evaluation_list(generated_filename, target_filename, batch_number, method):
    one_list = [batch_number, method, mass_error_mean(csv_to_list(generated_filename)), negativity_violation_mean(csv_to_list(generated_filename)), feasibility_rate2(csv_to_list(generated_filename)), mode_balance(N_DIM, generated_filename)[0], mode_balance(N_DIM, generated_filename)[1], swd_value(generated_filename, target_filename), covariance(generated_filename, target_filename)]
    all_lists.append(one_list)
    
    return one_list

def evaluation_list_to_csv(): # ger ut i CSV alla slutpunkter från data.csv
    filename = f"evaluation.csv"
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["batch","method","mass_error_mean","negativity_violation_mean","infeasible_count","mode_balance_right","mode_balance_left","swd","covariance_difference"])

        writer.writerow(evaluation_list("100steps_unconstrained_generated.csv", "target.csv", 1, "unc"))
        writer.writerow(evaluation_list("d_2_unconstrained_generated.csv", "target.csv", 2, "unc" ))
        writer.writerow(evaluation_list("d_3_unconstrained_generated.csv", "target.csv", 3, "unc" ))
        writer.writerow(evaluation_list("d_4_unconstrained_generated.csv", "target.csv", 4, "unc" ))
        writer.writerow(evaluation_list("d_5_unconstrained_generated.csv", "target.csv", 5, "unc" ))

        writer.writerow(evaluation_list("100steps_finalprojection_generated.csv", "target.csv", 1, "fpr"))
        writer.writerow(evaluation_list("d_2_finalprojection_generated.csv", "target.csv", 2, "fpr" ))
        writer.writerow(evaluation_list("d_3_finalprojection_generated.csv", "target.csv", 3, "fpr"))
        writer.writerow(evaluation_list("d_4_finalprojection_generated.csv", "target.csv", 4, "fpr" ))
        writer.writerow(evaluation_list("d_5_finalprojection_generated.csv", "target.csv", 5, "fpr"))

        writer.writerow(evaluation_list("100steps_stepbystepprojection_generated.csv", "target.csv", 1, "sbs"))
        writer.writerow(evaluation_list("d_2_stepbystepprojection_generated.csv", "target.csv", 2, "sbs"))
        writer.writerow(evaluation_list("d_3_stepbystepprojection_generated.csv", "target.csv", 3, "sbs"))
        writer.writerow(evaluation_list("d_4_stepbystepprojection_generated.csv", "target.csv", 4, "sbs"))
        writer.writerow(evaluation_list("d_5_stepbystepprojection_generated.csv", "target.csv", 5, "sbs" ))
        

evaluation_list_to_csv()

def mean_from_lists(list_position, all_lists):
    mass_error = 0
    negativity_violation_mean = 0
    infeasible_count = 0 
    mode_balance_right = 0
    mode_balance_left = 0
    swd = 0
    covariance_difference = 0
    sample_time = 0

    for n in range(15):
        mass_error += all_lists[n][2]
        negativity_violation_mean += all_lists[n][3]
        infeasible_count += all_lists[n][4]
        mode_balance_right += all_lists[n][5]
        mode_balance_left += all_lists[n][6]
        swd += all_lists[n][7]
        covariance_difference += all_lists[n][8]
    return mass_error

def mean_from_lists(list_position, all_lists, method):
    x = 0
    if method == "unc":
        additional = 0
    elif method == "fpr":
        additional = 5
    else:
        additional = 10
    for n in range(0 + additional, 5 + additional):
        x += all_lists[n][list_position]
    x = x / 5
    return x


while True:
    alt = int(input("Choose an alternative (1: check one seed. 2: check seed mean. 3: exit): "))
    if alt == 1:
        seed_alt = int(input("Choose which seed you want to check (1-5): "))
        if seed_alt == 1:
            unconstrained_filename = f"100steps_unconstrained_generated.csv"
            finalprojection_filename = f"100steps_finalprojection_generated.csv"
            stepbystepprojection_filename = f"100steps_stepbystepprojection_generated.csv"
            target_filename = f"target.csv" 
            unconstrained_list = csv_to_list(unconstrained_filename)
            finalprojection_list = csv_to_list(finalprojection_filename)
            stepbystepprojection_list = csv_to_list(stepbystepprojection_filename)
        elif seed_alt == 2:
            unconstrained_filename = f"d_2_unconstrained_generated.csv"
            finalprojection_filename = f"d_2_finalprojection_generated.csv"
            stepbystepprojection_filename = f"d_2_stepbystepprojection_generated.csv"
            target_filename = f"target.csv" 
            unconstrained_list = csv_to_list(unconstrained_filename)
            finalprojection_list = csv_to_list(finalprojection_filename)
            stepbystepprojection_list = csv_to_list(stepbystepprojection_filename)
        elif seed_alt == 3:
            unconstrained_filename = f"d_3_unconstrained_generated.csv"
            finalprojection_filename = f"d_3_finalprojection_generated.csv"
            stepbystepprojection_filename = f"d_3_stepbystepprojection_generated.csv"
            target_filename = f"target.csv" 
            unconstrained_list = csv_to_list(unconstrained_filename)
            finalprojection_list = csv_to_list(finalprojection_filename)
            stepbystepprojection_list = csv_to_list(stepbystepprojection_filename)
        elif seed_alt == 4:
            unconstrained_filename = f"d_4_unconstrained_generated.csv"
            finalprojection_filename = f"d_4_finalprojection_generated.csv"
            stepbystepprojection_filename = f"d_4_stepbystepprojection_generated.csv"
            target_filename = f"target.csv" 
            unconstrained_list = csv_to_list(unconstrained_filename)
            finalprojection_list = csv_to_list(finalprojection_filename)
            stepbystepprojection_list = csv_to_list(stepbystepprojection_filename)
        elif seed_alt == 5:
            unconstrained_filename = f"d_5_unconstrained_generated.csv"
            finalprojection_filename = f"d_5_finalprojection_generated.csv"
            stepbystepprojection_filename = f"d_5_stepbystepprojection_generated.csv"
            target_filename = f"target.csv" 
            unconstrained_list = csv_to_list(unconstrained_filename)
            finalprojection_list = csv_to_list(finalprojection_filename)
            stepbystepprojection_list = csv_to_list(stepbystepprojection_filename)
        else:
            print("Not a possible alternative")
            continue

        print_title("Evaluation of unconstraint generated points:")
        print("Mass error mean: ", mass_error_mean(unconstrained_list))
        print("Negativity violation mean: ", negativity_violation_mean(unconstrained_list))
        print("Feasibility rate: ", feasibility_rate2(unconstrained_list), "out of 10 000 are infeasible.")
        print("The mode balance is: ", mode_balance(N_DIM, unconstrained_filename))
        print("Swd_value: ", swd_value(unconstrained_filename, target_filename))
        print("The covarience matrix difference: ", covariance(unconstrained_filename, target_filename))

        print_title("Evaluation of final projection generated points:")
        print("Mass error mean: ", mass_error_mean(finalprojection_list))
        print("Negativity violation mean: ", negativity_violation_mean(finalprojection_list))
        print("Feasibility rate: ", feasibility_rate2(finalprojection_list), "out of 10 000 are infeasible.")
        print("The mode balance is: ", mode_balance(N_DIM, finalprojection_filename))
        print("Swd_value: ", swd_value(finalprojection_filename, target_filename))
        print("The covarience matrix difference: ", covariance(finalprojection_filename, target_filename))


        print_title("Evaluation of step by step generated points:")
        print("Mass error mean: ", mass_error_mean(stepbystepprojection_list))
        print("Negativity violation mean: ", negativity_violation_mean(stepbystepprojection_list))
        print("Feasibility rate: ", feasibility_rate2(stepbystepprojection_list), "out of 10 000 are infeasible.")
        print("The mode balance is: ", mode_balance(N_DIM, stepbystepprojection_filename))
        print("Swd_value: ", swd_value(stepbystepprojection_filename, target_filename))
        print("The covarience matrix difference: ", covariance(stepbystepprojection_filename, target_filename))

        PCA_plot(unconstrained_filename, finalprojection_filename, stepbystepprojection_filename, target_filename)


    elif alt == 2:
        print_title("Seed mean for unconstrained:")
        print("Mass error mean: ", mean_from_lists(2, all_lists, "unc"))
        print("Negativity violation mean: ", mean_from_lists(3, all_lists, "unc"))
        print("Feasibility rate: ",  mean_from_lists(4, all_lists, "unc"), "out of 10 000 are infeasible.")
        print("The mode balance mean on the right: ",  mean_from_lists(5, all_lists, "unc"))
        print("The mode balance mean on the right: ",  mean_from_lists(6, all_lists, "unc"))
        print("Swd_value mean: ", mean_from_lists(7, all_lists, "unc"))
        print("The covarience matrix difference mean: ", mean_from_lists(8, all_lists, "unc"))

        print_title("Seed mean for final projection:")
        print("Mass error mean: ", mean_from_lists(2, all_lists, "fpr"))
        print("Negativity violation mean: ", mean_from_lists(3, all_lists, "fpr"))
        print("Feasibility rate: ",  mean_from_lists(4, all_lists, "fpr"), "out of 10 000 are infeasible.")
        print("The mode balance mean on the right: ",  mean_from_lists(5, all_lists, "fpr"))
        print("The mode balance mean on the right: ",  mean_from_lists(6, all_lists, "fpr"))
        print("Swd_value mean: ", mean_from_lists(7, all_lists, "fpr"))
        print("The covarience matrix difference mean: ", mean_from_lists(8, all_lists, "fpr"))

        print_title("Seed mean for step-by-step projection:")
        print("Mass error mean: ", mean_from_lists(2, all_lists, "sbs"))
        print("Negativity violation mean: ", mean_from_lists(3, all_lists, "sbs"))
        print("Feasibility rate: ",  mean_from_lists(4, all_lists, "sbs"), "out of 10 000 are infeasible.")
        print("The mode balance mean on the right: ",  mean_from_lists(5, all_lists, "sbs"))
        print("The mode balance mean on the right: ",  mean_from_lists(6, all_lists, "sbs"))
        print("Swd_value mean: ", mean_from_lists(7, all_lists, "sbs"))
        print("The covarience matrix difference mean: ", mean_from_lists(8, all_lists, "sbs"))

    elif alt == 3:
        print("Program closed.")
        break

    else:
        print("Not a possible alternative.")



        


# "mass_error_mean": None, "negativity_violation_mean": None, "feasibility_rate2": None, "mode_balance": None, "swd_value": None, "covariance": None,

#import numpy as np
#mean_swd = np.mean(unconstrained_swd)
#std_swd = np.std(unconstrained_swd, ddof=1)
#print(f"Unconstrained SWD: {mean_swd:.3f} ± {std_swd:.3f}")
