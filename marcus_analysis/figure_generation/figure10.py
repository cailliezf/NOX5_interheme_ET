#!/usr/bin/env python3


#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd
import sklearn.metrics as sklearn
from sklearn.linear_model import LinearRegression
from adjustText import adjust_text
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


#%%

################## CORE ####################


#%%   Asymmetry impact study - Delta E all trajs (Figure10A)

# Files for each trajectory
filenames_energy1 = [
    '../csNOX5_mbW/data/nrj_st1_e1_1_40.dat',
    '../csNOX5_mbW/data/nrj_st2_e1_1_40.dat',
    '../csNOX5_mbH/data/nrj_st1_e1_1_40.dat',
    '../csNOX5_mbH/data/nrj_st2_e1_1_40.dat',
    '../hNOX5_mbW/data/nrj_st1_e1_1_40.dat',
    '../hNOX5_mbW/data/nrj_st2_e1_1_40.dat',
    '../hNOX5_mbH/data/nrj_st1_e1_1_40.dat',
    '../hNOX5_mbH/data/nrj_st2_e1_1_40.dat',
]

filenames_energy2 = [
    '../csNOX5_mbW/data/nrj_st1_e2_1_40.dat',
    '../csNOX5_mbW/data/nrj_st2_e2_1_40.dat',
    '../csNOX5_mbH/data/nrj_st1_e2_1_40.dat',
    '../csNOX5_mbH/data/nrj_st2_e2_1_40.dat',
    '../hNOX5_mbW/data/nrj_st1_e2_1_40.dat',
    '../hNOX5_mbW/data/nrj_st2_e2_1_40.dat',
    '../hNOX5_mbH/data/nrj_st1_e2_1_40.dat',
    '../hNOX5_mbH/data/nrj_st2_e2_1_40.dat',
]

filenames_lipids = [
    '../csNOX5_mbW/data/lipid_distances2D_csNOX5_mbW_st1.pkl',
    '../csNOX5_mbW/data/lipid_distances2D_csNOX5_mbW_st2.pkl',
    '../csNOX5_mbH/data/lipid_distances2D_csNOX5_mbH_st1.pkl',
    '../csNOX5_mbH/data/lipid_distances2D_csNOX5_mbH_st2.pkl',
    '../hNOX5_mbW/data/lipid_distances2D_hNOX5_mbW_st1.pkl',
    '../hNOX5_mbW/data/lipid_distances2D_hNOX5_mbW_st2.pkl',
    '../hNOX5_mbH/data/lipid_distances2D_hNOX5_mbH_st1.pkl',
    '../hNOX5_mbH/data/lipid_distances2D_hNOX5_mbH_st2.pkl',
]

colors = ['r', 'r', 'orange', 'orange', 'b', 'b', 'purple', 'purple']
edgecolors1 = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black',]
markers1 = ['s', 's', 's', 's', 's', 's', 's', 's']  
line_styles = ['-', '-', '-', '-', '-', '-', '-', '-']
reglin_ls = ['-',':','-',':','-',':','-',':']
texts = []

# Figure initialization
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)


for idx, (filename1, filename2, filename_lipids) in enumerate(zip(filenames_energy1, filenames_energy2, filenames_lipids)):
    
    # Read energies for redox state 1
    time_of_simulation = []
    POPGup_energy1 = []
    POPGlo_energy1 = []
    POPGup_energy2 = []
    POPGlo_energy2 = []

    with open(filename1, 'r') as file:
        for time_of_simulations, line in enumerate(file, start=1):
            line = line.strip()
            numbers = line.split()
            if len(numbers) >= 10:
                POPGup_energies = float(numbers[7]) - float(numbers[3]) - float(numbers[4]) - (float(numbers[2]) - float(numbers[3])  - float(numbers[4])) 
                POPGlo_energies = float(numbers[9]) - float(numbers[3]) - float(numbers[4]) - (float(numbers[2]) - float(numbers[3])  - float(numbers[4]))                 
                time_of_simulation.append(time_of_simulations)
                POPGup_energy1.append(POPGup_energies)
                POPGlo_energy1.append(POPGlo_energies)

    # Read energies for redox state 2
    with open(filename2, 'r') as file:
        for time_of_simulations, line in enumerate(file, start=1):
            line = line.strip()
            numbers = line.split()
            if len(numbers) >= 10:
                POPGup_energies = float(numbers[7]) - float(numbers[3]) - float(numbers[4]) - (float(numbers[2]) - float(numbers[3])  - float(numbers[4])) 
                POPGlo_energies = float(numbers[9]) - float(numbers[3]) - float(numbers[4]) - (float(numbers[2]) - float(numbers[3])  - float(numbers[4]))  
                POPGup_energy2.append(POPGup_energies)
                POPGlo_energy2.append(POPGlo_energies)

    # Computation of energy gaps (DeltaE)
    POPGup_energy_diff = [b - a for a, b in zip(POPGup_energy1, POPGup_energy2)]
    POPGlo_energy_diff = [b - a for a, b in zip(POPGlo_energy1, POPGlo_energy2)]
    total_deltaE = [up + lo for up, lo in zip(POPGup_energy_diff, POPGlo_energy_diff)]
    
    

    # Read pkl file containing 2D distances between lipids and hemes
    with open(filename_lipids, 'rb') as f:
        data_loaded = pickle.load(f)

    POPE_upper_dist2D = data_loaded['POPE_upper']
    POPG_upper_dist2D = data_loaded['POPG_upper']
    POPE_lower_dist2D = data_loaded['POPE_lower']
    POPG_lower_dist2D = data_loaded['POPG_lower']

    POPE_upper_coeff = 1/np.array(POPE_upper_dist2D)
    POPG_upper_coeff = 1/np.array(POPG_upper_dist2D)
    POPE_lower_coeff = 1/np.array(POPE_lower_dist2D)
    POPG_lower_coeff = 1/np.array(POPG_lower_dist2D)
    POPE_upper_moy = np.mean(POPE_upper_coeff, axis=1)
    POPG_upper_moy = np.mean(POPG_upper_coeff, axis=1)
    POPE_lower_moy = np.mean(POPE_lower_coeff, axis=1)
    POPG_lower_moy = np.mean(POPG_lower_coeff, axis=1)
    POPE_ratio2D = POPE_upper_moy / POPE_lower_moy
    POPG_ratio2D = POPG_upper_moy / POPG_lower_moy    

    # Select only the second half of the simulations
    total_deltaE_filtered = np.array(total_deltaE[3000:])/ (np.size(POPG_upper_coeff,axis=1)+np.size(POPG_lower_coeff,axis=1))
    POPGup_energy_diff_filtered = np.array(POPGup_energy_diff[3000:])
    POPGup_energy1_filtered = np.array(POPGup_energy1[3000:])
    POPGup_energy2_filtered = np.array(POPGup_energy2[3000:])
    POPGlo_energy_diff_filtered = np.array(POPGlo_energy_diff[3000:])
    POPG_ratio2D_filtered = POPG_ratio2D[3000:]

    temps = [x for x in range(len(total_deltaE_filtered))]
    

    ax.scatter(POPG_ratio2D_filtered, total_deltaE_filtered, label=f'Trajectoire {idx+1}', edgecolor=edgecolors1[idx], color=colors[idx], marker=markers1[idx], linestyle=line_styles[idx], alpha=0.3)
               
    
ax.set_xlabel(r'$I_{as}$', fontsize=20)
ax.set_ylabel(r'$ΔE_{POPG}$(eV)', fontsize=20)
ax.tick_params(axis='x', labelsize=15) 
ax.tick_params(axis='y', labelsize=15) 
ax.grid(True)

plt.tight_layout()
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(width=1.5, length=8) 
fname = 'Figure10A.tiff'
plt.savefig(fname, format="tiff", dpi=300)

#%%   Asymmetry impact study - Delta G (Figure10B)

data_bis = 'deltaG'   # deltaG, lambda, lambdavar

# ALL Systemes
filenames_lipids_st1 = [
    '../csNOX5_mbW/data/lipid_distances2D_csNOX5_mbW_st1.pkl',
    '../csNOX5_mbH/data/lipid_distances2D_csNOX5_mbH_st1.pkl',
    '../hNOX5_mbW/data/lipid_distances2D_hNOX5_mbW_st1.pkl',
    '../hNOX5_mbH/data/lipid_distances2D_hNOX5_mbH_st1.pkl',
    '../csNOX5_mbW/data/replicas/300ns/lipid_distances2D_csNOX5_mbW_st1_A.pkl',
    '../csNOX5_mbW/data/replicas/405ns/lipid_distances2D_csNOX5_mbW_st1_B.pkl',
    '../csNOX5_mbW/data/replicas/510ns/lipid_distances2D_csNOX5_mbW_st1_C.pkl',
    '../csNOX5_mbW/data/replicas/600ns/lipid_distances2D_csNOX5_mbW_st1_D.pkl',
    '../csNOX5_mbH/data/replicas/300ns/lipid_distances2D_csNOX5_mbH_st1_A.pkl',
    '../csNOX5_mbH/data/replicas/405ns/lipid_distances2D_csNOX5_mbH_st1_B.pkl',
    '../csNOX5_mbH/data/replicas/510ns/lipid_distances2D_csNOX5_mbH_st1_C.pkl',
    '../csNOX5_mbH/data/replicas/600ns/lipid_distances2D_csNOX5_mbH_st1_D.pkl',
    '../hNOX5_mbW/data/replicas/300ns/lipid_distances2D_hNOX5_mbW_st1_A.pkl',
    '../hNOX5_mbW/data/replicas/405ns/lipid_distances2D_hNOX5_mbW_st1_B.pkl',
    '../hNOX5_mbW/data/replicas/510ns/lipid_distances2D_hNOX5_mbW_st1_C.pkl',
    '../hNOX5_mbW/data/replicas/600ns/lipid_distances2D_hNOX5_mbW_st1_D.pkl',
    '../hNOX5_mbH/data/replicas/300ns/lipid_distances2D_hNOX5_mbH_st1_A.pkl',
    '../hNOX5_mbH/data/replicas/405ns/lipid_distances2D_hNOX5_mbH_st1_B.pkl',
    '../hNOX5_mbH/data/replicas/510ns/lipid_distances2D_hNOX5_mbH_st1_C.pkl',
    '../hNOX5_mbH/data/replicas/600ns/lipid_distances2D_hNOX5_mbH_st1_D.pkl',
]

filenames_lipids_st2 = [
    '../csNOX5_mbW/data/lipid_distances2D_csNOX5_mbW_st2.pkl',
    '../csNOX5_mbH/data/lipid_distances2D_csNOX5_mbH_st2.pkl',
    '../hNOX5_mbW/data/lipid_distances2D_hNOX5_mbW_st2.pkl',
    '../hNOX5_mbH/data/lipid_distances2D_hNOX5_mbH_st2.pkl',
    '../csNOX5_mbW/data/replicas/300ns/lipid_distances2D_csNOX5_mbW_st2_A.pkl',
    '../csNOX5_mbW/data/replicas/405ns/lipid_distances2D_csNOX5_mbW_st2_B.pkl',
    '../csNOX5_mbW/data/replicas/510ns/lipid_distances2D_csNOX5_mbW_st2_C.pkl',
    '../csNOX5_mbW/data/replicas/600ns/lipid_distances2D_csNOX5_mbW_st2_D.pkl',
    '../csNOX5_mbH/data/replicas/300ns/lipid_distances2D_csNOX5_mbH_st2_A.pkl',
    '../csNOX5_mbH/data/replicas/405ns/lipid_distances2D_csNOX5_mbH_st2_B.pkl',
    '../csNOX5_mbH/data/replicas/510ns/lipid_distances2D_csNOX5_mbH_st2_C.pkl',
    '../csNOX5_mbH/data/replicas/600ns/lipid_distances2D_csNOX5_mbH_st2_D.pkl',
    '../hNOX5_mbW/data/replicas/300ns/lipid_distances2D_hNOX5_mbW_st2_A.pkl',
    '../hNOX5_mbW/data/replicas/405ns/lipid_distances2D_hNOX5_mbW_st2_B.pkl',
    '../hNOX5_mbW/data/replicas/510ns/lipid_distances2D_hNOX5_mbW_st2_C.pkl',
    '../hNOX5_mbW/data/replicas/600ns/lipid_distances2D_hNOX5_mbW_st2_D.pkl',
    '../hNOX5_mbH/data/replicas/300ns/lipid_distances2D_hNOX5_mbH_st2_A.pkl',
    '../hNOX5_mbH/data/replicas/405ns/lipid_distances2D_hNOX5_mbH_st2_B.pkl',
    '../hNOX5_mbH/data/replicas/510ns/lipid_distances2D_hNOX5_mbH_st2_C.pkl',
    '../hNOX5_mbH/data/replicas/600ns/lipid_distances2D_hNOX5_mbH_st2_D.pkl',
]

filenames_deltaG = [
    '../csNOX5_mbW/data/deltaG_csNOX5_mbW_3000-6000.pkl',
    '../csNOX5_mbH/data/deltaG_csNOX5_mbH_3000-6000.pkl',
    '../hNOX5_mbW/data/deltaG_hNOX5_mbW_3000-6000.pkl',
    '../hNOX5_mbH/data/deltaG_hNOX5_mbH_3000-6000.pkl',
    '../csNOX5_mbW/data/replicas/300ns/deltaG_csNOX5_mbW_A_200-900.pkl',
    '../csNOX5_mbW/data/replicas/405ns/deltaG_csNOX5_mbW_B_200-900.pkl',
    '../csNOX5_mbW/data/replicas/510ns/deltaG_csNOX5_mbW_C_200-900.pkl',
    '../csNOX5_mbW/data/replicas/600ns/deltaG_csNOX5_mbW_D_200-900.pkl',
    '../csNOX5_mbH/data/replicas/300ns/deltaG_csNOX5_mbH_A_200-900.pkl',
    '../csNOX5_mbH/data/replicas/405ns/deltaG_csNOX5_mbH_B_200-900.pkl',
    '../csNOX5_mbH/data/replicas/510ns/deltaG_csNOX5_mbH_C_200-900.pkl',
    '../csNOX5_mbH/data/replicas/600ns/deltaG_csNOX5_mbH_D_200-900.pkl',
    '../hNOX5_mbW/data/replicas/300ns/deltaG_hNOX5_mbW_A_200-900.pkl',
    '../hNOX5_mbW/data/replicas/405ns/deltaG_hNOX5_mbW_B_200-900.pkl',
    '../hNOX5_mbW/data/replicas/510ns/deltaG_hNOX5_mbW_C_200-900.pkl',
    '../hNOX5_mbW/data/replicas/600ns/deltaG_hNOX5_mbW_D_200-900.pkl',
    '../hNOX5_mbH/data/replicas/300ns/deltaG_hNOX5_mbH_A_200-900.pkl',
    '../hNOX5_mbH/data/replicas/405ns/deltaG_hNOX5_mbH_B_200-900.pkl',
    '../hNOX5_mbH/data/replicas/510ns/deltaG_hNOX5_mbH_C_200-900.pkl',
    '../hNOX5_mbH/data/replicas/600ns/deltaG_hNOX5_mbH_D_200-900.pkl',
]

def mean_by_block(data, num_blocks=5):
    """
    Divide les data in blocs, calculate the average and uncertainty of the mean
    Parameters:
        data (array-like): data to be treated
        num_blocks (int): Nunmber of blocks
    Returns:
        tuple: Average and uncertainty of the mean
    """
    data = np.array(data)  # Convert to numpy array
    n = len(data)
    block_size = n // num_blocks  # Size of each block
    if n % num_blocks != 0:
        print(f"Attention : les données ne se divisent pas exactement en {num_blocks} blocs.")    
    # Computation of block averages
    block_means = []
    for i in range(num_blocks):
        block_data = data[i * block_size:(i + 1) * block_size]
        block_mean = np.mean(block_data)
        block_means.append(block_mean)
        print(f"Block {i + 1}: Moyenne = {block_mean:.4f}")    
    # Average
    global_mean = np.mean(block_means)    
    # Unvertainty of the mean
    uncertainty = np.std(block_means, ddof=1) / np.sqrt(num_blocks)    
    return global_mean, uncertainty

# Figure Initialization
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

colors = ['red', 'orange', 'blue', 'purple', 'orangered', 'firebrick', 'crimson', 'lightcoral', 'peru', 'coral', 'orangered', 'lightsalmon', 'royalblue', 'navy', 'skyblue', 'lightblue', 'indigo', 'blueviolet', 'mediumorchid', 'lavender']  # cem mbH
markers = ['x', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']  
legends = ['XRD mbW 600ns', 'XRD mbH 600ns', 'CEM mbW 600ns', 'CEM mbH 600ns', 'XRD mbW A', 'XRD mbW B', 'XRD mbW C', 'XRD mbW D', 'XRD mbH A', 'XRD mbH B', 'XRD mbH C', 'XRD mbH D', 'CEM mbW A', 'CEM mbW B', 'CEM mbW C', 'CEM mbW D', 'CEM mbH A', 'CEM mbH B', 'CEM mbH C', 'CEM mbH D']
circle_color = ['red', 'orange', 'blue', 'purple', 'orangered', 'firebrick', 'crimson', 'lightcoral', 'peru', 'coral', 'orangered', 'lightsalmon', 'royalblue', 'navy', 'skyblue', 'lightblue', 'indigo', 'blueviolet', 'mediumorchid', 'lavender'] 

saved_points = []

# Loop over each system
for idx, (filename_deltaG, filename_lipids_st1, filename_lipids_st2) in enumerate(zip(filenames_deltaG, filenames_lipids_st1, filenames_lipids_st2)):
    
    # Read deltaG values
    with open(filename_deltaG, 'rb') as fichier:
        tab1 = pickle.load(fichier)  # deltaG
        tab2 = pickle.load(fichier)  # incertitude 

    # Extraction of the contribution to delta G
    deltaG_value = tab1[13] 
    deltaG_error = tab2[13]
    start_index = 3000 if idx < 4 else 200  # 3000 for the 600ns-long MD trajectory, 200 for the replicas

    # 2D-distances between hemes and lipids in state 1
    with open(filename_lipids_st1, 'rb') as fichier_st1:
        data_st1 = pickle.load(fichier_st1)
    POPE_upper_dist2D_st1 = data_st1['POPE_upper']
    POPG_upper_dist2D_st1 = data_st1['POPG_upper']
    POPE_lower_dist2D_st1 = data_st1['POPE_lower']
    POPG_lower_dist2D_st1 = data_st1['POPG_lower']
    POPE_upper_coeff_st1 = 1/np.array(POPE_upper_dist2D_st1)
    POPG_upper_coeff_st1 = 1/np.array(POPG_upper_dist2D_st1)
    POPE_lower_coeff_st1 = 1/np.array(POPE_lower_dist2D_st1)
    POPG_lower_coeff_st1 = 1/np.array(POPG_lower_dist2D_st1)
    POPE_upper_moy_st1 = np.mean(POPE_upper_coeff_st1, axis=1)
    POPG_upper_moy_st1 = np.mean(POPG_upper_coeff_st1, axis=1)
    POPE_lower_moy_st1 = np.mean(POPE_lower_coeff_st1, axis=1)
    POPG_lower_moy_st1 = np.mean(POPG_lower_coeff_st1, axis=1)
    POPE_ratio2D_st1 = POPE_upper_moy_st1 / POPE_lower_moy_st1
    POPG_ratio2D_st1 = POPG_upper_moy_st1 / POPG_lower_moy_st1    
    
    # 2D-distances between hemes and lipids in state 2
    with open(filename_lipids_st2, 'rb') as fichier_st2:
        data_st2 = pickle.load(fichier_st2)
    POPE_upper_dist2D_st2 = data_st2['POPE_upper']
    POPG_upper_dist2D_st2 = data_st2['POPG_upper']
    POPE_lower_dist2D_st2 = data_st2['POPE_lower']
    POPG_lower_dist2D_st2 = data_st2['POPG_lower']
    POPE_upper_coeff_st2 = 1/np.array(POPE_upper_dist2D_st2)
    POPG_upper_coeff_st2 = 1/np.array(POPG_upper_dist2D_st2)
    POPE_lower_coeff_st2 = 1/np.array(POPE_lower_dist2D_st2)
    POPG_lower_coeff_st2 = 1/np.array(POPG_lower_dist2D_st2)
    POPE_upper_moy_st2 = np.mean(POPE_upper_coeff_st2, axis=1)
    POPG_upper_moy_st2 = np.mean(POPG_upper_coeff_st2, axis=1)
    POPE_lower_moy_st2 = np.mean(POPE_lower_coeff_st2, axis=1)
    POPG_lower_moy_st2 = np.mean(POPG_lower_coeff_st2, axis=1)
    POPE_ratio2D_st2 = POPE_upper_moy_st2 / POPE_lower_moy_st2
    POPG_ratio2D_st2 = POPG_upper_moy_st2 / POPG_lower_moy_st2   

    # Block average for POPG in state 1
    mean_POPG_ratio_st1, dx_POPG_ratio_st1 = mean_by_block(POPG_ratio2D_st1[start_index:], num_blocks=5)
    # Block average for POPG in state 2
    mean_POPG_ratio_st2, dx_POPG_ratio_st2 = mean_by_block(POPG_ratio2D_st2[start_index:], num_blocks=5)
    # Combined averages and std 
    mean_POPG_ratio = (mean_POPG_ratio_st1 + mean_POPG_ratio_st2) / 2
    dx_POPG_ratio = np.sqrt(dx_POPG_ratio_st1**2 + dx_POPG_ratio_st2**2) 

    saved_points.append((mean_POPG_ratio, deltaG_value))    
    ax.errorbar(mean_POPG_ratio, deltaG_value, yerr=deltaG_error, xerr=dx_POPG_ratio, fmt=markers[idx], label=legends[idx], color=colors[idx], markersize=8, capsize=5)    


def plot_regression(points_range, label, color):
    # Read data points
    x_vals, y_vals = zip(*[saved_points[i] for i in points_range])    
    # Conversion to numpy arrays
    x_vals = np.array(x_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)    
    y_vals = np.array([y[0] for y in y_vals], dtype=float)    
    if x_vals.ndim != 1 or y_vals.ndim != 1:
        raise ValueError("Regression data are not in 1D.")    
    # Linear regression
    coeffs = np.polyfit(x_vals, y_vals, 1)
    poly_eq = np.poly1d(coeffs)
    
    correlation_matrix = np.corrcoef(x_vals, y_vals)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy**2
    
    print(f"Slope {label}: {coeffs[0]}")   
    print(f"Intercept {label}: {coeffs[1]}")
    print(f"R² {label}: {r_squared:.4f}")
    # Plot line
    x_range = np.linspace(min(x_vals), max(x_vals), 100)
    ax.plot(x_range, poly_eq(x_range), linestyle='--', label=label, color=color, lw=2)
    
#plot_regression([0, 4, 5, 6, 7], "Regression csnox5mbN", "red") 
#plot_regression([1, 8, 9, 10, 11], "Regression csnox5mbA", "orange")   
#plot_regression([2, 12, 13, 14, 15], "Regression hnox5mbN", "blue")   
#plot_regression([3, 16, 17, 18, 19], "Regression hnox5mbA", "purple") 
plot_regression([1, 3, 8, 9, 10, 11, 16, 17, 18, 19], "Regression nox5mbA", "black") 
      
ax.set_xlabel(r'$\bar{I}_{as}$', fontsize=20)
ax.set_ylabel(r'$ΔG_{memb}$(eV)', fontsize=20)
ax.set_xlim(0.88, 1.05)
ax.set_ylim(-1.5, 1.5)
ax.tick_params(axis='x', labelsize=15) 
ax.tick_params(axis='y', labelsize=15) 

ax.grid(True)

plt.tight_layout()
plt.tight_layout()
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5) 
ax.tick_params(width=1.5, length=8) 
fname = 'Figure10B.tiff'
plt.savefig(fname, format="tiff", dpi=300)


