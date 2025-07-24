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


#%%   Asymmetry impact study - Delta E replicas   FigureS12-A-B-C-D

############################################### Uncomment this part to perform calculations for system csNOX5 mbW (FigureS12A)
filenames_energy1 = [
    '../csNOX5_mbW/data/nrj_st1_e1_1_40.dat',
    '../csNOX5_mbW/data/nrj_st2_e1_1_40.dat',
    '../csNOX5_mbW/data/replicas/300ns/nrj_st1_e1_21_26.dat',
    '../csNOX5_mbW/data/replicas/300ns/nrj_st2_e1_21_26.dat',
    '../csNOX5_mbW/data/replicas/405ns/nrj_st1_e1_28_33.dat',
    '../csNOX5_mbW/data/replicas/405ns/nrj_st2_e1_28_33.dat',
    '../csNOX5_mbW/data/replicas/510ns/nrj_st1_e1_35_40.dat',
    '../csNOX5_mbW/data/replicas/510ns/nrj_st2_e1_35_40.dat',
    '../csNOX5_mbW/data/replicas/600ns/nrj_st1_e1_41_46.dat',
    '../csNOX5_mbW/data/replicas/600ns/nrj_st2_e1_41_46.dat',
]

filenames_energy2 = [
    '../csNOX5_mbW/data/nrj_st1_e2_1_40.dat',
    '../csNOX5_mbW/data/nrj_st2_e2_1_40.dat',
    '../csNOX5_mbW/data/replicas/300ns/nrj_st1_e2_21_26.dat',
    '../csNOX5_mbW/data/replicas/300ns/nrj_st2_e2_21_26.dat',
    '../csNOX5_mbW/data/replicas/405ns/nrj_st1_e2_28_33.dat',
    '../csNOX5_mbW/data/replicas/405ns/nrj_st2_e2_28_33.dat',
    '../csNOX5_mbW/data/replicas/510ns/nrj_st1_e2_35_40.dat',
    '../csNOX5_mbW/data/replicas/510ns/nrj_st2_e2_35_40.dat',
    '../csNOX5_mbW/data/replicas/600ns/nrj_st1_e2_41_46.dat',
    '../csNOX5_mbW/data/replicas/600ns/nrj_st2_e2_41_46.dat',
]

# Proximity 2D
filenames_lipids = [
    '../csNOX5_mbW/data/lipid_distances2D_csNOX5_mbW_st1.pkl',
    '../csNOX5_mbW/data/lipid_distances2D_csNOX5_mbW_st2.pkl',
    '../csNOX5_mbW/data/replicas/300ns/lipid_distances2D_csNOX5_mbW_st1_A.pkl',
    '../csNOX5_mbW/data/replicas/300ns/lipid_distances2D_csNOX5_mbW_st2_A.pkl',
    '../csNOX5_mbW/data/replicas/405ns/lipid_distances2D_csNOX5_mbW_st1_B.pkl',
    '../csNOX5_mbW/data/replicas/405ns/lipid_distances2D_csNOX5_mbW_st2_B.pkl',
    '../csNOX5_mbW/data/replicas/510ns/lipid_distances2D_csNOX5_mbW_st1_C.pkl',
    '../csNOX5_mbW/data/replicas/510ns/lipid_distances2D_csNOX5_mbW_st2_C.pkl',
    '../csNOX5_mbW/data/replicas/600ns/lipid_distances2D_csNOX5_mbW_st1_D.pkl',
    '../csNOX5_mbW/data/replicas/600ns/lipid_distances2D_csNOX5_mbW_st2_D.pkl',
]
colors = ['red', 'red', 'orangered', 'orangered', 'firebrick', 'firebrick', 'crimson', 'crimson', 'lightcoral', 'lightcoral'] # csNOX5 mbW
fname = 'FigureS14A.tiff'



############################################### Uncomment this part to perform calculations for system csNOX5 mbH (FigureS12B)
#filenames_energy1 = [
#    '../csNOX5_mbH/data/nrj_st1_e1_1_40.dat',
#    '../csNOX5_mbH/data/nrj_st2_e1_1_40.dat',
#    '../csNOX5_mbH/data/replicas/300ns/nrj_st1_e1_21_26.dat',
#    '../csNOX5_mbH/data/replicas/300ns/nrj_st2_e1_21_26.dat',
#    '../csNOX5_mbH/data/replicas/405ns/nrj_st1_e1_28_33.dat',
#    '../csNOX5_mbH/data/replicas/405ns/nrj_st2_e1_28_33.dat',
#    '../csNOX5_mbH/data/replicas/510ns/nrj_st1_e1_35_40.dat',
#    '../csNOX5_mbH/data/replicas/510ns/nrj_st2_e1_35_40.dat',
#    '../csNOX5_mbH/data/replicas/600ns/nrj_st1_e1_41_46.dat',
#    '../csNOX5_mbH/data/replicas/600ns/nrj_st2_e1_41_46.dat',
#]
#
#filenames_energy2 = [
#    '../csNOX5_mbH/data/nrj_st1_e2_1_40.dat',
#    '../csNOX5_mbH/data/nrj_st2_e2_1_40.dat',
#    '../csNOX5_mbH/data/replicas/300ns/nrj_st1_e2_21_26.dat',
#    '../csNOX5_mbH/data/replicas/300ns/nrj_st2_e2_21_26.dat',
#    '../csNOX5_mbH/data/replicas/405ns/nrj_st1_e2_28_33.dat',
#    '../csNOX5_mbH/data/replicas/405ns/nrj_st2_e2_28_33.dat',
#    '../csNOX5_mbH/data/replicas/510ns/nrj_st1_e2_35_40.dat',
#    '../csNOX5_mbH/data/replicas/510ns/nrj_st2_e2_35_40.dat',
#    '../csNOX5_mbH/data/replicas/600ns/nrj_st1_e2_41_46.dat',
#    '../csNOX5_mbH/data/replicas/600ns/nrj_st2_e2_41_46.dat',
#]
#
## Proximity 2D
#filenames_lipids = [
#    '../csNOX5_mbH/data/lipid_distances2D_csNOX5_mbH_st1.pkl',
#    '../csNOX5_mbH/data/lipid_distances2D_csNOX5_mbH_st2.pkl',
#    '../csNOX5_mbH/data/replicas/300ns/lipid_distances2D_csNOX5_mbH_st1_A.pkl',
#    '../csNOX5_mbH/data/replicas/300ns/lipid_distances2D_csNOX5_mbH_st2_A.pkl',
#    '../csNOX5_mbH/data/replicas/405ns/lipid_distances2D_csNOX5_mbH_st1_B.pkl',
#    '../csNOX5_mbH/data/replicas/405ns/lipid_distances2D_csNOX5_mbH_st2_B.pkl',
#    '../csNOX5_mbH/data/replicas/510ns/lipid_distances2D_csNOX5_mbH_st1_C.pkl',
#    '../csNOX5_mbH/data/replicas/510ns/lipid_distances2D_csNOX5_mbH_st2_C.pkl',
#    '../csNOX5_mbH/data/replicas/600ns/lipid_distances2D_csNOX5_mbH_st1_D.pkl',
#    '../csNOX5_mbH/data/replicas/600ns/lipid_distances2D_csNOX5_mbH_st2_D.pkl',
#]
#colors = ['orange', 'orange', 'peru', 'peru', 'coral', 'coral', 'orangered', 'orangered', 'lightsalmon', 'lightsalmon'] # csNOX5 mbH
#fname = 'FigureS14B.tiff'



############################################### Uncomment this part to perform calculations for system hNOX5 mbW (FigureS12C)
#filenames_energy1 = [
#    '../hNOX5_mbW/data/nrj_st1_e1_1_40.dat',
#    '../hNOX5_mbW/data/nrj_st2_e1_1_40.dat',
#    '../hNOX5_mbW/data/replicas/300ns/nrj_st1_e1_21_26.dat',
#    '../hNOX5_mbW/data/replicas/300ns/nrj_st2_e1_21_26.dat',
#    '../hNOX5_mbW/data/replicas/405ns/nrj_st1_e1_28_33.dat',
#    '../hNOX5_mbW/data/replicas/405ns/nrj_st2_e1_28_33.dat',
#    '../hNOX5_mbW/data/replicas/510ns/nrj_st1_e1_35_40.dat',
#    '../hNOX5_mbW/data/replicas/510ns/nrj_st2_e1_35_40.dat',
#    '../hNOX5_mbW/data/replicas/600ns/nrj_st1_e1_41_46.dat',
#    '../hNOX5_mbW/data/replicas/600ns/nrj_st2_e1_41_46.dat',
#]
#
#filenames_energy2 = [
#    '../hNOX5_mbW/data/nrj_st1_e2_1_40.dat',
#    '../hNOX5_mbW/data/nrj_st2_e2_1_40.dat',
#    '../hNOX5_mbW/data/replicas/300ns/nrj_st1_e2_21_26.dat',
#    '../hNOX5_mbW/data/replicas/300ns/nrj_st2_e2_21_26.dat',
#    '../hNOX5_mbW/data/replicas/405ns/nrj_st1_e2_28_33.dat',
#    '../hNOX5_mbW/data/replicas/405ns/nrj_st2_e2_28_33.dat',
#    '../hNOX5_mbW/data/replicas/510ns/nrj_st1_e2_35_40.dat',
#    '../hNOX5_mbW/data/replicas/510ns/nrj_st2_e2_35_40.dat',
#    '../hNOX5_mbW/data/replicas/600ns/nrj_st1_e2_41_46.dat',
#    '../hNOX5_mbW/data/replicas/600ns/nrj_st2_e2_41_46.dat',
#]
#
## Proximity 2D
#filenames_lipids = [
#    '../hNOX5_mbW/data/lipid_distances2D_hNOX5_mbW_st1.pkl',
#    '../hNOX5_mbW/data/lipid_distances2D_hNOX5_mbW_st2.pkl',
#    '../hNOX5_mbW/data/replicas/300ns/lipid_distances2D_hNOX5_mbW_st1_A.pkl',
#    '../hNOX5_mbW/data/replicas/300ns/lipid_distances2D_hNOX5_mbW_st2_A.pkl',
#    '../hNOX5_mbW/data/replicas/405ns/lipid_distances2D_hNOX5_mbW_st1_B.pkl',
#    '../hNOX5_mbW/data/replicas/405ns/lipid_distances2D_hNOX5_mbW_st2_B.pkl',
#    '../hNOX5_mbW/data/replicas/510ns/lipid_distances2D_hNOX5_mbW_st1_C.pkl',
#    '../hNOX5_mbW/data/replicas/510ns/lipid_distances2D_hNOX5_mbW_st2_C.pkl',
#    '../hNOX5_mbW/data/replicas/600ns/lipid_distances2D_hNOX5_mbW_st1_D.pkl',
#    '../hNOX5_mbW/data/replicas/600ns/lipid_distances2D_hNOX5_mbW_st2_D.pkl',
#]
#colors = ['blue', 'blue', 'royalblue', 'royalblue', 'navy', 'navy', 'skyblue', 'skyblue', 'lightblue', 'lightblue'] # hNOX5 mbW
#fname = 'FigureS14C.tiff'



############################################### Uncomment this part to perform calculations for system hNOX5 mbH (FigureS12D)
#filenames_energy1 = [
#    '../hNOX5_mbH/data/nrj_st1_e1_1_40.dat',
#    '../hNOX5_mbH/data/nrj_st2_e1_1_40.dat',
#    '../hNOX5_mbH/data/replicas/300ns/nrj_st1_e1_21_26.dat',
#    '../hNOX5_mbH/data/replicas/300ns/nrj_st2_e1_21_26.dat',
#    '../hNOX5_mbH/data/replicas/405ns/nrj_st1_e1_28_33.dat',
#    '../hNOX5_mbH/data/replicas/405ns/nrj_st2_e1_28_33.dat',
#    '../hNOX5_mbH/data/replicas/510ns/nrj_st1_e1_35_40.dat',
#    '../hNOX5_mbH/data/replicas/510ns/nrj_st2_e1_35_40.dat',
#    '../hNOX5_mbH/data/replicas/600ns/nrj_st1_e1_41_46.dat',
#    '../hNOX5_mbH/data/replicas/600ns/nrj_st2_e1_41_46.dat',
#]
#
#filenames_energy2 = [
#    '../hNOX5_mbH/data/nrj_st1_e2_1_40.dat',
#    '../hNOX5_mbH/data/nrj_st2_e2_1_40.dat',
#    '../hNOX5_mbH/data/replicas/300ns/nrj_st1_e2_21_26.dat',
#    '../hNOX5_mbH/data/replicas/300ns/nrj_st2_e2_21_26.dat',
#    '../hNOX5_mbH/data/replicas/405ns/nrj_st1_e2_28_33.dat',
#    '../hNOX5_mbH/data/replicas/405ns/nrj_st2_e2_28_33.dat',
#    '../hNOX5_mbH/data/replicas/510ns/nrj_st1_e2_35_40.dat',
#    '../hNOX5_mbH/data/replicas/510ns/nrj_st2_e2_35_40.dat',
#    '../hNOX5_mbH/data/replicas/600ns/nrj_st1_e2_41_46.dat',
#    '../hNOX5_mbH/data/replicas/600ns/nrj_st2_e2_41_46.dat',
#]
#
## Proximity 2D
#filenames_lipids = [
#    '../hNOX5_mbH/data/lipid_distances2D_hNOX5_mbH_st1.pkl',
#    '../hNOX5_mbH/data/lipid_distances2D_hNOX5_mbH_st2.pkl',
#    '../hNOX5_mbH/data/replicas/300ns/lipid_distances2D_hNOX5_mbH_st1_A.pkl',
#    '../hNOX5_mbH/data/replicas/300ns/lipid_distances2D_hNOX5_mbH_st2_A.pkl',
#    '../hNOX5_mbH/data/replicas/405ns/lipid_distances2D_hNOX5_mbH_st1_B.pkl',
#    '../hNOX5_mbH/data/replicas/405ns/lipid_distances2D_hNOX5_mbH_st2_B.pkl',
#    '../hNOX5_mbH/data/replicas/510ns/lipid_distances2D_hNOX5_mbH_st1_C.pkl',
#    '../hNOX5_mbH/data/replicas/510ns/lipid_distances2D_hNOX5_mbH_st2_C.pkl',
#    '../hNOX5_mbH/data/replicas/600ns/lipid_distances2D_hNOX5_mbH_st1_D.pkl',
#    '../hNOX5_mbH/data/replicas/600ns/lipid_distances2D_hNOX5_mbH_st2_D.pkl',
#]
#colors = ['purple', 'purple', 'indigo', 'indigo', 'blueviolet', 'blueviolet', 'mediumorchid', 'mediumorchid', 'lavender', 'lavender'] # hNOX5 mbH
#fname = 'FigureS14D.tiff'


markers = ['s', 's', '*', '*', 'o', 'o', 'P', 'P', 'v', 'v']  
line_styles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
edgecolors = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black']
alphas = [0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# Figure Initialization
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
for idx, (filename1, filename2, filename_lipids) in enumerate(zip(filenames_energy1, filenames_energy2, filenames_lipids)):
    
    # Read energy data on redox state 1
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

    # Read energy data on redox state 2
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
    # Extraction of the contribution to delta G
    start_index = 3000 if idx < 2 else 200  # 3000 for the 600ns-long MD trajectory, 200 for the replicas

    with open(filename_lipids, 'rb') as fichier:
        data = pickle.load(fichier)
    # 2D-distances between hemes and lipids
    POPE_upper_dist2D = data['POPE_upper']
    POPG_upper_dist2D = data['POPG_upper']
    POPE_lower_dist2D = data['POPE_lower']
    POPG_lower_dist2D = data['POPG_lower']
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

    total_deltaE_filtered = np.array(total_deltaE[start_index:]) 
    POPGup_energy_diff_filtered = np.array(POPGup_energy_diff[start_index:])
    POPGup_energy1_filtered = np.array(POPGup_energy1[start_index:])
    POPGup_energy2_filtered = np.array(POPGup_energy2[start_index:])
    POPGlo_energy_diff_filtered = np.array(POPGlo_energy_diff[start_index:])
    POPG_ratio2D_filtered = POPG_ratio2D[start_index:]

    temps = [x for x in range(len(total_deltaE_filtered))]
    ax.scatter(POPG_ratio2D_filtered, total_deltaE_filtered, label=f'Trajectoire {idx+1}', color=colors[idx], edgecolor=edgecolors[idx], marker=markers[idx], linestyle=line_styles[idx], alpha=alphas[idx])

ax.set_xlabel(r'$I_{as}$', fontsize=20)
ax.set_ylabel(r'$ΔE_{POPG}$(eV)', fontsize=20)
ax.tick_params(axis='x', labelsize=15) 
ax.tick_params(axis='y', labelsize=15) 
ax.grid(True)

plt.tight_layout()
plt.tight_layout()
ax = plt.gca() 
for spine in ax.spines.values():
    spine.set_linewidth(1.5) 
ax.tick_params(width=1.5, length=8) 
plt.savefig(fname, format="tiff", dpi=300)
