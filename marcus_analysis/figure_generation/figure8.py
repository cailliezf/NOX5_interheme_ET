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

#%% Delta G data
    
data = 'deltaG' # deltaG, lambda, lambdavar

with open(f'../csNOX5_mbW/data/{data}_csNOX5_mbW_3000-6000.pkl', 'rb') as fichier:
    tab1_csNOX5_mbW_3000_6000 = pickle.load(fichier)
    tab2_csNOX5_mbW_3000_6000 = pickle.load(fichier)   
with open(f'../csNOX5_mbW/data/replicas/300ns/{data}_csNOX5_mbW_A_200-900.pkl', 'rb') as fichier:
    tab1_csNOX5_mbW_A_200_900 = pickle.load(fichier)  
    tab2_csNOX5_mbW_A_200_900 = pickle.load(fichier)      
with open(f'../csNOX5_mbW/data/replicas/405ns/{data}_csNOX5_mbW_B_200-900.pkl', 'rb') as fichier:
    tab1_csNOX5_mbW_B_200_900 = pickle.load(fichier)  
    tab2_csNOX5_mbW_B_200_900 = pickle.load(fichier) 
with open(f'../csNOX5_mbW/data/replicas/510ns/{data}_csNOX5_mbW_C_200-900.pkl', 'rb') as fichier:
    tab1_csNOX5_mbW_C_200_900 = pickle.load(fichier)  
    tab2_csNOX5_mbW_C_200_900 = pickle.load(fichier)     
with open(f'../csNOX5_mbW/data/replicas/600ns/{data}_csNOX5_mbW_D_200-900.pkl', 'rb') as fichier:
    tab1_csNOX5_mbW_D_200_900 = pickle.load(fichier)  
    tab2_csNOX5_mbW_D_200_900 = pickle.load(fichier)         
    
with open(f'../csNOX5_mbH/data/{data}_csNOX5_mbH_3000-6000.pkl', 'rb') as fichier:
    tab1_csNOX5_mbH_3000_6000 = pickle.load(fichier)
    tab2_csNOX5_mbH_3000_6000 = pickle.load(fichier)   
with open(f'../csNOX5_mbH/data/replicas/300ns/{data}_csNOX5_mbH_A_200-900.pkl', 'rb') as fichier:
    tab1_csNOX5_mbH_A_200_900 = pickle.load(fichier)  
    tab2_csNOX5_mbH_A_200_900 = pickle.load(fichier)      
with open(f'../csNOX5_mbH/data/replicas/405ns/{data}_csNOX5_mbH_B_200-900.pkl', 'rb') as fichier:
    tab1_csNOX5_mbH_B_200_900 = pickle.load(fichier)  
    tab2_csNOX5_mbH_B_200_900 = pickle.load(fichier) 
with open(f'../csNOX5_mbH/data/replicas/510ns/{data}_csNOX5_mbH_C_200-900.pkl', 'rb') as fichier:
    tab1_csNOX5_mbH_C_200_900 = pickle.load(fichier)  
    tab2_csNOX5_mbH_C_200_900 = pickle.load(fichier)     
with open(f'../csNOX5_mbH/data/replicas/600ns/{data}_csNOX5_mbH_D_200-900.pkl', 'rb') as fichier:
    tab1_csNOX5_mbH_D_200_900 = pickle.load(fichier)  
    tab2_csNOX5_mbH_D_200_900 = pickle.load(fichier)
        
with open(f'../hNOX5_mbW/data/{data}_hNOX5_mbW_3000-6000.pkl', 'rb') as fichier:
    tab1_hNOX5_mbW_3000_6000 = pickle.load(fichier)
    tab2_hNOX5_mbW_3000_6000 = pickle.load(fichier)   
with open(f'../hNOX5_mbW/data/replicas/300ns/{data}_hNOX5_mbW_A_200-900.pkl', 'rb') as fichier:
    tab1_hNOX5_mbW_A_200_900 = pickle.load(fichier)  
    tab2_hNOX5_mbW_A_200_900 = pickle.load(fichier)      
with open(f'../hNOX5_mbW/data/replicas/405ns/{data}_hNOX5_mbW_B_200-900.pkl', 'rb') as fichier:
    tab1_hNOX5_mbW_B_200_900 = pickle.load(fichier)  
    tab2_hNOX5_mbW_B_200_900 = pickle.load(fichier) 
with open(f'../hNOX5_mbW/data/replicas/510ns/{data}_hNOX5_mbW_C_200-900.pkl', 'rb') as fichier:
    tab1_hNOX5_mbW_C_200_900 = pickle.load(fichier)  
    tab2_hNOX5_mbW_C_200_900 = pickle.load(fichier)     
with open(f'../hNOX5_mbW/data/replicas/600ns/{data}_hNOX5_mbW_D_200-900.pkl', 'rb') as fichier:
    tab1_hNOX5_mbW_D_200_900 = pickle.load(fichier)  
    tab2_hNOX5_mbW_D_200_900 = pickle.load(fichier) 
    
with open(f'../hNOX5_mbH/data/{data}_hNOX5_mbH_3000-6000.pkl', 'rb') as fichier:
    tab1_hNOX5_mbH_3000_6000 = pickle.load(fichier)
    tab2_hNOX5_mbH_3000_6000 = pickle.load(fichier)   
with open(f'../hNOX5_mbH/data/replicas/300ns/{data}_hNOX5_mbH_A_200-900.pkl', 'rb') as fichier:
    tab1_hNOX5_mbH_A_200_900 = pickle.load(fichier)  
    tab2_hNOX5_mbH_A_200_900 = pickle.load(fichier)      
with open(f'../hNOX5_mbH/data/replicas/405ns/{data}_hNOX5_mbH_B_200-900.pkl', 'rb') as fichier:
    tab1_hNOX5_mbH_B_200_900 = pickle.load(fichier)  
    tab2_hNOX5_mbH_B_200_900 = pickle.load(fichier) 
with open(f'../hNOX5_mbH/data/replicas/510ns/{data}_hNOX5_mbH_C_200-900.pkl', 'rb') as fichier:
    tab1_hNOX5_mbH_C_200_900 = pickle.load(fichier)  
    tab2_hNOX5_mbH_C_200_900 = pickle.load(fichier)     
with open(f'../hNOX5_mbH/data/replicas/600ns/{data}_hNOX5_mbH_D_200-900.pkl', 'rb') as fichier:
    tab1_hNOX5_mbH_D_200_900 = pickle.load(fichier)  
    tab2_hNOX5_mbH_D_200_900 = pickle.load(fichier) 


#%% Delta G tot all traj replicas   Figure8
listcol = [0]
TOT = np.zeros((4,1))
A = np.zeros((4,1))
B = np.zeros((4,1))
C = np.zeros((4,1))
D = np.zeros((4,1))
E = np.zeros((4,1))
TOT[0] = sum(tab1_csNOX5_mbW_3000_6000[listcol])
A[0] = sum(tab1_csNOX5_mbW_A_200_900[listcol])
B[0] = sum(tab1_csNOX5_mbW_B_200_900[listcol])
C[0] = sum(tab1_csNOX5_mbW_C_200_900[listcol])
D[0] = sum(tab1_csNOX5_mbW_D_200_900[listcol])
E[0] = (sum(tab1_csNOX5_mbW_A_200_900[listcol]) + sum(tab1_csNOX5_mbW_B_200_900[listcol]) + sum(tab1_csNOX5_mbW_C_200_900[listcol]) + sum(tab1_csNOX5_mbW_D_200_900[listcol])) / 4
TOT[1] = sum(tab1_csNOX5_mbH_3000_6000[listcol])
A[1] = sum(tab1_csNOX5_mbH_A_200_900[listcol])
B[1] = sum(tab1_csNOX5_mbH_B_200_900[listcol])
C[1] = sum(tab1_csNOX5_mbH_C_200_900[listcol])
D[1] = sum(tab1_csNOX5_mbH_D_200_900[listcol])
E[1] = (sum(tab1_csNOX5_mbH_A_200_900[listcol]) + sum(tab1_csNOX5_mbH_B_200_900[listcol]) + sum(tab1_csNOX5_mbH_C_200_900[listcol]) + sum(tab1_csNOX5_mbH_D_200_900[listcol])) / 4
TOT[2] = sum(tab1_hNOX5_mbW_3000_6000[listcol])
A[2] = sum(tab1_hNOX5_mbW_A_200_900[listcol])
B[2] = sum(tab1_hNOX5_mbW_B_200_900[listcol])
C[2] = sum(tab1_hNOX5_mbW_C_200_900[listcol])
D[2] = sum(tab1_hNOX5_mbW_D_200_900[listcol])
E[2] = (sum(tab1_hNOX5_mbW_A_200_900[listcol]) + sum(tab1_hNOX5_mbW_B_200_900[listcol]) + sum(tab1_hNOX5_mbW_C_200_900[listcol]) + sum(tab1_hNOX5_mbW_D_200_900[listcol])) / 4
TOT[3] = sum(tab1_hNOX5_mbH_3000_6000[listcol])
A[3] = sum(tab1_hNOX5_mbH_A_200_900[listcol])
B[3] = sum(tab1_hNOX5_mbH_B_200_900[listcol])
C[3] = sum(tab1_hNOX5_mbH_C_200_900[listcol])
D[3] = sum(tab1_hNOX5_mbH_D_200_900[listcol])
E[3] = (sum(tab1_hNOX5_mbH_A_200_900[listcol]) + sum(tab1_hNOX5_mbH_B_200_900[listcol]) + sum(tab1_hNOX5_mbH_C_200_900[listcol]) + sum(tab1_hNOX5_mbH_D_200_900[listcol])) / 4

error_TOT = np.zeros((4,1))
error_A = np.zeros((4,1))
error_B = np.zeros((4,1))
error_C = np.zeros((4,1))
error_D = np.zeros((4,1))
error_E = np.zeros((4,1))
error_TOT[0] = sum(tab2_csNOX5_mbW_3000_6000[listcol])
error_A[0] = sum(tab2_csNOX5_mbW_A_200_900[listcol])
error_B[0] = sum(tab2_csNOX5_mbW_B_200_900[listcol])
error_C[0] = sum(tab2_csNOX5_mbW_C_200_900[listcol])
error_D[0] = sum(tab2_csNOX5_mbW_D_200_900[listcol])
error_E[0] = (sum(tab2_csNOX5_mbW_A_200_900[listcol])+sum(tab2_csNOX5_mbW_B_200_900[listcol])+sum(tab2_csNOX5_mbW_C_200_900[listcol])+sum(tab2_csNOX5_mbW_D_200_900[listcol])) / np.sqrt(4)
error_TOT[1] = sum(tab2_csNOX5_mbH_3000_6000[listcol])
error_A[1] = sum(tab2_csNOX5_mbH_A_200_900[listcol])
error_B[1] = sum(tab2_csNOX5_mbH_B_200_900[listcol])
error_C[1] = sum(tab2_csNOX5_mbH_C_200_900[listcol])
error_D[1] = sum(tab2_csNOX5_mbH_D_200_900[listcol])
error_E[1] = (sum(tab2_csNOX5_mbH_A_200_900[listcol])+sum(tab2_csNOX5_mbH_B_200_900[listcol])+sum(tab2_csNOX5_mbH_C_200_900[listcol])+sum(tab2_csNOX5_mbH_D_200_900[listcol])) / np.sqrt(4)
error_TOT[2] = sum(tab2_hNOX5_mbW_3000_6000[listcol])
error_A[2] = sum(tab2_hNOX5_mbW_A_200_900[listcol])
error_B[2] = sum(tab2_hNOX5_mbW_B_200_900[listcol])
error_C[2] = sum(tab2_hNOX5_mbW_C_200_900[listcol])
error_D[2] = sum(tab2_hNOX5_mbW_D_200_900[listcol])
error_E[2] = (sum(tab2_hNOX5_mbW_A_200_900[listcol])+sum(tab2_hNOX5_mbW_B_200_900[listcol])+sum(tab2_hNOX5_mbW_C_200_900[listcol])+sum(tab2_hNOX5_mbW_D_200_900[listcol])) / np.sqrt(4)
error_TOT[3] = sum(tab2_hNOX5_mbH_3000_6000[listcol])
error_A[3] = sum(tab2_hNOX5_mbH_A_200_900[listcol])
error_B[3] = sum(tab2_hNOX5_mbH_B_200_900[listcol])
error_C[3] = sum(tab2_hNOX5_mbH_C_200_900[listcol])
error_D[3] = sum(tab2_hNOX5_mbH_D_200_900[listcol])
error_E[3] = (sum(tab2_hNOX5_mbH_A_200_900[listcol])+sum(tab2_hNOX5_mbH_B_200_900[listcol])+sum(tab2_hNOX5_mbH_C_200_900[listcol])+sum(tab2_hNOX5_mbH_D_200_900[listcol])) / np.sqrt(4)

elements = ['csNOX5_mbW', 'csNOX5_mbH', 'hNOX5_mbW', 'hNOX5_mbH']
origine = np.zeros((4,1))
barWidth = 0.12
plt.figure(figsize=(12,8), dpi=300)

br1 = 0
br2 = br1 + barWidth 
br3 = br2 + barWidth 
br4 = br3 + barWidth 
br5 = br4 + barWidth 
br6 = br5 + barWidth 
br7 = 1
br8 = br7 + barWidth 
br9 = br8 + barWidth 
br10 = br9 + barWidth 
br11 = br10 + barWidth 
br12 = br11 + barWidth 
br13 = 2
br14 = br13 + barWidth 
br15 = br14 + barWidth 
br16 = br15 + barWidth 
br17 = br16 + barWidth 
br18 = br17 + barWidth 
br19 = 3 
br20 = br19 + barWidth 
br21 = br20 + barWidth 
br22 = br21 + barWidth 
br23 = br22 + barWidth 
br24 = br23 + barWidth 
# XRD Neutral
plt.bar(br1, A[0, 0], width=barWidth, edgecolor='black', label='dyn_300ns', color='orangered', yerr=error_A[0, 0], ecolor='black', capsize=5)
plt.bar(br2, B[0, 0], width=barWidth, edgecolor='black', label='dyn_405ns', color='firebrick', yerr=error_B[0, 0], ecolor='black', capsize=5)
plt.bar(br3, C[0, 0], width=barWidth, edgecolor='black', label='dyn_510ns', color='crimson', yerr=error_C[0, 0], ecolor='black', capsize=5)
plt.bar(br4, D[0, 0], width=barWidth, edgecolor='black', label='dyn_600ns', color='lightcoral', yerr=error_D[0, 0], ecolor='black', capsize=5)
plt.bar(br5, E[0, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkred', yerr=error_E[0, 0], ecolor='black', capsize=5)
plt.bar(br6, TOT[0, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='red', yerr=error_TOT[0, 0], ecolor='black', capsize=5)
# XRD Anionic
plt.bar(br7, A[1, 0], width=barWidth, edgecolor='black', label='dyn_300ns', color='peru', yerr=error_A[1, 0], ecolor='black', capsize=5)
plt.bar(br8, B[1, 0], width=barWidth, edgecolor='black', label='dyn_405ns', color='coral', yerr=error_B[1, 0], ecolor='black', capsize=5)
plt.bar(br9, C[1, 0], width=barWidth, edgecolor='black', label='dyn_510ns', color='orangered', yerr=error_C[1, 0], ecolor='black', capsize=5)
plt.bar(br10, D[1, 0], width=barWidth, edgecolor='black', label='dyn_600ns', color='lightsalmon', yerr=error_D[1, 0], ecolor='black', capsize=5)
plt.bar(br11, E[1, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkorange', yerr=error_E[1, 0], ecolor='black', capsize=5)
plt.bar(br12, TOT[1, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='orange', yerr=error_TOT[1, 0], ecolor='black', capsize=5)
# CEM Neutral
plt.bar(br13, A[2, 0], width=barWidth, edgecolor='black', label='dyn_300ns', color='royalblue', yerr=error_A[2, 0], ecolor='black', capsize=5)
plt.bar(br14, B[2, 0], width=barWidth, edgecolor='black', label='dyn_405ns', color='navy', yerr=error_B[2, 0], ecolor='black', capsize=5)
plt.bar(br15, C[2, 0], width=barWidth, edgecolor='black', label='dyn_510ns', color='skyblue', yerr=error_C[2, 0], ecolor='black', capsize=5)
plt.bar(br16, D[2, 0], width=barWidth, edgecolor='black', label='dyn_600ns', color='lightblue', yerr=error_D[2, 0], ecolor='black', capsize=5)
plt.bar(br17, E[2, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkblue', yerr=error_E[2, 0], ecolor='black', capsize=5)
plt.bar(br18, TOT[2, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='blue', yerr=error_TOT[2, 0], ecolor='black', capsize=5)
# CEM Anionic
plt.bar(br19, A[3, 0], width=barWidth, edgecolor='black', label='dyn_300ns', color='indigo', yerr=error_A[3, 0], ecolor='black', capsize=5)
plt.bar(br20, B[3, 0], width=barWidth, edgecolor='black', label='dyn_405ns', color='blueviolet', yerr=error_B[3, 0], ecolor='black', capsize=5)
plt.bar(br21, C[3, 0], width=barWidth, edgecolor='black', label='dyn_510ns', color='mediumorchid', yerr=error_C[3, 0], ecolor='black', capsize=5)
plt.bar(br22, D[3, 0], width=barWidth, edgecolor='black', label='dyn_600ns', color='lavender', yerr=error_D[3, 0], ecolor='black', capsize=5)
plt.bar(br23, E[3, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkviolet', yerr=error_E[3, 0], ecolor='black', capsize=5)
plt.bar(br24, TOT[3, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='purple', yerr=error_TOT[3, 0], ecolor='black', capsize=5)
plt.axhline(y=0, lw=1.0, color='black', linestyle='-')
plt.plot(elements, origine, lw=1.0, color='black', linestyle='-')
for i in np.arange(-0.2, 6.2, 1.0):
    plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
for i in np.arange(-3.0, 2.5, 0.01):
    plt.axhline(y=i, color='gray', linestyle='--', linewidth=0.5)
#plt.grid()
plt.ylabel('ΔG(eV)', fontsize="20") 	
plt.xlabel('Systems', fontsize="20")
plt.xticks(fontsize=20)
plt.xticks([r + 0.20 + barWidth for r in range (len(TOT))],elements)
plt.yticks(fontsize=15)
plt.autoscale(enable=True, tight=True)
plt.xlim(-0.2, 3.8)
plt.ylim(-0.3, 0.1)

ax = plt.gca()  
for spine in ax.spines.values():
    spine.set_linewidth(1.5) 
ax.tick_params(width=1.5, length=8) 
fname = 'Figure8.tiff'
plt.savefig(fname, format="tiff", dpi=300)

