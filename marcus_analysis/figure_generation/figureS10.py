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



#%% Delta G decomposition (average replicas + lns-ong simulations)  (FigureS10)

nombre_contrib = np.zeros((6,1))
listcol1 = [2]
listcol2 = [13]
listcol3 = [7]
listcol4 = [8]
listcol5 = [15]
listcol6 = [16]
TOT1 = np.zeros((4,1))
MOY1 = np.zeros((4,1))
TOT2 = np.zeros((4,1))
MOY2 = np.zeros((4,1))
TOT3 = np.zeros((4,1))
MOY3 = np.zeros((4,1))
TOT4 = np.zeros((4,1))
MOY4 = np.zeros((4,1))
TOT5 = np.zeros((4,1))
MOY5 = np.zeros((4,1))
TOT6 = np.zeros((4,1))
MOY6 = np.zeros((4,1))
TOT1[0] = sum(tab1_csNOX5_mbW_3000_6000[listcol1])
MOY1[0] = (sum(tab1_csNOX5_mbW_A_200_900[listcol1]) + sum(tab1_csNOX5_mbW_B_200_900[listcol1]) + sum(tab1_csNOX5_mbW_C_200_900[listcol1]) + sum(tab1_csNOX5_mbW_D_200_900[listcol1])) / 4
TOT1[1] = sum(tab1_csNOX5_mbH_3000_6000[listcol1])
MOY1[1] = (sum(tab1_csNOX5_mbH_A_200_900[listcol1]) + sum(tab1_csNOX5_mbH_B_200_900[listcol1]) + sum(tab1_csNOX5_mbH_C_200_900[listcol1]) + sum(tab1_csNOX5_mbH_D_200_900[listcol1])) / 4
TOT1[2] = sum(tab1_hNOX5_mbW_3000_6000[listcol1])
MOY1[2] = (sum(tab1_hNOX5_mbW_A_200_900[listcol1]) + sum(tab1_hNOX5_mbW_B_200_900[listcol1]) + sum(tab1_hNOX5_mbW_C_200_900[listcol1]) + sum(tab1_hNOX5_mbW_D_200_900[listcol1])) / 4
TOT1[3] = sum(tab1_hNOX5_mbH_3000_6000[listcol1])
MOY1[3] = (sum(tab1_hNOX5_mbH_A_200_900[listcol1]) + sum(tab1_hNOX5_mbH_B_200_900[listcol1]) + sum(tab1_hNOX5_mbH_C_200_900[listcol1]) + sum(tab1_hNOX5_mbH_D_200_900[listcol1])) / 4
TOT2[0] = sum(tab1_csNOX5_mbW_3000_6000[listcol2])
MOY2[0] = (sum(tab1_csNOX5_mbW_A_200_900[listcol2]) + sum(tab1_csNOX5_mbW_B_200_900[listcol2]) + sum(tab1_csNOX5_mbW_C_200_900[listcol2]) + sum(tab1_csNOX5_mbW_D_200_900[listcol2])) / 4
TOT2[1] = sum(tab1_csNOX5_mbH_3000_6000[listcol2])
MOY2[1] = (sum(tab1_csNOX5_mbH_A_200_900[listcol2]) + sum(tab1_csNOX5_mbH_B_200_900[listcol2]) + sum(tab1_csNOX5_mbH_C_200_900[listcol2]) + sum(tab1_csNOX5_mbH_D_200_900[listcol2])) / 4
TOT2[2] = sum(tab1_hNOX5_mbW_3000_6000[listcol2])
MOY2[2] = (sum(tab1_hNOX5_mbW_A_200_900[listcol2]) + sum(tab1_hNOX5_mbW_B_200_900[listcol2]) + sum(tab1_hNOX5_mbW_C_200_900[listcol2]) + sum(tab1_hNOX5_mbW_D_200_900[listcol2])) / 4
TOT2[3] = sum(tab1_hNOX5_mbH_3000_6000[listcol2])
MOY2[3] = (sum(tab1_hNOX5_mbH_A_200_900[listcol2]) + sum(tab1_hNOX5_mbH_B_200_900[listcol2]) + sum(tab1_hNOX5_mbH_C_200_900[listcol2]) + sum(tab1_hNOX5_mbH_D_200_900[listcol2])) / 4
TOT3[0] = sum(tab1_csNOX5_mbW_3000_6000[listcol3])
MOY3[0] = (sum(tab1_csNOX5_mbW_A_200_900[listcol3]) + sum(tab1_csNOX5_mbW_B_200_900[listcol3]) + sum(tab1_csNOX5_mbW_C_200_900[listcol3]) + sum(tab1_csNOX5_mbW_D_200_900[listcol3])) / 4
TOT3[1] = sum(tab1_csNOX5_mbH_3000_6000[listcol3])
MOY3[1] = (sum(tab1_csNOX5_mbH_A_200_900[listcol3]) + sum(tab1_csNOX5_mbH_B_200_900[listcol3]) + sum(tab1_csNOX5_mbH_C_200_900[listcol3]) + sum(tab1_csNOX5_mbH_D_200_900[listcol3])) / 4
TOT3[2] = sum(tab1_hNOX5_mbW_3000_6000[listcol3])
MOY3[2] = (sum(tab1_hNOX5_mbW_A_200_900[listcol3]) + sum(tab1_hNOX5_mbW_B_200_900[listcol3]) + sum(tab1_hNOX5_mbW_C_200_900[listcol3]) + sum(tab1_hNOX5_mbW_D_200_900[listcol3])) / 4
TOT3[3] = sum(tab1_hNOX5_mbH_3000_6000[listcol3])
MOY3[3] = (sum(tab1_hNOX5_mbH_A_200_900[listcol3]) + sum(tab1_hNOX5_mbH_B_200_900[listcol3]) + sum(tab1_hNOX5_mbH_C_200_900[listcol3]) + sum(tab1_hNOX5_mbH_D_200_900[listcol3])) / 4
TOT4[0] = sum(tab1_csNOX5_mbW_3000_6000[listcol4])
MOY4[0] = (sum(tab1_csNOX5_mbW_A_200_900[listcol4]) + sum(tab1_csNOX5_mbW_B_200_900[listcol4]) + sum(tab1_csNOX5_mbW_C_200_900[listcol4]) + sum(tab1_csNOX5_mbW_D_200_900[listcol4])) / 4
TOT4[1] = sum(tab1_csNOX5_mbH_3000_6000[listcol4])
MOY4[1] = (sum(tab1_csNOX5_mbH_A_200_900[listcol4]) + sum(tab1_csNOX5_mbH_B_200_900[listcol4]) + sum(tab1_csNOX5_mbH_C_200_900[listcol4]) + sum(tab1_csNOX5_mbH_D_200_900[listcol4])) / 4
TOT4[2] = sum(tab1_hNOX5_mbW_3000_6000[listcol4])
MOY4[2] = (sum(tab1_hNOX5_mbW_A_200_900[listcol4]) + sum(tab1_hNOX5_mbW_B_200_900[listcol4]) + sum(tab1_hNOX5_mbW_C_200_900[listcol4]) + sum(tab1_hNOX5_mbW_D_200_900[listcol4])) / 4
TOT4[3] = sum(tab1_hNOX5_mbH_3000_6000[listcol4])
MOY4[3] = (sum(tab1_hNOX5_mbH_A_200_900[listcol4]) + sum(tab1_hNOX5_mbH_B_200_900[listcol4]) + sum(tab1_hNOX5_mbH_C_200_900[listcol4]) + sum(tab1_hNOX5_mbH_D_200_900[listcol4])) / 4
TOT5[0] = sum(tab1_csNOX5_mbW_3000_6000[listcol5])
MOY5[0] = (sum(tab1_csNOX5_mbW_A_200_900[listcol5]) + sum(tab1_csNOX5_mbW_B_200_900[listcol5]) + sum(tab1_csNOX5_mbW_C_200_900[listcol5]) + sum(tab1_csNOX5_mbW_D_200_900[listcol5])) / 4
TOT5[1] = sum(tab1_csNOX5_mbH_3000_6000[listcol5])
MOY5[1] = (sum(tab1_csNOX5_mbH_A_200_900[listcol5]) + sum(tab1_csNOX5_mbH_B_200_900[listcol5]) + sum(tab1_csNOX5_mbH_C_200_900[listcol5]) + sum(tab1_csNOX5_mbH_D_200_900[listcol5])) / 4
TOT5[2] = sum(tab1_hNOX5_mbW_3000_6000[listcol5])
MOY5[2] = (sum(tab1_hNOX5_mbW_A_200_900[listcol5]) + sum(tab1_hNOX5_mbW_B_200_900[listcol5]) + sum(tab1_hNOX5_mbW_C_200_900[listcol5]) + sum(tab1_hNOX5_mbW_D_200_900[listcol5])) / 4
TOT5[3] = sum(tab1_hNOX5_mbH_3000_6000[listcol5])
MOY5[3] = (sum(tab1_hNOX5_mbH_A_200_900[listcol5]) + sum(tab1_hNOX5_mbH_B_200_900[listcol5]) + sum(tab1_hNOX5_mbH_C_200_900[listcol5]) + sum(tab1_hNOX5_mbH_D_200_900[listcol5])) / 4
TOT6[0] = sum(tab1_csNOX5_mbW_3000_6000[listcol6])
MOY6[0] = (sum(tab1_csNOX5_mbW_A_200_900[listcol6]) + sum(tab1_csNOX5_mbW_B_200_900[listcol6]) + sum(tab1_csNOX5_mbW_C_200_900[listcol6]) + sum(tab1_csNOX5_mbW_D_200_900[listcol6])) / 4
TOT6[1] = sum(tab1_csNOX5_mbH_3000_6000[listcol6])
MOY6[1] = (sum(tab1_csNOX5_mbH_A_200_900[listcol6]) + sum(tab1_csNOX5_mbH_B_200_900[listcol6]) + sum(tab1_csNOX5_mbH_C_200_900[listcol6]) + sum(tab1_csNOX5_mbH_D_200_900[listcol6])) / 4
TOT6[2] = sum(tab1_hNOX5_mbW_3000_6000[listcol6])
MOY6[2] = (sum(tab1_hNOX5_mbW_A_200_900[listcol6]) + sum(tab1_hNOX5_mbW_B_200_900[listcol6]) + sum(tab1_hNOX5_mbW_C_200_900[listcol6]) + sum(tab1_hNOX5_mbW_D_200_900[listcol6])) / 4
TOT6[3] = sum(tab1_hNOX5_mbH_3000_6000[listcol6])
MOY6[3] = (sum(tab1_hNOX5_mbH_A_200_900[listcol6]) + sum(tab1_hNOX5_mbH_B_200_900[listcol6]) + sum(tab1_hNOX5_mbH_C_200_900[listcol6]) + sum(tab1_hNOX5_mbH_D_200_900[listcol6])) / 4
error_TOT1 = np.zeros((4,1))
error_MOY1 = np.zeros((4,1))
error_TOT2 = np.zeros((4,1))
error_MOY2 = np.zeros((4,1))
error_TOT3 = np.zeros((4,1))
error_MOY3 = np.zeros((4,1))
error_TOT4 = np.zeros((4,1))
error_MOY4 = np.zeros((4,1))
error_TOT5 = np.zeros((4,1))
error_MOY5 = np.zeros((4,1))
error_TOT6 = np.zeros((4,1))
error_MOY6 = np.zeros((4,1))
error_TOT1[0] = sum(tab2_csNOX5_mbW_3000_6000[listcol1])
error_MOY1[0] = (sum(tab2_csNOX5_mbW_A_200_900[listcol1])+sum(tab2_csNOX5_mbW_B_200_900[listcol1])+sum(tab2_csNOX5_mbW_C_200_900[listcol1])+sum(tab2_csNOX5_mbW_D_200_900[listcol1])) / np.sqrt(4)
error_TOT1[1] = sum(tab2_csNOX5_mbH_3000_6000[listcol1])
error_MOY1[1] = (sum(tab2_csNOX5_mbH_A_200_900[listcol1])+sum(tab2_csNOX5_mbH_B_200_900[listcol1])+sum(tab2_csNOX5_mbH_C_200_900[listcol1])+sum(tab2_csNOX5_mbH_D_200_900[listcol1])) / np.sqrt(4)
error_TOT1[2] = sum(tab2_hNOX5_mbW_3000_6000[listcol1])
error_MOY1[2] = (sum(tab2_hNOX5_mbW_A_200_900[listcol1])+sum(tab2_hNOX5_mbW_B_200_900[listcol1])+sum(tab2_hNOX5_mbW_C_200_900[listcol1])+sum(tab2_hNOX5_mbW_D_200_900[listcol1])) / np.sqrt(4)
error_TOT1[3] = sum(tab2_hNOX5_mbH_3000_6000[listcol1])
error_MOY1[3] = (sum(tab2_hNOX5_mbH_A_200_900[listcol1])+sum(tab2_hNOX5_mbH_B_200_900[listcol1])+sum(tab2_hNOX5_mbH_C_200_900[listcol1])+sum(tab2_hNOX5_mbH_D_200_900[listcol1])) / np.sqrt(4)
error_TOT2[0] = sum(tab2_csNOX5_mbW_3000_6000[listcol2])
error_MOY2[0] = (sum(tab2_csNOX5_mbW_A_200_900[listcol2])+sum(tab2_csNOX5_mbW_B_200_900[listcol2])+sum(tab2_csNOX5_mbW_C_200_900[listcol2])+sum(tab2_csNOX5_mbW_D_200_900[listcol2])) / np.sqrt(4)
error_TOT2[1] = sum(tab2_csNOX5_mbH_3000_6000[listcol2])
error_MOY2[1] = (sum(tab2_csNOX5_mbH_A_200_900[listcol2])+sum(tab2_csNOX5_mbH_B_200_900[listcol2])+sum(tab2_csNOX5_mbH_C_200_900[listcol2])+sum(tab2_csNOX5_mbH_D_200_900[listcol2])) / np.sqrt(4)
error_TOT2[2] = sum(tab2_hNOX5_mbW_3000_6000[listcol2])
error_MOY2[2] = (sum(tab2_hNOX5_mbW_A_200_900[listcol2])+sum(tab2_hNOX5_mbW_B_200_900[listcol2])+sum(tab2_hNOX5_mbW_C_200_900[listcol2])+sum(tab2_hNOX5_mbW_D_200_900[listcol2])) / np.sqrt(4)
error_TOT2[3] = sum(tab2_hNOX5_mbH_3000_6000[listcol2])
error_MOY2[3] = (sum(tab2_hNOX5_mbH_A_200_900[listcol2])+sum(tab2_hNOX5_mbH_B_200_900[listcol2])+sum(tab2_hNOX5_mbH_C_200_900[listcol2])+sum(tab2_hNOX5_mbH_D_200_900[listcol2])) / np.sqrt(4)
error_TOT3[0] = sum(tab2_csNOX5_mbW_3000_6000[listcol3])
error_MOY3[0] = (sum(tab2_csNOX5_mbW_A_200_900[listcol3])+sum(tab2_csNOX5_mbW_B_200_900[listcol3])+sum(tab2_csNOX5_mbW_C_200_900[listcol3])+sum(tab2_csNOX5_mbW_D_200_900[listcol3])) / np.sqrt(4)
error_TOT3[1] = sum(tab2_csNOX5_mbH_3000_6000[listcol3])
error_MOY3[1] = (sum(tab2_csNOX5_mbH_A_200_900[listcol3])+sum(tab2_csNOX5_mbH_B_200_900[listcol3])+sum(tab2_csNOX5_mbH_C_200_900[listcol3])+sum(tab2_csNOX5_mbH_D_200_900[listcol3])) / np.sqrt(4)
error_TOT3[2] = sum(tab2_hNOX5_mbW_3000_6000[listcol3])
error_MOY3[2] = (sum(tab2_hNOX5_mbW_A_200_900[listcol3])+sum(tab2_hNOX5_mbW_B_200_900[listcol3])+sum(tab2_hNOX5_mbW_C_200_900[listcol3])+sum(tab2_hNOX5_mbW_D_200_900[listcol3])) / np.sqrt(4)
error_TOT3[3] = sum(tab2_hNOX5_mbH_3000_6000[listcol3])
error_MOY3[3] = (sum(tab2_hNOX5_mbH_A_200_900[listcol3])+sum(tab2_hNOX5_mbH_B_200_900[listcol3])+sum(tab2_hNOX5_mbH_C_200_900[listcol3])+sum(tab2_hNOX5_mbH_D_200_900[listcol3])) / np.sqrt(4)
error_TOT4[0] = sum(tab2_csNOX5_mbW_3000_6000[listcol4])
error_MOY4[0] = (sum(tab2_csNOX5_mbW_A_200_900[listcol4])+sum(tab2_csNOX5_mbW_B_200_900[listcol4])+sum(tab2_csNOX5_mbW_C_200_900[listcol4])+sum(tab2_csNOX5_mbW_D_200_900[listcol4])) / np.sqrt(4)
error_TOT4[1] = sum(tab2_csNOX5_mbH_3000_6000[listcol4])
error_MOY4[1] = (sum(tab2_csNOX5_mbH_A_200_900[listcol4])+sum(tab2_csNOX5_mbH_B_200_900[listcol4])+sum(tab2_csNOX5_mbH_C_200_900[listcol4])+sum(tab2_csNOX5_mbH_D_200_900[listcol4])) / np.sqrt(4)
error_TOT4[2] = sum(tab2_hNOX5_mbW_3000_6000[listcol4])
error_MOY4[2] = (sum(tab2_hNOX5_mbW_A_200_900[listcol4])+sum(tab2_hNOX5_mbW_B_200_900[listcol4])+sum(tab2_hNOX5_mbW_C_200_900[listcol4])+sum(tab2_hNOX5_mbW_D_200_900[listcol4])) / np.sqrt(4)
error_TOT4[3] = sum(tab2_hNOX5_mbH_3000_6000[listcol4])
error_MOY4[3] = (sum(tab2_hNOX5_mbH_A_200_900[listcol4])+sum(tab2_hNOX5_mbH_B_200_900[listcol4])+sum(tab2_hNOX5_mbH_C_200_900[listcol4])+sum(tab2_hNOX5_mbH_D_200_900[listcol4])) / np.sqrt(4)
error_TOT5[0] = sum(tab2_csNOX5_mbW_3000_6000[listcol5])
error_MOY5[0] = (sum(tab2_csNOX5_mbW_A_200_900[listcol5])+sum(tab2_csNOX5_mbW_B_200_900[listcol5])+sum(tab2_csNOX5_mbW_C_200_900[listcol5])+sum(tab2_csNOX5_mbW_D_200_900[listcol5])) / np.sqrt(4)
error_TOT5[1] = sum(tab2_csNOX5_mbH_3000_6000[listcol5])
error_MOY5[1] = (sum(tab2_csNOX5_mbH_A_200_900[listcol5])+sum(tab2_csNOX5_mbH_B_200_900[listcol5])+sum(tab2_csNOX5_mbH_C_200_900[listcol5])+sum(tab2_csNOX5_mbH_D_200_900[listcol5])) / np.sqrt(4)
error_TOT5[2] = sum(tab2_hNOX5_mbW_3000_6000[listcol5])
error_MOY5[2] = (sum(tab2_hNOX5_mbW_A_200_900[listcol5])+sum(tab2_hNOX5_mbW_B_200_900[listcol5])+sum(tab2_hNOX5_mbW_C_200_900[listcol5])+sum(tab2_hNOX5_mbW_D_200_900[listcol5])) / np.sqrt(4)
error_TOT5[3] = sum(tab2_hNOX5_mbH_3000_6000[listcol5])
error_MOY5[3] = (sum(tab2_hNOX5_mbH_A_200_900[listcol5])+sum(tab2_hNOX5_mbH_B_200_900[listcol5])+sum(tab2_hNOX5_mbH_C_200_900[listcol5])+sum(tab2_hNOX5_mbH_D_200_900[listcol5])) / np.sqrt(4)
error_TOT6[0] = sum(tab2_csNOX5_mbW_3000_6000[listcol6])
error_MOY6[0] = (sum(tab2_csNOX5_mbW_A_200_900[listcol6])+sum(tab2_csNOX5_mbW_B_200_900[listcol6])+sum(tab2_csNOX5_mbW_C_200_900[listcol6])+sum(tab2_csNOX5_mbW_D_200_900[listcol6])) / np.sqrt(4)
error_TOT6[1] = sum(tab2_csNOX5_mbH_3000_6000[listcol6])
error_MOY6[1] = (sum(tab2_csNOX5_mbH_A_200_900[listcol6])+sum(tab2_csNOX5_mbH_B_200_900[listcol6])+sum(tab2_csNOX5_mbH_C_200_900[listcol6])+sum(tab2_csNOX5_mbH_D_200_900[listcol6])) / np.sqrt(4)
error_TOT6[2] = sum(tab2_hNOX5_mbW_3000_6000[listcol6])
error_MOY6[2] = (sum(tab2_hNOX5_mbW_A_200_900[listcol6])+sum(tab2_hNOX5_mbW_B_200_900[listcol6])+sum(tab2_hNOX5_mbW_C_200_900[listcol6])+sum(tab2_hNOX5_mbW_D_200_900[listcol6])) / np.sqrt(4)
error_TOT6[3] = sum(tab2_hNOX5_mbH_3000_6000[listcol6])
error_MOY6[3] = (sum(tab2_hNOX5_mbH_A_200_900[listcol6])+sum(tab2_hNOX5_mbH_B_200_900[listcol6])+sum(tab2_hNOX5_mbH_C_200_900[listcol6])+sum(tab2_hNOX5_mbH_D_200_900[listcol6])) / np.sqrt(4)
elements = ['FADH', 'MEMB', 'TM', 'DH', 'ENV', 'HEME']
origine = np.zeros((6,1))
barWidth = 0.085
plt.figure(figsize=(12,8), dpi=300)

br1 = 0
br2 = br1 + barWidth 
br3 = br2 + barWidth 
br4 = br3 + barWidth 
br5 = br4 + barWidth 
br6 = br5 + barWidth 
br7 = br6 + barWidth 
br8 = br7 + barWidth 
br9 = 1
br10 = br9 + barWidth 
br11 = br10 + barWidth 
br12 = br11 + barWidth 
br13 = br12 + barWidth 
br14 = br13 + barWidth 
br15 = br14 + barWidth 
br16 = br15 + barWidth 
br17 = 2
br18 = br17 + barWidth 
br19 = br18 + barWidth 
br20 = br19 + barWidth 
br21 = br20 + barWidth 
br22 = br21 + barWidth 
br23 = br22 + barWidth 
br24 = br23 + barWidth 
br25 = 3 
br26 = br25 + barWidth 
br27 = br26 + barWidth 
br28 = br27 + barWidth 
br29 = br28 + barWidth 
br30 = br29 + barWidth 
br31 = br30 + barWidth 
br32 = br31 + barWidth 
br33 = 4 
br34 = br33 + barWidth 
br35 = br34 + barWidth 
br36 = br35 + barWidth 
br37 = br36 + barWidth 
br38 = br37 + barWidth 
br39 = br38 + barWidth 
br40 = br39 + barWidth 
br41 = 5 
br42 = br41 + barWidth 
br43 = br42 + barWidth 
br44 = br43 + barWidth 
br45 = br44 + barWidth 
br46 = br45 + barWidth 
br47 = br46 + barWidth 
br48 = br47 + barWidth 
# XRD Neutral
plt.bar(br1, MOY1[0, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkred', yerr=error_MOY1[0, 0], ecolor='black', capsize=5)
plt.bar(br2, TOT1[0, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='red', yerr=error_TOT1[0, 0], ecolor='black', capsize=5)
plt.bar(br9, MOY2[0, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkred', yerr=error_MOY2[0, 0], ecolor='black', capsize=5)
plt.bar(br10, TOT2[0, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='red', yerr=error_TOT2[0, 0], ecolor='black', capsize=5)
plt.bar(br17, MOY3[0, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkred', yerr=error_MOY3[0, 0], ecolor='black', capsize=5)
plt.bar(br18, TOT3[0, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='red', yerr=error_TOT3[0, 0], ecolor='black', capsize=5)
plt.bar(br25, MOY4[0, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkred', yerr=error_MOY4[0, 0], ecolor='black', capsize=5)
plt.bar(br26, TOT4[0, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='red', yerr=error_TOT4[0, 0], ecolor='black', capsize=5)
plt.bar(br33, MOY5[0, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkred', yerr=error_MOY5[0, 0], ecolor='black', capsize=5)
plt.bar(br34, TOT5[0, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='red', yerr=error_TOT5[0, 0], ecolor='black', capsize=5)
plt.bar(br41, MOY6[0, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkred', yerr=error_MOY6[0, 0], ecolor='black', capsize=5)
plt.bar(br42, TOT6[0, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='red', yerr=error_TOT6[0, 0], ecolor='black', capsize=5)
# XRD Anionic
plt.bar(br3, MOY1[1, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkorange', yerr=error_MOY1[1, 0], ecolor='black', capsize=5)
plt.bar(br4, TOT1[1, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='orange', yerr=error_TOT1[1, 0], ecolor='black', capsize=5)
plt.bar(br11, MOY2[1, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkorange', yerr=error_MOY2[1, 0], ecolor='black', capsize=5)
plt.bar(br12, TOT2[1, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='orange', yerr=error_TOT2[1, 0], ecolor='black', capsize=5)
plt.bar(br19, MOY3[1, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkorange', yerr=error_MOY3[1, 0], ecolor='black', capsize=5)
plt.bar(br20, TOT3[1, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='orange', yerr=error_TOT3[1, 0], ecolor='black', capsize=5)
plt.bar(br27, MOY4[1, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkorange', yerr=error_MOY4[1, 0], ecolor='black', capsize=5)
plt.bar(br28, TOT4[1, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='orange', yerr=error_TOT4[1, 0], ecolor='black', capsize=5)
plt.bar(br35, MOY5[1, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkorange', yerr=error_MOY5[1, 0], ecolor='black', capsize=5)
plt.bar(br36, TOT5[1, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='orange', yerr=error_TOT5[1, 0], ecolor='black', capsize=5)
plt.bar(br43, MOY6[1, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkorange', yerr=error_MOY6[1, 0], ecolor='black', capsize=5)
plt.bar(br44, TOT6[1, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='orange', yerr=error_TOT6[1, 0], ecolor='black', capsize=5)
# CEM Neutral
plt.bar(br5, MOY1[2, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkblue', yerr=error_MOY1[2, 0], ecolor='black', capsize=5)
plt.bar(br6, TOT1[2, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='blue', yerr=error_TOT1[2, 0], ecolor='black', capsize=5)
plt.bar(br13, MOY2[2, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkblue', yerr=error_MOY2[2, 0], ecolor='black', capsize=5)
plt.bar(br14, TOT2[2, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='blue', yerr=error_TOT2[2, 0], ecolor='black', capsize=5)
plt.bar(br21, MOY3[2, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkblue', yerr=error_MOY3[2, 0], ecolor='black', capsize=5)
plt.bar(br22, TOT3[2, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='blue', yerr=error_TOT3[2, 0], ecolor='black', capsize=5)
plt.bar(br29, MOY4[2, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkblue', yerr=error_MOY4[2, 0], ecolor='black', capsize=5)
plt.bar(br30, TOT4[2, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='blue', yerr=error_TOT4[2, 0], ecolor='black', capsize=5)
plt.bar(br37, MOY5[2, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkblue', yerr=error_MOY5[2, 0], ecolor='black', capsize=5)
plt.bar(br38, TOT5[2, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='blue', yerr=error_TOT5[2, 0], ecolor='black', capsize=5)
plt.bar(br45, MOY6[2, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkblue', yerr=error_MOY6[2, 0], ecolor='black', capsize=5)
plt.bar(br46, TOT6[2, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='blue', yerr=error_TOT6[2, 0], ecolor='black', capsize=5)
# CEM Anionic
plt.bar(br7, MOY1[3, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkviolet', yerr=error_MOY1[3, 0], ecolor='black', capsize=5)
plt.bar(br8, TOT1[3, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='purple', yerr=error_TOT1[3, 0], ecolor='black', capsize=5)
plt.bar(br15, MOY2[3, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkviolet', yerr=error_MOY2[3, 0], ecolor='black', capsize=5)
plt.bar(br16, TOT2[3, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='purple', yerr=error_TOT2[3, 0], ecolor='black', capsize=5)
plt.bar(br23, MOY3[3, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkviolet', yerr=error_MOY3[3, 0], ecolor='black', capsize=5)
plt.bar(br24, TOT3[3, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='purple', yerr=error_TOT3[3, 0], ecolor='black', capsize=5)
plt.bar(br31, MOY4[3, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkviolet', yerr=error_MOY4[3, 0], ecolor='black', capsize=5)
plt.bar(br32, TOT4[3, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='purple', yerr=error_TOT4[3, 0], ecolor='black', capsize=5)
plt.bar(br39, MOY5[3, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkviolet', yerr=error_MOY5[3, 0], ecolor='black', capsize=5)
plt.bar(br40, TOT5[3, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='purple', yerr=error_TOT5[3, 0], ecolor='black', capsize=5)
plt.bar(br47, MOY6[3, 0], width=barWidth, edgecolor='black', label='mean_rep', color='darkviolet', yerr=error_MOY6[3, 0], ecolor='black', capsize=5)
plt.bar(br48, TOT6[3, 0], width=barWidth, edgecolor='black', label='dyn_TOT', color='purple', yerr=error_TOT6[3, 0], ecolor='black', capsize=5)
plt.axhline(y=0, lw=1.0, color='black', linestyle='-')
plt.plot(elements, origine, lw=1.0, color='black', linestyle='-')
for i in np.arange(-0.2, 6.2, 1.0):
    plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
for i in np.arange(-11.5, 11.5, 0.1):
    plt.axhline(y=i, color='gray', linestyle='--', linewidth=0.5)
plt.ylabel('ΔG(eV)', fontsize="20") 	
plt.xticks(fontsize=15)
plt.xticks([r + 0.20 + barWidth for r in range (len(nombre_contrib))],elements)
plt.yticks(fontsize=15)
plt.autoscale(enable=True, tight=True)
plt.xlim(-0.2, 5.8)
plt.ylim(-2.5, 2.5)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(width=1.5, length=8) 
fname = 'FigureS10.tiff'
plt.savefig(fname, format="tiff", dpi=300)


