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

#%% Gap graphe all long traj   FigureS9

# Conversion factor
kcalmol2ev = 0.0433634

# Definition of system energy data files 
systems = {
    "csNOX5mbN": {
        "st1_e1": "../csNOX5_mbW/data/nrj_st1_e1_1_40.dat",
        "st1_e2": "../csNOX5_mbW/data/nrj_st1_e2_1_40.dat",
        "st2_e1": "../csNOX5_mbW/data/nrj_st2_e1_1_40.dat",
        "st2_e2": "../csNOX5_mbW/data/nrj_st2_e2_1_40.dat",
    },
    "csNOX5mbA": {
        "st1_e1": "../csNOX5_mbH/data/nrj_st1_e1_1_40.dat",
        "st1_e2": "../csNOX5_mbH/data/nrj_st1_e2_1_40.dat",
        "st2_e1": "../csNOX5_mbH/data/nrj_st2_e1_1_40.dat",
        "st2_e2": "../csNOX5_mbH/data/nrj_st2_e2_1_40.dat",
    },
    "hNOX5mbN": {
        "st1_e1": "../hNOX5_mbW/data/nrj_st1_e1_1_40.dat",
        "st1_e2": "../hNOX5_mbW/data/nrj_st1_e2_1_40.dat",
        "st2_e1": "../hNOX5_mbW/data/nrj_st2_e1_1_40.dat",
        "st2_e2": "../hNOX5_mbW/data/nrj_st2_e2_1_40.dat",
    },
    "hNOX5mbA": {
        "st1_e1": "../hNOX5_mbH/data/nrj_st1_e1_1_40.dat",
        "st1_e2": "../hNOX5_mbH/data/nrj_st1_e2_1_40.dat",
        "st2_e1": "../hNOX5_mbH/data/nrj_st2_e1_1_40.dat",
        "st2_e2": "../hNOX5_mbH/data/nrj_st2_e2_1_40.dat",
    },
}

colors = ['r', 'orange', 'blue', 'purple', 'orangered', 'firebrick', 'crimson', 'lightcoral', 'peru', 'coral', 'orangered', 'lightsalmon', 'royalblue', 'navy', 'skyblue', 'lightblue', 'indigo', 'blueviolet', 'mediumorchid', 'lavender']
linestyles = ['-', '--']  # Solid line for gap1, dotted for gap2

plt.figure(figsize=(10, 6), dpi=300)
contr = 'All'

for i, (system_name, files) in enumerate(systems.items()):
    # Load data
    energy_st1_e1 = pd.read_csv(files["st1_e1"],sep='\s+',header=None,
                            names=['Index','All','Heme1+Heme2','Heme1','Heme2','FAD','POPEup','POPGup','POPElo',
                                   'POPGlo','TM','DH','Na','Cl','Wat','Prop'])
    energy_st1_e2 = pd.read_csv(files["st1_e2"],sep='\s+',header=None,
                            names=['Index','All','Heme1+Heme2','Heme1','Heme2','FAD','POPEup','POPGup','POPElo',
                                   'POPGlo','TM','DH','Na','Cl','Wat','Prop'])
    energy_st2_e1 = pd.read_csv(files["st2_e1"],sep='\s+',header=None,
                            names=['Index','All','Heme1+Heme2','Heme1','Heme2','FAD','POPEup','POPGup','POPElo',
                                   'POPGlo','TM','DH','Na','Cl','Wat','Prop'])
    energy_st2_e2 = pd.read_csv(files["st2_e2"],sep='\s+',header=None,
                            names=['Index','All','Heme1+Heme2','Heme1','Heme2','FAD','POPEup','POPGup','POPElo',
                                   'POPGlo','TM','DH','Na','Cl','Wat','Prop'])

    # Calculation of energy gaps
    gap1 = (energy_st1_e2[contr] - energy_st1_e2['Heme1'] - energy_st1_e2['Heme2']) - (energy_st1_e1[contr] - energy_st1_e1['Heme1'] - energy_st1_e1['Heme2'])
    gap2 = (energy_st2_e2[contr] - energy_st2_e2['Heme1'] - energy_st2_e2['Heme2']) - (energy_st2_e1[contr] - energy_st2_e1['Heme1'] - energy_st2_e1['Heme2'])
    

    # Conversion in eV
    gap1 = gap1 * kcalmol2ev
    gap2 = gap2 * kcalmol2ev
    # Generate time axis
    time = np.arange(len(gap1)) / 10  # 10 frames = 1 ns
    # Rolling average calculation
    window_size = 200  # Ajuster selon le lissage voulu
    gap1_smooth = gap1.rolling(window=window_size, center=True).mean()
    gap2_smooth = gap2.rolling(window=window_size, center=True).mean()
    # Original data curves in transparent
    plt.plot(time, gap1, linestyle='-', color=colors[i % len(colors)], alpha=0.2, label=f"{system_name} Gap 1 (raw)")
    plt.plot(time, gap2, linestyle='-', color=colors[i % len(colors)], alpha=0.2, label=f"{system_name} Gap 2 (raw)")
    # Rolling average curves in solid lines
    plt.plot(time, gap1_smooth, linestyle='-', color=colors[i % len(colors)], linewidth=2, label=f"{system_name} Gap 1 (smooth)")
    plt.plot(time, gap2_smooth, linestyle=':', color=colors[i % len(colors)], linewidth=2, label=f"{system_name} Gap 2 (smooth)")

plt.xlabel("Time (ns)", fontsize="20")
plt.ylabel("Energy Gap (eV)", fontsize="20")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 600)
plt.grid()

plt.tight_layout()
plt.tight_layout()
ax = plt.gca()  
for spine in ax.spines.values():
    spine.set_linewidth(1.5)  
ax.tick_params(width=1.5, length=8) 
fname = 'FigureS9.tiff'
plt.savefig(fname, format="tiff", dpi=300)
