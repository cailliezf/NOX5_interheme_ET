#!/usr/bin/env python3

#%% Lipid distances of upper and lower leaflets with the outer and inner heme respectively along the different trajectories. 

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance_matrix
from matplotlib.widgets import Slider, Button
from MDAnalysis.analysis import distances

#%% Loading of trajectory and psf files for one MD simulation

u1 = mda.Universe("nox5st1.psf", "csNOX5_mbW_st1.dcd")

#%% Define the atomes selections

# Take atomes corresponding to the lipids heads 
POPE_heads_upper_u1 = u1.select_atoms('resname POPE and resid 1 to 210 and name P')
POPG_heads_upper_u1 = u1.select_atoms('resname POPG and resid 1 to 210 and name P')
POPE_heads_lower_u1 = u1.select_atoms('resname POPE and resid 211 to 420 and name P')
POPG_heads_lower_u1 = u1.select_atoms('resname POPG and resid 211 to 420 and name P')
HEME_upper_u1 = u1.select_atoms('resname HEME and resid 2 and name FE')
HEME_lower_u1 = u1.select_atoms('resname HEME and resid 1 and name FE')

#%% Treatment of trajectory

distances2D_data_u1 = {
    'POPE_upper': [],
    'POPG_upper': [],
    'POPE_lower': [],
    'POPG_lower': [],
}

for ts in u1.trajectory:
    
    # Project all lipids heads coordinate onto the XY plane
    POPE_heads_upper_xy_u1 = POPE_heads_upper_u1.positions.copy()
    POPE_heads_upper_xy_u1[:, 2] = 0  
    POPG_heads_upper_xy_u1 = POPG_heads_upper_u1.positions.copy()
    POPG_heads_upper_xy_u1[:, 2] = 0  
    POPE_heads_lower_xy_u1 = POPE_heads_lower_u1.positions.copy()
    POPE_heads_lower_xy_u1[:, 2] = 0  
    POPG_heads_lower_xy_u1 = POPG_heads_lower_u1.positions.copy()
    POPG_heads_lower_xy_u1[:, 2] = 0  
    
    HEME_upper_xy_u1 = HEME_upper_u1.positions.copy()
    HEME_upper_xy_u1[:, 2] = 0  
    HEME_lower_xy_u1 = HEME_lower_u1.positions.copy()
    HEME_lower_xy_u1[:, 2] = 0  
    
    # Compute the XY-plane distance between each lipid and each Heme group
    POPE_upper_distances_u1 = distances.distance_array(POPE_heads_upper_xy_u1, HEME_upper_xy_u1, box=ts.dimensions)
    POPG_upper_distances_u1 = distances.distance_array(POPG_heads_upper_xy_u1, HEME_upper_xy_u1, box=ts.dimensions)
    POPE_lower_distances_u1 = distances.distance_array(POPE_heads_lower_xy_u1, HEME_lower_xy_u1, box=ts.dimensions)
    POPG_lower_distances_u1 = distances.distance_array(POPG_heads_lower_xy_u1, HEME_lower_xy_u1, box=ts.dimensions)
        
    # Storage of the distances data for each frame
    distances2D_data_u1['POPE_upper'].append(POPE_upper_distances_u1)
    distances2D_data_u1['POPG_upper'].append(POPG_upper_distances_u1)
    distances2D_data_u1['POPE_lower'].append(POPE_lower_distances_u1)
    distances2D_data_u1['POPG_lower'].append(POPG_lower_distances_u1)
    
            

with open('lipid_distances2D_csNOX5_mbW_st1.pkl', 'wb') as f:
    pickle.dump(distances2D_data_u1, f)



