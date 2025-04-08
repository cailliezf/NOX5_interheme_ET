#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Loading libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az
import pickle

#%% Conversion factors
ev2ha=0.0367502
kcalmol2ua=0.00159362
kcalmol2ev=0.0433634
j2eV=6.24181e18
bohr=5.2917721092*1E-11

#Setting temperature
temp=298.

# Constants
kb=1.3806488e-23
hbar=1.054571726E-34
kbTeV=kb*temp*j2eV
kbTha=kbTeV*ev2ha


#%% Parameters to be adapted
# Files containing the energies
file_st1_e1="nrj_st1_e1_41_46.dat"   # Files containing the energies of state 1 for MD performed on state 1
file_st1_e2="nrj_st1_e2_41_46.dat"   # Files containing the energies of state 2 for MD performed on state 1
file_st2_e1="nrj_st2_e1_41_46.dat"   # Files containing the energies of state 1 for MD performed on state 2
file_st2_e2="nrj_st2_e2_41_46.dat"   # Files containing the energies of state 2 for MD performed on state 2
# Contribution to be computed
contr="Heme"
# Starting of production phase
xmin=200                           # begining of production phase
xmax2=900
# Number of blocks for block average uncertainty calculation
nblocks=5                           # Number of blocks for block average
# Marcus parabolas
binsize=0.05                        # Bin size (in eV) for histograms of energy gaps
# Showplots
showplot=False
#%% Read Energy and compute energy gaps

energy_st1_e1 = pd.read_csv(file_st1_e1,sep='\s+',header=None,
                            names=['Index','All','Heme1+Heme2','Heme1','Heme2','FAD','POPEup','POPGup','POPElo',
                                   'POPGlo','TM','DH','Na','Cl','Wat','Prop'])
energy_st1_e2 = pd.read_csv(file_st1_e2,sep='\s+',header=None,
                            names=['Index','All','Heme1+Heme2','Heme1','Heme2','FAD','POPEup','POPGup','POPElo',
                                   'POPGlo','TM','DH','Na','Cl','Wat','Prop'])
energy_st2_e1 = pd.read_csv(file_st2_e1,sep='\s+',header=None,
                            names=['Index','All','Heme1+Heme2','Heme1','Heme2','FAD','POPEup','POPGup','POPElo',
                                   'POPGlo','TM','DH','Na','Cl','Wat','Prop'])
energy_st2_e2 = pd.read_csv(file_st2_e2,sep='\s+',header=None,
                            names=['Index','All','Heme1+Heme2','Heme1','Heme2','FAD','POPEup','POPGup','POPElo',
                                   'POPGlo','TM','DH','Na','Cl','Wat','Prop'])


#%% Decomposition + Uncertainty 

energy_st1_e1['Memb'] = energy_st1_e1['POPEup'] + energy_st1_e1['POPGup'] + energy_st1_e1['POPElo'] + energy_st1_e1['POPGlo'] - 3*energy_st1_e1['Heme1+Heme2']
energy_st1_e2['Memb'] = energy_st1_e2['POPEup'] + energy_st1_e2['POPGup'] + energy_st1_e2['POPElo'] + energy_st1_e2['POPGlo'] - 3*energy_st1_e2['Heme1+Heme2']
energy_st2_e1['Memb'] = energy_st2_e1['POPEup'] + energy_st2_e1['POPGup'] + energy_st2_e1['POPElo'] + energy_st2_e1['POPGlo'] - 3*energy_st2_e1['Heme1+Heme2']
energy_st2_e2['Memb'] = energy_st2_e2['POPEup'] + energy_st2_e2['POPGup'] + energy_st2_e2['POPElo'] + energy_st2_e2['POPGlo'] - 3*energy_st2_e2['Heme1+Heme2']

energy_st1_e1['Prot'] = energy_st1_e1['TM'] + energy_st1_e1['DH'] - energy_st1_e1['Heme1+Heme2']
energy_st1_e2['Prot'] = energy_st1_e2['TM'] + energy_st1_e2['DH'] - energy_st1_e2['Heme1+Heme2']
energy_st2_e1['Prot'] = energy_st2_e1['TM'] + energy_st2_e1['DH'] - energy_st2_e1['Heme1+Heme2']
energy_st2_e2['Prot'] = energy_st2_e2['TM'] + energy_st2_e2['DH'] - energy_st2_e2['Heme1+Heme2']

energy_st1_e1['Env'] = energy_st1_e1['Na'] + energy_st1_e1['Cl'] + energy_st1_e1['Wat'] - 2*energy_st1_e1['Heme1+Heme2']
energy_st1_e2['Env'] = energy_st1_e2['Na'] + energy_st1_e2['Cl'] + energy_st1_e2['Wat'] - 2*energy_st1_e2['Heme1+Heme2']
energy_st2_e1['Env'] = energy_st2_e1['Na'] + energy_st2_e1['Cl'] + energy_st2_e1['Wat'] - 2*energy_st2_e1['Heme1+Heme2']
energy_st2_e2['Env'] = energy_st2_e2['Na'] + energy_st2_e2['Cl'] + energy_st2_e2['Wat'] - 2*energy_st2_e2['Heme1+Heme2']

energy_st1_e1['Heme'] = energy_st1_e1['Heme1+Heme2'] + energy_st1_e1['Prop'] - energy_st1_e1['Heme1'] - energy_st1_e1['Heme2']
energy_st1_e2['Heme'] = energy_st1_e2['Heme1+Heme2'] + energy_st1_e2['Prop'] - energy_st1_e2['Heme1'] - energy_st1_e2['Heme2']
energy_st2_e1['Heme'] = energy_st2_e1['Heme1+Heme2'] + energy_st2_e1['Prop'] - energy_st2_e1['Heme1'] - energy_st2_e1['Heme2']
energy_st2_e2['Heme'] = energy_st2_e2['Heme1+Heme2'] + energy_st2_e2['Prop'] - energy_st2_e2['Heme1'] - energy_st2_e2['Heme2']
#%% Sanity check
# Sum of deltaE for all cntributions should be equal to deltaE of the whole system
#
deltae1=(energy_st1_e1-energy_st1_e2).to_numpy()
deltae2=(energy_st2_e1-energy_st2_e2).to_numpy()
dd1=np.zeros(len(deltae1))
dd2=np.zeros(len(deltae2))
for i in range(len(deltae1)):
    dd1[i]=((np.sum(deltae1[i,5:])-10*deltae1[i,2])-deltae1[i,1]) # *100/deltae1[i,1]
    dd2[i]=((np.sum(deltae2[i,5:])-10*deltae2[i,2])-deltae2[i,1]) # *100/deltae2[i,1]

if (showplot):
    plt.figure()
    plt.plot(range(len(deltae1)),dd1,label="State 1")
    plt.plot(range(len(deltae2)),dd2,label="State 2")

#%%
interH_st1_e2 = (energy_st1_e2['Heme1+Heme2']-energy_st1_e2['Heme2']-energy_st1_e2['Heme1'])  
interH_st1_e1 = (energy_st1_e1['Heme1+Heme2']-energy_st1_e1['Heme2']-energy_st1_e1['Heme1'])
interH_st2_e2 = (energy_st2_e2['Heme1+Heme2']-energy_st2_e2['Heme2']-energy_st2_e2['Heme1'])
interH_st2_e1 = (energy_st2_e1['Heme1+Heme2']-energy_st2_e1['Heme2']-energy_st2_e1['Heme1'])
# Gaps in kcal/mol
# 1-2 lines for All and InterH contrib    3-4 for others
#gap_1   = (energy_st1_e2[contr]-energy_st1_e2['Heme2']-energy_st1_e2['Heme1']) - (energy_st1_e1[contr]-energy_st1_e1['Heme2']-energy_st1_e1['Heme1'])
#gap_2   = (energy_st2_e2[contr]-energy_st2_e2['Heme2']-energy_st2_e2['Heme1']) - (energy_st2_e1[contr]-energy_st2_e1['Heme2']-energy_st2_e1['Heme1'])
gap_1   = (energy_st1_e2[contr]-energy_st1_e2['Heme2']-energy_st1_e2['Heme1']-interH_st1_e2) - (energy_st1_e1[contr]-energy_st1_e1['Heme2']-energy_st1_e1['Heme1']-interH_st1_e1)
gap_2   = (energy_st2_e2[contr]-energy_st2_e2['Heme2']-energy_st2_e2['Heme1']-interH_st2_e2) - (energy_st2_e1[contr]-energy_st2_e1['Heme2']-energy_st2_e1['Heme1']-interH_st2_e1)

# Conversion eV 
gap_1 = gap_1*kcalmol2ev
gap_2 = gap_2*kcalmol2ev

# Gather in data frame
dfgaps  = pd.concat((gap_1.rename('gap1'),gap_2.rename('gap2')),axis=1)

#%% Plot energy gaps with respect to time for control
xmax=len(dfgaps.index) ; x=np.arange(xmax)/10
nrol=int(xmax/10)
gap1 = dfgaps['gap1'] ; gap1roll = gap1.rolling(window=nrol,center=True).mean()
gap2 = dfgaps['gap2'] ; gap2roll = gap2.rolling(window=nrol,center=True).mean()
if (showplot):
    plt.figure()
    plt.plot(x,gap1,'r-',label="Gap State 1")
    plt.plot(x,gap1roll,'y-',linewidth=2)
    plt.plot(x,gap2,'b-',label="Gap State 2")
    plt.plot(x,gap2roll,'g-',linewidth=2)
    plt.xlabel("Time (ns)") ; plt.ylabel("Energy gaps (eV)")
    plt.legend() ; plt.grid()

#%% Compute Marcus parameters
gape1=gap1.drop(np.arange(xmin))   # drop equilibration phase
gape1=gape1.drop(np.arange(xmax2,xmax))   # drop equilibration phase
gape2=gap2.drop(np.arange(xmin))   # drop equilibration phase
gape2=gape2.drop(np.arange(xmax2,xmax))   # drop equilibration phase
dfgapse=dfgaps.drop(np.arange(xmin))   # drop equilibration phase
dfgapse=dfgapse.drop(np.arange(xmax2,xmax))   # drop equilibration phase

dg    = 0.5*(gape1.mean(axis=0)+gape2.mean(axis=0))
u_dg  = np.sqrt(0.25*((gape1.var(axis=0))/az.ess(gape1.to_numpy()) + gape2.var(axis=0)/az.ess(gape2.to_numpy())))

lst   = 0.5*(gape1.mean(axis=0)+gape2.mean(axis=0))
u_lst = u_dg

lv1   = gape1.var(axis=0)/(2*kbTeV)
u_lv1 = np.sqrt((gape1.var(axis=0))**2/(az.ess(gape1.to_numpy())*kbTeV**2))

lv2   = gape2.var(axis=0)/(2*kbTeV)
u_lv2 = np.sqrt((gape2.var(axis=0))**2/(az.ess(gape2.to_numpy())*kbTeV**2))


#%% Compute block averages
sb = int(len(gape1)/nblocks)  # size of blocks
dgb = np.zeros(nblocks) ; lstb = np.zeros(nblocks) ; lv1b = np.zeros(nblocks) ; lv2b = np.zeros(nblocks)

for i in range(nblocks):
  # Definition of the block
  xi=i*sb ;  xf=xi+sb
  if (i == nblocks):
      xf=len(gape1)
  dgb[i]  = 0.5*(np.mean(gape1.iloc[xi:xf]) + np.mean(gape2.iloc[xi:xf]))
  lstb[i] = 0.5*(np.mean(gape1.iloc[xi:xf]) - np.mean(gape2.iloc[xi:xf]))
  lv1b[i] = np.var(gape1.iloc[xi:xf],ddof=1)/(2*kbTeV)
  lv2b[i] = np.var(gape2.iloc[xi:xf],ddof=1)/(2*kbTeV)

dgavg = np.mean(dgb) ; lstavg = np.mean(lstb) ; lv1avg = np.mean(lv1b) ; lv2avg = np.mean(lv2b)
udgb  = np.std(dgb,ddof=1)/np.sqrt(nblocks)  ; ulstb = np.std(lstb,ddof=1)/np.sqrt(nblocks) 
ulv1b = np.std(lv1b,ddof=1)/np.sqrt(nblocks) ; ulv2b = np.std(lv2b,ddof=1)/np.sqrt(nblocks)

ndig=4
print("Statistics for Marcus parameters with block averaging using ",nblocks," blocks")
print("DeltaG (eV)")
print("<DG>   = ",round(dgavg,ndig),"    u(DG) = ",round(udgb,ndig))
print("Lambda Stokes (eV)")
print("<lst>  = ",round(lstavg,ndig),"    u(lst) = ",round(ulstb,ndig))
print("Lambda Variance state 1 and 2 (eV)")
print("<lst1> = ",round(lv1avg,ndig),"    u(lst1) = ",round(ulv1b,ndig))
print("<lst2> = ",round(lv2avg,ndig),"    u(lst2) = ",round(ulv2b,ndig))



#%% Plot Marcus parabolas
# First build the histograms of the vertical energy gap on the two PES
minh=np.min(dfgaps.min())-binsize
maxh=np.max(dfgaps.max())+binsize
exth=max(np.abs(minh),np.abs(maxh))
xplot = np.arange(-exth,exth,0.01)
sbins = np.arange(minh,maxh,binsize)
midbins = np.arange(minh+binsize/2,maxh-binsize/2,binsize)

# Make histograms for energy gaps
# on state 1
histo1 = np.histogram(gape1,bins=sbins) ; histo1d = np.histogram(gape1,bins=sbins,density=True)
filled1 = histo1[0] >= 2
x1 = midbins[filled1]
freq1 = histo1d[0][filled1]
# on state 2
histo2 = np.histogram(gape2,bins=sbins) ; histo2d = np.histogram(gape2,bins=sbins,density=True)
filled2 = histo2[0] >= 2
x2 = midbins[filled2]
freq2 = histo2d[0][filled2]


# Define the free energy curve, and assume ergodic hypothesis to build the other free energy surface
# From datat on state 1
gnum = np.concatenate((x1,-kbTeV*np.log(freq1),-kbTeV*np.log(freq1)+x1)).reshape((-1, 3), order='F')
gnum = pd.DataFrame(gnum,columns=["de","act1","act2"]) 
g1min=gnum['act1'].min()
gnum['act1']=gnum['act1']-g1min
gnum['act2']=gnum['act2']-g1min
# From data on state 2 and shift data with reaction free energy
gnum2 = np.concatenate((x2,-kbTeV*np.log(freq2)-x2,-kbTeV*np.log(freq2))).reshape((-1, 3), order='F')
gnum2 = pd.DataFrame(gnum2,columns=["de","act1","act2"]) 
g2min=gnum2['act2'].min()
gnum2['act1']=gnum2['act1']-g2min+dg
gnum2['act2']=gnum2['act2']-g2min+dg

# Plot histograms
if (showplot):
    plt.figure()
    n, bins, patches = plt.hist(x=gape1, bins=sbins, edgecolor='red',align='mid',fill=False)
    n2, bins2, patches2 = plt.hist(x=gape2, bins=sbins, edgecolor='blue',align='mid',fill=False)
    plt.xlabel('Energy gap (eV)')
    plt.ylabel('Counts')

# Fit computational poins with parabolas.
gnumall = pd.concat([gnum,gnum2]) ; weights=np.concatenate([histo1[0][filled1],histo2[0][filled2]])
# For state 1
fitg3 = np.polyfit(gnumall['de'],gnumall['act1'],deg=2,w=weights)
xmin1 = -fitg3[1]/(2.*fitg3[0])
para3 = np.polyval(fitg3,xplot)
# For state 2
fitg4 = np.polyfit(gnumall['de'],gnumall['act2'],deg=2,w=weights)
xmin2 = -fitg4[1]/(2.*fitg4[0])
para4 = np.polyval(fitg4,xplot)

# Plot the parabolas
if (showplot):
    plt.figure()
    plt.plot(-gnumall['de'],gnumall['act1'],'ro',linewidth=3)
    plt.plot(-gnumall['de'],gnumall['act2'],'bo',linewidth=3)
    plt.plot(-xplot,para3,'r-',linewidth=3)
    plt.plot(-xplot,para4,'b-',linewidth=3)
    plt.xlabel('Energy gaps (eV)')
    plt.ylabel('Free energy (eV)')
    
#%% 
# Keep the values of Free energy
try:
    with open(f'deltaG_hNOX5_mbW_D_{xmin}-{xmax2}.pkl', 'rb') as fichier:
        deltaG = pickle.load(fichier)
        deltaG_error = pickle.load(fichier)
except FileNotFoundError:
    deltaG = np.zeros((17,1))
    deltaG_error = np.zeros((17,1))
if contr == "All":
    deltaG[0, 0] += round(dgavg,ndig)
    deltaG_error[0, 0] += round(udgb,ndig)    
if contr == "Heme1+Heme2":
    deltaG[1, 0] += round(dgavg,ndig)
    deltaG_error[1, 0] += round(udgb,ndig)    
if contr == "FAD":
    deltaG[2, 0] += round(dgavg,ndig)
    deltaG_error[2, 0] += round(udgb,ndig)    
if contr == "POPEup":
    deltaG[3, 0] += round(dgavg,ndig)
    deltaG_error[3, 0] += round(udgb,ndig)    
if contr == "POPGup":
    deltaG[4, 0] += round(dgavg,ndig)
    deltaG_error[4, 0] += round(udgb,ndig)    
if contr == "POPElo":
    deltaG[5, 0] += round(dgavg,ndig)
    deltaG_error[5, 0] += round(udgb,ndig)    
if contr == "POPGlo":
    deltaG[6, 0] += round(dgavg,ndig)
    deltaG_error[6, 0] += round(udgb,ndig)    
if contr == "TM":
    deltaG[7, 0] += round(dgavg,ndig)
    deltaG_error[7, 0] += round(udgb,ndig)    
if contr == "DH":
    deltaG[8, 0] += round(dgavg,ndig)
    deltaG_error[8, 0] += round(udgb,ndig)    
if contr == "Na":
    deltaG[9, 0] += round(dgavg,ndig)
    deltaG_error[9, 0] += round(udgb,ndig)    
if contr == "Cl":
    deltaG[10, 0] += round(dgavg,ndig)
    deltaG_error[10, 0] += round(udgb,ndig)    
if contr == "Wat":
    deltaG[11, 0] += round(dgavg,ndig)
    deltaG_error[11, 0] += round(udgb,ndig)    
if contr == "Prop":
    deltaG[12, 0] += round(dgavg,ndig)
    deltaG_error[12, 0] += round(udgb,ndig)
if contr == "Memb":
    deltaG[13, 0] += round(dgavg,ndig)
    deltaG_error[13, 0] += round(udgb,ndig)
if contr == "Prot":
    deltaG[14, 0] += round(dgavg,ndig)
    deltaG_error[14, 0] += round(udgb,ndig)
if contr == "Env":
    deltaG[15, 0] += round(dgavg,ndig)
    deltaG_error[15, 0] += round(udgb,ndig)
if contr == "Heme":
    deltaG[16, 0] += round(dgavg,ndig)
    deltaG_error[16, 0] += round(udgb,ndig)


    
with open(f'deltaG_hNOX5_mbW_D_{xmin}-{xmax2}.pkl', 'wb') as fichier:
    pickle.dump(deltaG, fichier)
    pickle.dump(deltaG_error, fichier)
