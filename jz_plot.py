#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:00:00 2024

@author: Yan
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np

# You need these parameters for conversion. Do not change
nn = 32 * 32  # Number of points in each direction
npc = 2000    # Number of particles per cell
#################

# Load data from fields_1400000.h5
filename = "fields_1400000.h5"

# z, x, y are coordinates
# E for electric field
# B for magnetic field
# j for current density
with h5py.File(filename, "r") as f:
    hf = h5py.File(filename, 'r')
    z = hf.get('z')[()]

# Load data from moments_1400000.h5
# rho = number density
# i = ion
# e = electron 
# tar = piston plasma
# am = ambient plasma
# Tss = momentum-velocity tensor
# j = current density
# p = momentum density
filename2 = "moments_1400000.h5"
with h5py.File(filename2, "r") as f:
    hf2 = h5py.File(filename2, 'r')
    jz = hf2.get('jz_i_tar')[()]
    rho = hf2.get('rho_i_tar')[()]

fig, ax = plt.subplots()

# Plot mean Jz from moments_1400000.h5
mean_jz = np.mean(jz, 1)
mean_rho = np.mean(rho, 1)
ax.plot(z, mean_jz, label='Mean Jz from moments_1400000.h5', color='blue')

# Set x-axis limit
plt.xlim(0, 81600)

# Load data from prts_1400000.h5
filename3 = "prts_1400000.h5"
with h5py.File(filename3, "r") as f:
    hf3 = h5py.File(filename3, 'r')
    xb = hf3.get('xb')[()]
    xe = hf3.get('xe')[()]
    z2 = (xb + xe) / 2
    z2 = z2[8500:17000, 2]  # Selecting a specific range and column
    
    jz = np.zeros(len(z2))
    den = np.zeros(len(z2))
    uz = hf3.get('uz')[()]
    kind = hf3.get('kind')[()]
    npatch = hf3.get('n_patch')[()]

    p1 = 0
    for i in range(8500):
        p2 = p1 + npatch[8500 + i]
        uz_z = uz[p1:p2, 0]
        kind_z = kind[p1:p2, 0]
        ls = np.where(kind_z == 3)[0]
        den[i] = len(ls)
        if len(ls) > 0:
            jz[i] = np.sum(uz_z[ls])
        else:
            jz[i] = 0
        p1 = p2

# Normalize values by the number of particles and cells
jz = jz / (npc * nn)
den = den / (npc * nn)

# Plot Jz from prts_1400000.h5
ax.plot(z2, jz, label='Jz from prts_1400000.h5', color='orange')

# Add labels and title
ax.set_xlabel('Z coordinate')
ax.set_ylabel('Current Density (Jz)')
ax.set_title('Current Density Plot')

# Add legend
ax.legend()

# Show the plot
plt.show()
