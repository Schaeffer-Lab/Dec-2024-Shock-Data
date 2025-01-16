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
nn = 32 * 32
npc = 2000
#################

filename = "smaller_fields.h5"

# z, x, y are coordinates
# E for electric field
# B for magnetic field
# j for current density

with h5py.File(filename, "r") as hf:
    z = hf['z'][()]

# rho = number density
# i = ion
# e = electron 
# tar = piston plasma
# am = ambient plasma
# Tss = momentum-velocity tensor
# j = current density
# p = momentum density

filename2 = "smaller_moments.h5"

with h5py.File(filename2, "r") as hf2:
    jz = hf2['jz_i_tar'][()]
    rho = hf2['rho_i_tar'][()]
    px = hf2['px_e_tar'][()]

print(px.shape)
print(rho.shape)
fig, ax = plt.subplots()

mean_jz = np.mean(jz, axis=1)
mean_rho = np.mean(rho, axis=1)

ax.plot(z, mean_jz, label='Mean Jz from smaller_moments.h5', color='green')
# ax.plot(z, mean_rho, label='Mean Rho from moments_1400000.h5') # Uncomment if you want to plot mean_rho

plt.xlim(0, 81600)

filename3 = "smaller_prts.h5"

with h5py.File(filename3, "r") as hf3:
    xb = hf3['xb'][()]
    xe = hf3['xe'][()]
    z2 = (xb + xe) / 2
    zz = hf3['z'][()]
    npatch = hf3['n_patch'][:]  # Read npatch dataset
    
    z2 = z2[850:1700, 2]  # Ensure z2 does not exceed the length of npatch
    
    jz = np.zeros(len(z2))
    den = np.zeros(len(z2))
    uz = hf3['uz'][()]
    kind = hf3['kind'][()]

    p1 = 0
    for i in range(850):
        p2 = p1 + npatch[850+i]  # Access npatch directly using the index i
        uz_z = uz[p1:p2, 0]
        kind_z = kind[p1:p2, 0]
        ls = np.where(kind_z == 3)[0]
        den[i] = len(ls)
        if len(ls) > 0:
            jz[i] = np.sum(uz_z[ls])
        else:
            jz[i] = 0
        p1 = p2
print((xb[1,0]-xe[1,0])/2)
print(zz.shape)
jz = jz / (npc * nn)
den = den / (npc * nn)
ax.plot(z2, jz, label='Jz from smaller_prts.h5', color='red')
# ax.plot(z2, den, label='Density from prts_1400000.h5') # Uncomment if you want to plot density

# Add labels and title
ax.set_xlabel('Z coordinate')
ax.set_ylabel('Current Density (Jz)')
ax.set_title('Current Density Plot')
ax.legend()

# Show the plot
plt.show()
