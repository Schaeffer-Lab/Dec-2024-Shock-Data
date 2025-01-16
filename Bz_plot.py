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
    bx = hf['bx'][()]
    by = hf['by'][()]
    bz = hf['bz'][()]

# rho = number density
# i = ion
# e = electron 
# tar = piston plasma
# am = ambient plasma
# Tss = momentum-velocity tensor
# j = current density
# p = momentum density

#z coordinate conversion
#each grid point is 0.3 de0, upstream density is 0.01, mi/me =64
z = z*0.3/10/8

filename2 = "smaller_moments.h5"

with h5py.File(filename2, "r") as hf2:
    rho = hf2['rho_i_tar'][()]

fig, ax = plt.subplots()

mean_bx = np.mean(bx, axis=1)
mean_by = np.mean(by, axis=1)
mean_bz = np.mean(bz, axis=1)


ax.plot(z, mean_bx, label='mean_Bx from smaller_fields.h5', color='red')
ax.plot(z, mean_by, label='mean_By from smaller_fields.h5', color='yellow')
ax.plot(z, mean_bz, label='mean_Bz from smaller_fields.h5', color='green')
# ax.plot(z, mean_rho, label='Mean Rho from moments_1400000.h5') # Uncomment if you want to plot mean_rho

plt.xlim(0, 81600*0.3/10/8)

# Add labels and title
ax.set_xlabel('Z Coordinate (Upstream Ion Inertial Lengths)')
ax.set_ylabel('B Strength, (B0)')
ax.set_title('B Plot')
ax.legend()

# Show the plot
plt.show()

# plot during shock
fig, ax2 = plt.subplots()
ax2.plot(z, mean_bx, label='mean_Bx from smaller_fields.h5', color='red')
ax2.plot(z, mean_by, label='mean_By from smaller_fields.h5', color='yellow')
ax2.plot(z, mean_bz, label='mean_Bz from smaller_fields.h5', color='green')
# ax.plot(z, mean_rho, label='Mean Rho from moments_1400000.h5') # Uncomment if you want to plot mean_rho

plt.xlim(15000*0.3/10/8, 32500*0.3/10/8)

# Add labels and title
ax2.set_xlabel('Z Coordinate (Upstream Ion Inertial Lengths)')
ax2.set_ylabel('B Strength, (B0)')
ax2.set_title('B Plot shock')
ax2.legend()

plt.show()

#I changed something 