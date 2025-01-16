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

#z coordinate conversion
#each grid point is 0.3 de0, upstream density is 0.01, mi/me =64
z = z*0.3/10/8

filename2 = "smaller_moments.h5"

with h5py.File(filename2, "r") as hf2:
    rho_i_tar = hf2['rho_i_tar'][()]
    rho_i_am = hf2['rho_i_am'][()]
    rho_e_tar = -1*hf2['rho_e_tar'][()]
    rho_e_am = -1*hf2['rho_e_am'][()]

fig, ax = plt.subplots()

mean_rho_i_tar = np.mean(rho_i_tar, axis=1)
mean_rho_i_am = np.mean(rho_i_am, axis=1)
mean_rho_e_tar = np.mean(rho_e_tar, axis=1)
mean_rho_e_am = np.mean(rho_e_am, axis=1)

ax.plot(z, mean_rho_i_tar, color='green', label='mean_rho_i_tar')
ax.plot(z, mean_rho_i_am, color='orange', label='mean_rho_i_am')
ax.plot(z, mean_rho_e_tar, color='purple', label='mean_rho_e_tar')
ax.plot(z, mean_rho_e_am, color='blue', label='mean_rho_e_am')


plt.xlim(0, 81600*0.3/10/8)

# Add labels and titlev
ax.set_xlabel('Z Coordinate (Upstream Ion Inertial Lengths)')
ax.set_ylabel('Density (n0)')
ax.set_title('Ion and Electron Density Plot')
ax.legend()

# Show the plot
plt.show()



# 2nd to see if the e density matches i density
# what target material did combination use Peera, if it was hydrogen, 
# add green and organ add, and add purple and blue line, to see if they overlap everywhere
# constant and overlap
# density units? not yet, units fraction of the target density, 
# Z coordinate

fig, ax2 = plt.subplots()

sum_rho_i = mean_rho_i_am + mean_rho_i_tar
sum_rho_e = mean_rho_e_am + mean_rho_e_tar
ax2.plot(z, sum_rho_i, color ='red', label='sum_rho_i')
ax2.plot(z, sum_rho_e, color ='black', label='sum_rho_e')
ax2.set_xlabel('Z Coordinate (Upstream Ion Inertial Lengths)')
ax2.set_ylabel('Density (n0)')
ax2.set_title('Sum of Ion and Electron Density Plot')
ax2.legend()

#3rd plot difference
fig, ax3 = plt.subplots()
dif_rho_i_and_e =sum_rho_i-sum_rho_e
ax3.plot(z, dif_rho_i_and_e)
ax3.set_xlabel('Z Coordinate (Upstream Ion Inertial Lengths)')
ax3.set_ylabel('Density (n0)')
ax3.set_title('Difference Between Sum of Ion and Electron Density Plot')




