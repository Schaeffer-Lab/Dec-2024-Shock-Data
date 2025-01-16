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
    ex = hf['ex'][()]
    ey = hf['ey'][()]
    ez = hf['ez'][()]

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

print(ex.shape)

filename2 = "smaller_moments.h5"

with h5py.File(filename2, "r") as hf2:
    rho = hf2['rho_i_tar'][()]

fig, ax = plt.subplots()

mean_ex = np.mean(ex, axis=1)
mean_ey = np.mean(ey, axis=1)
mean_ez = np.mean(ez, axis=1)




ax.plot(z, mean_ex, label='mean_Ex from smaller_fields.h5', color='red')
ax.plot(z, mean_ey, label='mean_Ey from smaller_fields.h5', color='yellow')
ax.plot(z, mean_ez, label='mean_Ez from smaller_fields.h5', color='green')
# ax.plot(z, mean_rho, label='Mean Rho from moments_1400000.h5') # Uncomment if you want to plot mean_rho


plt.xlim(0, 81600*0.3/10/8)


# Add labels and title
ax.set_xlabel('Z Coordinate (Upstream Ion Inertial Lengths)')
ax.set_ylabel('E Strength, (cB0)')
ax.set_title('E Plot')
ax.legend()

# Show the plot
plt.show()

# units for axis, Z in units of electron inertial length, 
# c/omegaPE, convert to di = c/wpi, frequencies 
# NRL plasma forumlary, page 28, is a list of fundamental plasma parameters, ion inertial lengths, 
# how to calculate it. 
# de/di = 1/sqrt(m_i/m_e) , mass ratio is usually 100, = sqrt(me/mi) = 1/10
# 
# ask Peera about the mass ratio, 64 or 100. 
# shock itself is order of several inertial lengths.
# 10-100 ion inertial lengths.

# ask peera what the units for electric field. In units of upstream ion inertial lengths, 
# Ask peera whats the conversion factor to go from z coordinate to upstream ion inertial lengths
# di,u = 10-30 di,t, want upstream, several hunred or several thousand

# zoom into the plot 

fig, ax2 = plt.subplots()


ax2.plot(z, mean_ex, label='mean_Ex from smaller_fields.h5', color='red')
ax2.plot(z, mean_ey, label='mean_Ey from smaller_fields.h5', color='yellow')
ax2.plot(z, mean_ez, label='mean_Ez from smaller_fields.h5', color='green')
# ax.plot(z, mean_rho, label='Mean Rho from moments_1400000.h5') # Uncomment if you want to plot mean_rho

plt.xlim(20000*0.3/10/8, 31000*0.3/10/8)


# Add labels and title
ax2.set_xlabel('Z Coordinate (Upstream Ion Inertial Lengths)')
ax2.set_ylabel('E Strength, (cB0)')
ax2.set_title('E Plot Shock')
ax2.legend()


#gyro radius, downstream or upstream, compare it to shock ion gyroradius, 4x, scale lengths,
#gyro radius in simulation units, 

# new change 