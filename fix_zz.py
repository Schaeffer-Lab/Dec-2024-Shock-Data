#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:16:26 2024

@author: zhaoyansun
"""
import h5py
import numpy as np
import math
import time

filename = "prts_1400000.h5"

with h5py.File(filename, "r") as hf3:
    xb = hf3['xb'][()]
    npatch = hf3['n_patch'][:]
    zz = hf3['z'][()]

zz = zz[::10]

def index_count(nptch):
    inx_start = np.zeros(len(nptch))
    inx_end = np.zeros(len(nptch))
    for i in range(len(nptch)):
        inx_start[i] = np.sum(nptch[0:i])
        inx_end[i] = inx_start[i]+nptch[i]-1
    inx_start = inx_start.astype(int)
    inx_end = inx_end.astype(int)
    return inx_start, inx_end

def decimate_npatch(nptch):
    start, end = index_count(nptch)
    deci_patch = np.zeros(len(start))
    for i in range(len(start)):
        deci_patch[i] = math.floor(0.1*end[i])-math.floor(0.1*start[i])
        if start[i]%10 == 0:
            deci_patch[i] = deci_patch[i]+1
    deci_patch = deci_patch.astype(int)
    return deci_patch

deci_npatch = decimate_npatch(npatch[8500:])

def fix_zz(nptch,zz_input,xb_input):
    zz_fixed=[]
    p1 = 0
    for i in range(8500):
        delta = nptch[8500+i] # delta is the number of particles in a patch.
        p2 = p1 + delta  # Here we keep track of the total number of particles, the nth patch lies between the start of the patch p1 and p1+delta
        zz_patch = zz_input[p1:p2, 0].tolist() # we extract only the particles in the patch and put them in a list
        zz_patch_fix = [x+xb_input[8500+i,2] for x in zz_patch] # we then add the position of the patch xb to the relative position of the particles zz to get the absolut position of the particles in this patch
        zz_fixed.extend(zz_patch_fix) # we put the absolute positions of each particle into the new list
        print("npatch index:", i, end='\r')
        p1 = p2
    z_fixed = np.array(zz_fixed)
    return z_fixed
zz_fixed = fix_zz(npatch,zz,xb)

"""
def fix_zz(nptch,zz_input,xb_input):
    zz_fixed=[]
    p1 = 0
    for i in range(8500):
        delta = nptch[8500+i] # delta is the number of particles in a patch.
        p2 = p1 + delta  # Here we keep track of the total number of particles, the nth patch lies between the start of the patch p1 and p1+delta
        zz_patch = zz_input[p1:p2, 0].tolist() # we extract only the particles in the patch and put them in a list
        zz_patch_fix = [x+xb_input[8500+i,2] for x in zz_patch] # we then add the position of the patch xb to the relative position of the particles zz to get the absolut position of the particles in this patch
        zz_fixed.extend(zz_patch_fix) # we put the absolute positions of each particle into the new list
        deci_patch = math.floor(0.1*p2)-math.floor(0.1*p1)
        if p1%10 == 0:
            deci_patch = deci_patch+1

        print("npatch index:", i, end='\r') # this tracks which patch the program is on
        p1 = p2 #to get to the next patch, we set the start of the next patch to be the end of the previous patch
    z_fixed = np.array(zz_fixed) # we change the list back to an array
    return z_fixed

def fix_zz(nptch,zz_input,xb_input):
    zz_deci = zz_input[::10]
    zz_fixed=[]
    xb_deci = []
    p1 = 0
    for i in range(8500):
        delta = nptch[8500+i] # delta is the number of particles in a patch.
        p2 = p1 + delta  # Here we keep track of the total number of particles, the nth patch lies between the start of the patch p1 and p1+delta
        #p1 is the start of the 1st patch and p2 is the start of the second patch
        start = 10*int(math.ceil(0.1*p1))
        end = 10*int(math.floor(0.1*p2))
        z_deci = zz_input[start:end+1].tolist()
        z_deci = z_deci[::10]


        zz_patch_fix = [x+xb_input[8500+i,2] for x in zz_patch] # we then add the position of the patch xb to the relative position of the particles zz to get the absolut position of the particles in this patch
        zz_fixed.extend(zz_patch_fix) # we put the absolute positions of each particle into the new list
        print("npatch index:", i, end='\r') # this tracks which patch the program is on
        p1 = p2 #to get to the next patch, we set the start of the next patch to be the end of the previous patch
    z_fixed = np.array(zz_fixed) # we change the list back to an array
    return z_fixed

"""
zz_fixed = fix_zz(npatch,zz,xb)

#print(np.shape(zz[::10]))

print(np.shape(zz_fixed))