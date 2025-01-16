#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:16:26 2024

@author: zhaoyansun
"""
import h5py
import numpy as np
import math

def patch_sum(nptch):
    #patches = len(nptch)
    #tenth_patch = int(math.ceil(0.1*patches))
    #particle_index = -1
    nptch_sum = []
    i=0
    while i in range(len(nptch)):
        if i+9<len(nptch):
            sum = np.sum(nptch[i:i+10])
            nptch_sum.append(sum)
        else:
            sum = np.sum(nptch[i:])
            nptch_sum.append(sum)
        i = i+10
    nptch_sum = np.array(nptch_sum)
    return nptch_sum

def index_count(nptch_sum):
    inx_start = np.zeros(len(nptch_sum))
    inx_end = np.zeros(len(nptch_sum))
    for i in range(len(nptch_sum)):
        inx_start[i] = np.sum(nptch_sum[0:i])
        inx_end[i] = inx_start[i]+nptch_sum[i]-1
    inx_start = inx_start.astype(int)
    inx_end = inx_end.astype(int)
    return (inx_start, inx_end)

def decimate_npatch(nptch):
    nptch_sum = patch_sum(nptch)-+
    start, end = index_count(nptch_sum)
    deci_patch = np.zeros(len(start))
    for i in range(len(start)):
        deci_patch[i] = math.floor(0.1*end[i])-math.floor(0.1*start[i])
        if start[i]%10 == 0:
            deci_patch[i] = deci_patch[i]+1
    deci_patch = deci_patch.astype(int)
    return deci_patch

filename = "prts_1400000.h5"

with h5py.File(filename, "r") as hf3:
    npatch = hf3['n_patch'][:]
    kind = hf3['kind'][()]

kind = kind[::10]
deci_npatch = decimate_npatch(npatch[8500:])
"""
filename2 = "smaller_prts.h5"

with h5py.File(filename2, "r") as hf3:
    kind = hf3['kind'][()]
"""
    
print(len(kind))
print(np.sum(deci_npatch))