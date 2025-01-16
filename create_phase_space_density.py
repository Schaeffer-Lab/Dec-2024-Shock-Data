#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:00:00 2024

@author: Rayner
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np

def createphasespace(nptch,uarray,zarray,bins,urange,species=3,kind_z=[3],speedfactor=500):
  pspace = np.zeros((len(zarray),bins)) #initialize phase space starting from the lowest velocity bin
  ustep = 2*urange/(bins-1) #velocity space bin width
  ntotal = 0 #tracking total number of particles
  #intarray = np.round((1/ustep)*uarray).astype(int)
  for i in range(len(nptch)):
    j = ntotal
    while j in range(ntotal,ntotal+nptch[i]):
      if kind_z[j,0] == species:
        particlespeed = uarray[j] #this is to find the speed of the particle
        itgr = (particlespeed+urange)/ustep #find how many steps away from the lowest bin to find which bin the particle lies in
        itgr = np.round(itgr).astype(int)
        if 0<=itgr<bins:
          pspace[i,itgr] = pspace[i,itgr]+1
      j = j+speedfactor
    print("npatch index:", i, end='\r')
    ntotal = ntotal+nptch[i]
  return pspace