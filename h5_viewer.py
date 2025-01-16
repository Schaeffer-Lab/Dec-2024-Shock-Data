#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:18:18 2024

@author: zhaoyansun
"""
import h5py

hf1 = h5py.File('smaller_fields.h5','r')
Datasetnames=hf1.keys()
print(Datasetnames)
print('smaller_fields data')
print()

hf2 = h5py.File('smaller_moments.h5','r')
Datasetnames=hf2.keys()
print(Datasetnames)
print('smaller_moments data')
print()

hf3 = h5py.File('smaller_prts.h5','r')
Datasetnames=hf3.keys()
print(Datasetnames)
print('smaller_prts data')
print()
