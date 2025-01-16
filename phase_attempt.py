#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:54:43 2024

@author: zhaoyansun
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

filename = "smaller_prts.h5"

with h5py.File(filename, "r") as hf:
    # Convert data to NumPy arrays and extract the first column
    x = np.array(hf['z'][::1000])[:, 0]
    y = np.array(hf['uz'][::1000])[:, 0]

# Verify types and shapes to debug
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)

# Check a few data points to verify content
print("First few elements of x:", x[:10])
print("First few elements of y:", y[:10])

x_bins = 1000  # Number of bins along the x-axis
y_bins = 100  # Number of bins along the y-axis

# Create 2D histogram
plt.hist2d(x, y, bins=[x_bins, y_bins], cmap='Blues')

# Add colorbar and labels
plt.colorbar(label='Counts')
plt.xlabel('z')
plt.ylabel('uz')
plt.title('2D Phase Diagram')

# Display the plot
plt.show()
