#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:16:26 2024

@author: zhaoyansun
"""
import h5py

#moments
# Specify the filenames
input_filename = "moments_1400000.h5"
output_filename = "smaller_moments.h5"

# Open the input HDF5 file
with h5py.File(input_filename, 'r') as hf:
    # Open the output HDF5 file
    with h5py.File(output_filename, 'w') as mhf:
        # Iterate over each dataset in the input file
        for dataset_name in hf.keys():
            # Read the entire dataset
            data = hf[dataset_name][:]
            # Select every 10th element
            processed_data = data[::10]
            # Create a new dataset in the output file with the processed data
            mhf.create_dataset(dataset_name, data=processed_data)

print("Processed and saved to 'smaller_moments.h5' successfully.")


#fields
# Specify the filenames
input_filename = "fields_1400000.h5"
output_filename = "smaller_fields.h5"

# Open the input HDF5 file
with h5py.File(input_filename, 'r') as hf:
    # Open the output HDF5 file
    with h5py.File(output_filename, 'w') as mhf:
        # Iterate over each dataset in the input file
        for dataset_name in hf.keys():
            # Read the entire dataset
            data = hf[dataset_name][:]
            # Select every 10th element
            processed_data = data[::10]
            # Create a new dataset in the output file with the processed data
            mhf.create_dataset(dataset_name, data=processed_data)

print("Processed and saved to 'smaller_fields.h5' successfully.")

#prts
# Specify the filenames
input_filename = "prts_1400000.h5"
output_filename = "smaller_prts.h5"

# Open the input HDF5 file
with h5py.File(input_filename, 'r') as hf:
    # Open the output HDF5 file
    with h5py.File(output_filename, 'w') as mhf:
        # Iterate over each dataset in the input file
        for dataset_name in hf.keys():
            # Read the entire dataset
            data = hf[dataset_name][:]
            # Select every 10th element
            processed_data = data[::10]
            # Create a new dataset in the output file with the processed data
            mhf.create_dataset(dataset_name, data=processed_data)

print("Processed and saved to 'smaller_prts.h5' successfully.")
