import h5py
import matplotlib.pyplot as plt
import numpy as np
import math

#################

filename_0 = "prts_1400000.h5"

with h5py.File(filename_0, "r") as hf3:
    xb = hf3['xb'][()]
    npatch = hf3['n_patch'][:]

filename_1 = "smaller_prts.h5"

with h5py.File(filename_1, "r") as hf3:
    uz = hf3['uz'][()]
    zz = hf3['z'][()]
    kind = hf3['kind'][()]

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
print("shape of new npatch array:", np.shape(deci_npatch))
print("sum of deci_npatch:", np.sum(deci_npatch))
print("number of particles in zz array:", len(zz))

def fix_zz(nptch,zz_input,xb_input):
    zz_fixed=[]
    p1 = 0
    for i in range(8500):
        delta = nptch[i] # delta is the number of particles in a patch.
        p2 = p1 + delta  # Here we keep track of the total number of particles, the nth patch lies between the start of the patch p1 and p1+delta
        zz_patch = zz_input[p1:p2, 0].tolist() # we extract only the particles in the patch and put them in a list
        zz_patch_fix = [x+xb_input[8500+i,2] for x in zz_patch] # we then add the position of the patch xb to the relative position of the particles zz to get the absolut position of the particles in this patch
        zz_fixed.extend(zz_patch_fix) # we put the absolute positions of each particle into the new list
        print("npatch index:", i, end='\r')
        p1 = p2
    z_fixed = np.array(zz_fixed)
    return z_fixed
zz_fixed = fix_zz(deci_npatch,zz,xb)

def phasespacehistogram(u_input,z_fixed,kind_input,species=2,BINS=1700,RANGE=[[-0.25, 0.25], [0, 81600]]):
    ls = np.where(kind_input == species)[0] #extract an array of one species from total array of particles
    uz_spec = u_input[ls] 
    zz_spec = z_fixed[ls]
    dummy1 = np.transpose(uz_spec)[0] #for some reason, uz seems to be a column instead of a row. This fixes that
    dummy2 = zz_spec
    histogram = np.histogram2d(dummy1, dummy2, BINS, RANGE)[0] #we input out species-specific data into the histogram
    return histogram

phasehistogram_0 = phasespacehistogram(uz,zz_fixed,kind,0,RANGE=[[-0.6, 0.6], [0, 81600]])
phasehistogram_1 = phasespacehistogram(uz,zz_fixed,kind,1,RANGE=[[-0.6, 0.6], [0, 81600]])
phasehistogram_2 = phasespacehistogram(uz,zz_fixed,kind,2)
phasehistogram_3 = phasespacehistogram(uz,zz_fixed,kind,3)

plt.figure(0)
plt.imshow(np.log(phasehistogram_0+0.01), cmap='hot', aspect='auto', origin='lower', extent = (0,81600,-0.6,0.6))
plt.colorbar()
plt.figure(1)
plt.imshow(np.log(phasehistogram_1+0.01), cmap='hot', aspect='auto', origin='lower', extent = (0,81600,-0.6,0.6))
plt.colorbar()
plt.figure(2)
plt.imshow(np.log(phasehistogram_2+0.01), cmap='hot', aspect='auto', origin='lower', extent = (0,81600,-0.25,0.25))
plt.colorbar()
plt.figure(3)
plt.imshow(np.log(phasehistogram_3+0.01), cmap='hot', aspect='auto', origin='lower', extent = (0,81600,-0.25,0.25))
plt.colorbar()
plt.show()


#get arrays
#get lists that correspond to a species
ls_0 = np.where(kind == 0)[0]
ls_1 = np.where(kind == 1)[0]
ls_2 = np.where(kind == 2)[0]
ls_3 = np.where(kind == 3)[0]

#get position array
zz_0 = zz_fixed[ls_0]
zz_1 = zz_fixed[ls_1]
zz_2 = zz_fixed[ls_2]
zz_3 = zz_fixed[ls_3]

#get velocity array
uz_0 = uz[ls_0]
uz_1 = uz[ls_1]
uz_2 = uz[ls_2]
uz_3 = uz[ls_3]

