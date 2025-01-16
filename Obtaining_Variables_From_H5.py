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
    bz = hf['bz'][()]
    by = hf['by'][()]

# rho = number density
# i = ion
# e = electron 
# tar = piston plasma
# am = ambient plasma
# Tss = momentum-velocity tensor
# j = current density
# p = momentum density

filename2 = "smaller_moments.h5"

with h5py.File(filename2, "r") as hf2:
    jz_i_tar = hf2['jz_i_tar'][()]
    rho_i_tar = hf2['rho_i_tar'][()]
    rho_i_am = hf2['rho_i_am'][()]
    rho_e_tar = hf2['rho_e_tar'][()]
    rho_e_am = hf2['rho_e_am'][()]
    pz_i_tar = hf2['pz_i_tar'][()]

mean_jz_i_tar = np.mean(jz_i_tar, axis=1)
mean_pz_i_tar = np.mean(pz_i_tar, axis=1)
mean_rho_e_tar = np.mean(rho_e_tar, axis=1)
mean_rho_e_am = np.mean(rho_e_am, axis=1)
mean_rho_i_tar = np.mean(rho_i_tar, axis=1)
mean_rho_i_am = np.mean(rho_i_am, axis=1)


filename3 = "smaller_prts.h5"

with h5py.File(filename3, "r") as hf3:
    xb = hf3['xb'][()]
    xe = hf3['xe'][()]
    z2 = (xb + xe) / 2
    npatch = hf3['n_patch'][:]  # Read npatch dataset
    
    z2 = z2[850:1700, 2]  # Ensure z2 does not exceed the length of npatch
    
    pz = np.zeros(len(z2))
    den = np.zeros(len(z2))
    uz = hf3['uz'][()]
    zz = hf3['z'][()]
    kind = hf3['kind'][()]

    p1 = 0
    for i in range(850):
        p2 = p1 + npatch[850+i]  # Access npatch directly using the index i
        uz_z = uz[p1:p2, 0]
        kind_z = kind[p1:p2, 0]
        ls = np.where(kind_z == 3)[0]
        den[i] = len(ls)
        if len(ls) > 0:
            pz[i] = np.sum(uz_z[ls])
        else:
            pz[i] = 0
        p1 = p2

pz = pz / (npc * nn)
den = den / (npc * nn)

def fix_zz(nptch,zz_input,xb_input):
    zz_fixed=[]
    p1 = 0
    for i in range(850):
        delta = nptch[850+i] # delta is the number of particles in a patch.
        p2 = p1 + delta  # Here we keep track of the total number of particles, the nth patch lies between the start of the patch p1 and p1+delta
        zz_patch = zz_input[p1:p2, 0].tolist() # we extract only the particles in the patch and put them in a list
        zz_patch_fix = [10*x+xb_input[850+i,2] for x in zz_patch] # we then add the position of the patch xb to the relative position of the particles zz to get the absolut position of the particles in this patch
        zz_fixed.extend(zz_patch_fix) # we put the absolute positions of each particle into the new list
        print("npatch index:", i, end='\r') # this tracks which patch the program is on
        p1 = p2 #to get to the next patch, we set the start of the next patch to be the end of the previous patch
    z_fixed = np.array(zz_fixed) # we change the list back to an array
    return z_fixed
zz_fixed = fix_zz(npatch,zz,xb)

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
plt.imshow(phasehistogram_0, cmap='hot', aspect='auto', origin='lower', extent = (0,81600,-0.6,0.6))
plt.colorbar()
plt.figure(1)
plt.imshow(phasehistogram_1, cmap='hot', aspect='auto', origin='lower', extent = (0,81600,-0.6,0.6))
plt.colorbar()
plt.figure(2)
plt.imshow(phasehistogram_2, cmap='hot', aspect='auto', origin='lower', extent = (0,81600,-0.25,0.25))
plt.colorbar()
plt.figure(3)
plt.imshow(phasehistogram_3, cmap='hot', aspect='auto', origin='lower', extent = (0,81600,-0.25,0.25))
plt.colorbar()
plt.show