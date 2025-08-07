import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#from scipy.interpolate import interp1d

def larmor_radius(KE, B, charge_mass_ratio = 9.58e7, charge_in_e = 1):
    c = 3e8
    q = (1.6e-19)*charge_in_e
    m = q/charge_mass_ratio
    E = KE+m*9e16
    p = (1/c)*np.sqrt((E**2)-((m**2)*(c**4)))
    R = p/(q*B)
    return R

def particle_pusher(
        E, 
        B, 
        delta_d, 
        charge_mass_ratio, 
        charge_in_e, 
        theta_z, 
        theta, 
        x, 
        y, 
        z):
    if B != 0:
        #find larmor radius
        R = larmor_radius(E*math.cos(theta_z), B, charge_mass_ratio, charge_in_e)
        #find change in XY-plane angle theta for arc distance delta d
        delta_theta_adjusted = delta_d*math.cos(theta_z)/R
        #this is the x and y displacement in the rotated frame, where the xv axis points along the XY-plane velocity of the particle
        delta_xv = R*math.sin(delta_theta_adjusted)
        delta_yv = R*(1-math.cos(delta_theta_adjusted))
        #we now transform the displacement in the rotated xv, yv frame to the global non-rotated frame
        delta_x = delta_xv*math.cos(theta)-delta_yv*math.sin(theta)
        delta_y = delta_xv*math.sin(theta)+delta_yv*math.cos(theta)
        #finally we add the distances to their prior values and change the velocity vector direction by delta theta
        X = x+delta_x
        Y = y+delta_y
        Z = z+delta_d*math.sin(theta_z)
        Theta = theta+delta_theta_adjusted
    elif B == 0:
        #if there is no B field, the velocity does not change. the displacement in the rotated frame is just the XY-plane displacement only along the xv axis
        delta_theta_adjusted = 0
        delta_xv = delta_d*math.cos(theta_z)
        delta_yv = 0
        #we still have to transform from the rotated fram back to the global, unrotated frame
        delta_x = delta_xv*math.cos(theta)-delta_yv*math.sin(theta)
        delta_y = delta_xv*math.sin(theta)+delta_yv*math.cos(theta)
        X = x+delta_x
        Y = y+delta_y
        Z = z+delta_d*math.sin(theta_z)
        Theta = theta
    return Theta, X, Y, Z

def track_particle_2D(
        #input everything in SI units, except for E being in eV
        #B is in the Z direction for the 2D tracker
        Energy_eV,
        B_data = np.array([1]),
        B_axes = np.array([0]),
        theta_0 = 0,
        theta_z = 0,
        d_tcc = 0.5,
        delta_d = 0.0005,
        nonuniform_B = True,
        uniform_B = 1,
        charge_mass_ratio = 9.58e7,
        charge_in_e = 1,
        gapx = 0,
        gapy = 0,
        plate = 'curved',
        IP_radius = 0.14102,
        IP_center = np.array([0.0450215,-0.12394]),
        x_max = 1,
        y_max = 1,
        dynamic_step = 1,
        fine_adjustment = 16):
    #initializing variables
    E = (Energy_eV * 1.6e-19)
    x_array = []
    y_array = []
    z_array = []
    x = -gapx
    y = d_tcc*math.sin(theta_0)*math.cos(theta_z)
    z = d_tcc*math.sin(theta_z)
    theta = theta_0
    x_array.append(x)
    y_array.append(y)
    z_array.append(z)
    terminate = False

    if dynamic_step < 1:
        grad_B_data = np.gradient(B_data,B_axes)
        grad_B_max = np.max(grad_B_data)
        delta_D = delta_d

#code block for gap
    if gapx != 0:
        x = -gapx
        while -gapx-delta_d < x < 0 and -gapy < y < gapy and terminate == False:
            if nonuniform_B == True:
                B = np.interp(x+0.33*delta_d*math.cos(theta)*math.cos(theta_z),B_axes,B_data)
                if dynamic_step < 1:
                    grad_B_adj = 1+(dynamic_step-1)*np.interp(x,B_axes,grad_B_data)/grad_B_max
                    delta_d = delta_D/grad_B_adj
            elif nonuniform_B == False:
                B = uniform_B
            
            if B != 0:
                R = larmor_radius(E*math.cos(theta_z), B, charge_mass_ratio, charge_in_e)
                delta_theta_adjusted = delta_d*math.cos(theta_z)/R
                delta_xv = R*math.sin(delta_theta_adjusted)
                delta_yv = R*(1-math.cos(delta_theta_adjusted))
                delta_x = delta_xv*math.cos(theta)-delta_yv*math.sin(theta)
                delta_y = delta_xv*math.sin(theta)+delta_yv*math.cos(theta)
                x = x+delta_x
                y = y+delta_y
                z = z+delta_d*math.sin(theta_z)
            elif B == 0:
                delta_theta_adjusted = 0
                delta_xv = delta_d
                delta_yv = 0
                delta_x = delta_xv*math.cos(theta)-delta_yv*math.sin(theta)
                delta_y = delta_xv*math.sin(theta)+delta_yv*math.cos(theta)
                x = x+delta_x
                y = y+delta_y
                z = z+delta_d*math.sin(theta_z)
            if y > gapy or y < -gapy and x < 0:
                terminate = True
            
            theta = theta+delta_theta_adjusted
            x_array.append(x)
            y_array.append(y)
            z_array.append(z)

#code block for curved plate
    if plate == 'curved':
        while (((x-IP_center[0])**2)+((y-IP_center[1])**2) < (IP_radius-2*delta_d)**2) and -gapx < x < 1 and y < 1 and terminate == False:
            #find the B field at the particle position
            if nonuniform_B == True:
                #adjust B field given a straight trajectory, take the B field nearer the midpoint of travel
                B = np.interp(x+0.33*delta_d*math.cos(theta)*math.cos(theta_z),B_axes,B_data)
                if dynamic_step < 1:
                    #grad_B_adj has a minimum value of 1
                    #grad_B_adj increases to a maximum value of dynamic_step
                    #grad_B_adj determines h0w many times the step is reduced by
                    #dynamic_step determines the maximum value of grad_B_adj
                    grad_B_adj = 1+(dynamic_step-1)*np.interp(x,B_axes,grad_B_data)/grad_B_max
                    delta_d = delta_D/grad_B_adj
            elif nonuniform_B == False:
                B = uniform_B
            
            #find the larmor radius of the particle. If the particle has some z-momentum, the energy has a smaller 'effective energy' in the perpendicular direction
            R = larmor_radius(E*math.cos(theta_z), B, charge_mass_ratio, charge_in_e)
            
            #the distance traveled in the perp direction
            delta_theta_adjusted = delta_d*math.cos(theta_z)/R
            delta_xv = R*math.sin(delta_theta_adjusted)
            delta_yv = R*(1-math.cos(delta_theta_adjusted))
            delta_x = delta_xv*math.cos(theta)-delta_yv*math.sin(theta)
            delta_y = delta_xv*math.sin(theta)+delta_yv*math.cos(theta)
            x = x+delta_x
            y = y+delta_y
            #the distance traveled in the par direction
            z = z+delta_d*math.sin(theta_z)
            theta = theta+delta_theta_adjusted
            x_array.append(x)
            y_array.append(y)
            z_array.append(z)

        while (((x-IP_center[0])**2)+((y-IP_center[1])**2) < IP_radius**2) and -gapx < x < 1 and y < 1 and terminate == False:
            if nonuniform_B == True:
                B = np.interp(x+0.33*delta_d*math.cos(theta)*math.cos(theta_z),B_axes,B_data)
                if dynamic_step < 1:
                    grad_B_adj = 1+(dynamic_step-1)*np.interp(x,B_axes,grad_B_data)/grad_B_max
                    delta_d = delta_D/grad_B_adj
            elif nonuniform_B == False:
                B = uniform_B

            R = larmor_radius(E*math.cos(theta_z), B, charge_mass_ratio, charge_in_e)

            delta_theta_adjusted = delta_d*math.cos(theta_z)/(R*fine_adjustment)
            delta_xv = R*math.sin(delta_theta_adjusted)
            delta_yv = R*(1-math.cos(delta_theta_adjusted))
            delta_x = delta_xv*math.cos(theta)-delta_yv*math.sin(theta)
            delta_y = delta_xv*math.sin(theta)+delta_yv*math.cos(theta)
            x = x+delta_x
            y = y+delta_y
            z = z+delta_d*math.sin(theta_z)
            theta = theta+delta_theta_adjusted
            x_array.append(x)
            y_array.append(y)
            z_array.append(z)
            
# code block for flat plate
    if plate == 'flat':
        while -gapx < x < x_max-delta_d and y < y_max-delta_d and terminate == False:
            if nonuniform_B == True:
                B = np.interp(x+0.33*delta_d*math.cos(theta)*math.cos(theta_z),B_axes,B_data)
                if dynamic_step < 1:
                    grad_B_adj = 1+(dynamic_step-1)*np.interp(x,B_axes,grad_B_data)/grad_B_max
                    delta_d = delta_D/grad_B_adj
            elif nonuniform_B == False:
                B = uniform_B
            
            R = larmor_radius(E*math.cos(theta_z), B, charge_mass_ratio, charge_in_e)
            
            delta_theta_adjusted = delta_d*math.cos(theta_z)/R
            delta_xv = R*math.sin(delta_theta_adjusted)
            delta_yv = R*(1-math.cos(delta_theta_adjusted))
            delta_x = delta_xv*math.cos(theta)-delta_yv*math.sin(theta)
            delta_y = delta_xv*math.sin(theta)+delta_yv*math.cos(theta)
            x = x+delta_x
            y = y+delta_y
            z = z+delta_d*math.sin(theta_z)
            theta = theta+delta_theta_adjusted
            x_array.append(x)
            y_array.append(y)
            z_array.append(z)

        while -gapx < x < x_max and y < y_max and terminate == False:
            if nonuniform_B == True:
                B = np.interp(x+0.33*delta_d*math.cos(theta)*math.cos(theta_z),B_axes,B_data)
                if dynamic_step < 1:
                    grad_B_adj = 1+(dynamic_step-1)*np.interp(x,B_axes,grad_B_data)/grad_B_max
                    delta_d = delta_D/grad_B_adj
            elif nonuniform_B == False:
                B = uniform_B

            R = larmor_radius(E*math.cos(theta_z), B, charge_mass_ratio, charge_in_e)

            delta_theta_adjusted = delta_d*math.cos(theta_z)/(R*fine_adjustment)
            delta_xv = R*math.sin(delta_theta_adjusted)
            delta_yv = R*(1-math.cos(delta_theta_adjusted))
            delta_x = delta_xv*math.cos(theta)-delta_yv*math.sin(theta)
            delta_y = delta_xv*math.sin(theta)+delta_yv*math.cos(theta)
            x = x+delta_x
            y = y+delta_y
            z = z+delta_d*math.sin(theta_z)
            theta = theta+delta_theta_adjusted
            x_array.append(x)
            y_array.append(y)
            z_array.append(z)
            
    return np.array(x_array), np.array(y_array), np.array(z_array)