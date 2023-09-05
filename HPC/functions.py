import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import time
import os
import glob

G = 6.67259e-20  # (km**3/kg/s**2)
mass_sun = 1.989e30 # kg
AU = 1.496e8 #km
year = 3.156e7 #s
norm_const = (G * mass_sun * (year**2))/(AU**3)

def bulirsch_stoer_3d_optimized(position, velocity, timestep, Fx, Fy, Fz, args_x, args_y, args_z, N, h, q, p):
    # step 1: modified midpoint method with step size h
    x_values1, y_values1, z_values1, Vx_values1, Vy_values1, Vz_values1 = modified_midpoint_method_optimized(position, velocity, Fx, Fy, Fz, args_x, args_y, args_z, N, h)
    
    # step 2: modified midpoint method with step size h/q
    N_2 = int((timestep * q)/h)
    h_2 = h/q
    
    x_values2, y_values2, z_values2, Vx_values2, Vy_values2, Vz_values2 = modified_midpoint_method_optimized(position, velocity, Fx, Fy, Fz, args_x, args_y, args_z, N_2, h_2)
    
    # step 3: Richardson Extrapolation
    x_final = Richardson_extrapolation(x_values1, x_values2, q, p)
    y_final = Richardson_extrapolation(y_values1, y_values2, q, p)
    z_final = Richardson_extrapolation(z_values1, z_values2, q, p)
    
    Vx_final = Richardson_extrapolation(Vx_values1, Vx_values2, q, p)
    Vy_final = Richardson_extrapolation(Vy_values1, Vy_values2, q, p)
    Vz_final = Richardson_extrapolation(Vz_values1, Vz_values2, q, p)
    
    return (x_final, y_final, z_final), (Vx_final, Vy_final, Vz_final)

def Richardson_extrapolation(f_h, f_hq, q, p):
    return f_h + (f_h - f_hq)/((q**(-p)) -1)

def modified_midpoint_method_optimized(position, velocity, Fx, Fy, Fz, args_x, args_y, args_z, N, h):
    x_values = [None] * (N + 1)
    y_values = [None] * (N + 1)
    z_values = [None] * (N + 1)
    
    Vx_values = [None] * (N + 1)
    Vy_values = [None] * (N + 1)
    Vz_values = [None] * (N + 1)
    
    Fx_values = [None] * (N + 1)
    Fy_values = [None] * (N + 1)
    Fz_values = [None] * (N + 1)
    
    # prep
    x_values[0] = position[0]
    y_values[0] = position[1]
    z_values[0] = position[2]
    
    Vx_values[0] = velocity[0]
    Vy_values[0] = velocity[1]
    Vz_values[0] = velocity[2]
    
    Fx_values[0] = Fx(args_x, Vx_values[0], Vy_values[0], Vz_values[0], x_values[0], y_values[0], z_values[0])
    Fy_values[0] = Fy(args_y, Vx_values[0], Vy_values[0], Vz_values[0], x_values[0], y_values[0], z_values[0])
    Fz_values[0] = Fz(args_z, Vx_values[0], Vy_values[0], Vz_values[0], x_values[0], y_values[0], z_values[0])
    
    # step 1
    # print("step:", 1)
    x_values[1] = x_values[0] + Vx_values[0] * h
    y_values[1] = y_values[0] + Vy_values[0] * h
    z_values[1] = z_values[0] + Vz_values[0] * h

    Vx_values[1] = Vx_values[0] + Fx_values[0] * h
    Vy_values[1] = Vy_values[0] + Fy_values[0] * h
    Vz_values[1] = Vz_values[0] + Fz_values[0] * h

    Fx_values[1] = Fx(args_x, Vx_values[1], Vy_values[1], Vz_values[1], x_values[1], y_values[1], z_values[1])
    Fy_values[1] = Fy(args_y, Vx_values[1], Vy_values[1], Vz_values[1], x_values[1], y_values[1], z_values[1])
    Fz_values[1] = Fz(args_z, Vx_values[1], Vy_values[1], Vz_values[1], x_values[1], y_values[1], z_values[1])

    # steps 2 +
    for n in range(2, N+1):
        # print("step:", n)
        x_values[n] = x_values[n-2] + Vx_values[n-1] * 2 * h
        y_values[n] = y_values[n-2] + Vy_values[n-1] * 2 * h
        z_values[n] = z_values[n-2] + Vz_values[n-1] * 2 * h

        Vx_values[n] = Vx_values[n-2] + Fx_values[n-1] * 2 * h
        Vy_values[n] = Vy_values[n-2] + Fy_values[n-1] * 2 * h
        Vz_values[n] = Vz_values[n-2] + Fz_values[n-1] * 2 * h

        Fx_values[n] = Fx(args_x, Vx_values[n], Vy_values[n], Vz_values[n], x_values[n], y_values[n], z_values[n])
        Fy_values[n] = Fy(args_y, Vx_values[n], Vy_values[n], Vz_values[n], x_values[n], y_values[n], z_values[n])
        Fz_values[n] = Fz(args_z, Vx_values[n], Vy_values[n], Vz_values[n], x_values[n], y_values[n], z_values[n])
    
    # final step
    x_values = 1/2 * (x_values[N] + x_values[N-1] + (h * Vx_values[N]))
    y_values = 1/2 * (y_values[N] + y_values[N-1] + (h * Vy_values[N]))
    z_values = 1/2 * (z_values[N] + z_values[N-1] + (h * Vz_values[N]))

    Vx_values = 1/2 * (Vx_values[N] + Vx_values[N-1] + (h * Fx_values[N]))
    Vy_values = 1/2 * (Vy_values[N] + Vy_values[N-1] + (h * Fy_values[N]))
    Vz_values = 1/2 * (Vz_values[N] + Vz_values[N-1] + (h * Fz_values[N]))
    
    return x_values, y_values, z_values, Vx_values, Vy_values, Vz_values


def runga_kutta_3d_optimized(position, velocity, timestep, Fx, Fy, Fz, args_x, args_y, args_z):
    x_values = [None] * 4
    y_values = [None] * 4
    z_values = [None] * 4
    
    Vx_values = [None] * 4
    Vy_values = [None] * 4
    Vz_values = [None] * 4
    
    Fx_values = [None] * 4
    Fy_values = [None] * 4
    Fz_values = [None] * 4
    
    # step 1
    x_values[0] = position[0]
    y_values[0] = position[1]
    z_values[0] = position[2]
    
    Vx_values[0] = velocity[0]
    Vy_values[0] = velocity[1]
    Vz_values[0] = velocity[2]
    
    Fx_values[0] = Fx(args_x, Vx_values[0], Vy_values[0], Vz_values[0], x_values[0], y_values[0], z_values[0])
    Fy_values[0] = Fy(args_y, Vx_values[0], Vy_values[0], Vz_values[0], x_values[0], y_values[0], z_values[0])
    Fz_values[0] = Fz(args_z, Vx_values[0], Vy_values[0], Vz_values[0], x_values[0], y_values[0], z_values[0])
    
    # step 2
    x_values[1] = x_values[0] + Vx_values[0] * timestep/2
    y_values[1] = y_values[0] + Vy_values[0] * timestep/2
    z_values[1] = z_values[0] + Vz_values[0] * timestep/2
    
    Vx_values[1] = Vx_values[0] + Fx_values[0] * timestep/2
    Vy_values[1] = Vy_values[0] + Fy_values[0] * timestep/2
    Vz_values[1] = Vz_values[0] + Fz_values[0] * timestep/2
    
    Fx_values[1] = Fx(args_x, Vx_values[1], Vy_values[1], Vz_values[1], x_values[1], y_values[1], z_values[1])
    Fy_values[1] = Fy(args_y, Vx_values[1], Vy_values[1], Vz_values[1], x_values[1], y_values[1], z_values[1])
    Fz_values[1] = Fz(args_z, Vx_values[1], Vy_values[1], Vz_values[1], x_values[1], y_values[1], z_values[1])
    
    # step 3
    x_values[2] = x_values[0] + Vx_values[1] * timestep/2
    y_values[2] = y_values[0] + Vy_values[1] * timestep/2
    z_values[2] = z_values[0] + Vz_values[1] * timestep/2
    
    Vx_values[2] = Vx_values[0] + Fx_values[1] * timestep/2
    Vy_values[2] = Vy_values[0] + Fy_values[1] * timestep/2
    Vz_values[2] = Vz_values[0] + Fz_values[1] * timestep/2
    
    Fx_values[2] = Fx(args_x, Vx_values[2], Vy_values[2], Vz_values[2], x_values[2], y_values[2], z_values[2])
    Fy_values[2] = Fy(args_y, Vx_values[2], Vy_values[2], Vz_values[2], x_values[2], y_values[2], z_values[2])
    Fz_values[2] = Fz(args_z, Vx_values[2], Vy_values[2], Vz_values[2], x_values[2], y_values[2], z_values[2])
    
    # step 4
    x_values[3] = x_values[0] + Vx_values[2] * timestep
    y_values[3] = y_values[0] + Vy_values[2] * timestep
    z_values[3] = z_values[0] + Vz_values[2] * timestep
    
    Vx_values[3] = Vx_values[0] + Fx_values[2] * timestep
    Vy_values[3] = Vy_values[0] + Fy_values[2] * timestep
    Vz_values[3] = Vz_values[0] + Fz_values[2] * timestep
    
    Fx_values[3] = Fx(args_x, Vx_values[3], Vy_values[3], Vz_values[3], x_values[3], y_values[3], z_values[3])
    Fy_values[3] = Fy(args_y, Vx_values[3], Vy_values[3], Vz_values[3], x_values[3], y_values[3], z_values[3])
    Fz_values[3] = Fz(args_z, Vx_values[3], Vy_values[3], Vz_values[3], x_values[3], y_values[3], z_values[3])
    
    # step 5
    x_final = x_values[0] + ((Vx_values[0] + 2*Vx_values[1] + 2*Vx_values[2] + Vx_values[3])*timestep)/6
    y_final = y_values[0] + ((Vy_values[0] + 2*Vy_values[1] + 2*Vy_values[2] + Vy_values[3])*timestep)/6
    z_final = z_values[0] + ((Vz_values[0] + 2*Vz_values[1] + 2*Vz_values[2] + Vz_values[3])*timestep)/6
    
    Vx_final = Vx_values[0] + ((Fx_values[0] +2*Fx_values[1] +2*Fx_values[2] + Fx_values[3])*timestep)/6
    Vy_final = Vy_values[0] + ((Fy_values[0] +2*Fy_values[1] +2*Fy_values[2] + Fy_values[3])*timestep)/6
    Vz_final = Vz_values[0] + ((Fz_values[0] +2*Fz_values[1] +2*Fz_values[2] + Fz_values[3])*timestep)/6
    
    return (x_final, y_final, z_final), (Vx_final, Vy_final, Vz_final)

def gravitational_force(args, Vx, Vy, Vz, x, y, z):
    n = args[0] # position of body in array of all bodies
    m = args[1] # mass array
    component = args[2] # "x", "y", or "z"
    x_all = args[3] # last x position of all bodies (array)
    y_all = args[4] # last x position of all bodies (array)
    z_all = args[5] # last x position of all bodies (array)
    G_or_const = args[6]
    
    F = 0

    for body in range(len(x_all)):
        if body != n:
            # claculate distance between 2 bodies
            r =  math.sqrt((x_all[body] - x)**2 + (y_all[body] - y)**2 + (z_all[body] - z)**2)
            # print("r:", r)
            if component == "x":
                F -= G_or_const * m[body] * ((x - x_all[body])/r**3)
            elif component == "y":
                F -= G_or_const * m[body] * ((y - y_all[body])/r**3)
            elif component == "z":
                F -= G_or_const * m[body] * ((z - z_all[body])/r**3)      
    return F

# instead of keeping all positions in memory in array, this function stores positions in files
def run_simulation_save(input_file, n_timesteps, timestep, func_integration, args, folder, step_save):
    df = pd.read_csv(input_file)
    bodies = list(df['body'])
    df = df.set_index('body')
    n_bodies = len(df)
    # print(n_bodies, "bodies:", bodies)
    
    # set array of masses and normalize
    m = list(df["mass"])
    m = [i/mass_sun for i in m]
    # print("Masses:", m)

    # initialization positions and velocities in temporary arrays (changed at each step)
    x = [None] * n_bodies
    y = [None] * n_bodies
    z = [None] * n_bodies
    Vx = [None] * n_bodies
    Vy = [None] * n_bodies
    Vz = [None] * n_bodies
    
    # initiaization folder
    create_empty_folder(folder)
    
    for n in range(n_bodies):
        row = df.loc[bodies[n]]
        x[n] = row.x/AU
        y[n] = row.y/AU
        z[n] = row.z/AU

        Vx[n] = row.Vx*(year/AU)
        Vy[n] = row.Vy*(year/AU)
        Vz[n] = row.Vz*(year/AU)
    # save
    save_timestep(bodies, x, y, z, Vx, Vy, Vz, folder, 0)
    
    if func_integration == "Bulirsch-Stoer" or func_integration == "Bulirsch-Stoer_optimized":
        N = args[0]
        h = args[1]
        q = args[2]
        p = args[3]
    
    start_time = time.time()
    # run simulation
    for t in range(1, n_timesteps + 1):
    
        for n in range(n_bodies):
            # for a specific body: p_components are its position, c_components are its velocity components
            p_components = (x[n], y[n], z[n])
            v_components = (Vx[n], Vy[n], Vz[n])

            args_x = (n, m, "x", x, y, z, norm_const)
            args_y = (n, m, "y", x, y, z, norm_const)
            args_z = (n, m, "z", x, y, z, norm_const)
            
            if func_integration == "Bulirsch-Stoer":
                p_components, v_components = bulirsch_stoer_3d(p_components, v_components, timestep, gravitational_force, gravitational_force, gravitational_force, args_x, args_y, args_z, N, h, q, p)  
            elif func_integration == "Bulirsch-Stoer_optimized":
                p_components, v_components = bulirsch_stoer_3d_optimized(p_components, v_components, timestep, gravitational_force, gravitational_force, gravitational_force, args_x, args_y, args_z, N, h, q, p)  
            elif func_integration == "Runga-Kutta":
                p_components, v_components = runga_kutta_3d(p_components, v_components, timestep, gravitational_force, gravitational_force, gravitational_force, args_x, args_y, args_z)
            elif func_integration == "Runga-Kutta_optimized":
                p_components, v_components = runga_kutta_3d_optimized(p_components, v_components, timestep, gravitational_force, gravitational_force, gravitational_force, args_x, args_y, args_z)
        
            # save new values in arrays
            x[n] = p_components[0]
            y[n] = p_components[1]
            z[n] = p_components[2]

            Vx[n] = v_components[0]
            Vy[n] = v_components[1]
            Vz[n] = v_components[2]
            
        # save values in files
        if t % step_save == 0:
            save_timestep(bodies, x, y, z, Vx, Vy, Vz, folder, t)   
                 
    exec_time = time.time() - start_time
    return n_bodies, bodies, exec_time


# run simualtion with a FIXED SUN
def run_simulation_save_fix_sun(input_file, n_timesteps, timestep, func_integration, args, folder, step_save):
    df = pd.read_csv(input_file)
    bodies = list(df['body'])
    df = df.set_index('body')
    n_bodies = len(df)
    # print(n_bodies, "bodies:", bodies)
    
    # set array of masses and normalize
    m = list(df["mass"])
    m = [i/mass_sun for i in m]
    # print("Masses:", m)

    # initialization positions and velocities in temporary arrays (changed at each step)
    x = [None] * n_bodies
    y = [None] * n_bodies
    z = [None] * n_bodies
    Vx = [None] * n_bodies
    Vy = [None] * n_bodies
    Vz = [None] * n_bodies
    
    # initiaization folder
    create_empty_folder(folder)
    
    for n in range(n_bodies):
        row = df.loc[bodies[n]]
        x[n] = row.x/AU
        y[n] = row.y/AU
        z[n] = row.z/AU

        Vx[n] = row.Vx*(year/AU)
        Vy[n] = row.Vy*(year/AU)
        Vz[n] = row.Vz*(year/AU)
    # save
    save_timestep(bodies, x, y, z, Vx, Vy, Vz, folder, 0)
    
    if func_integration == "Bulirsch-Stoer" or func_integration == "Bulirsch-Stoer_optimized":
        N = args[0]
        h = args[1]
        q = args[2]
        p = args[3]
    
    start_time = time.time()
    # run simulation
    for t in range(1, n_timesteps + 1):
    
        for n in range(n_bodies):
            # for a specific body: p_components are its position, c_components are its velocity components
            p_components = (x[n], y[n], z[n])
            v_components = (Vx[n], Vy[n], Vz[n])
            
            if n != 0: # if it s not the sun
                args_x = (n, m, "x", x, y, z, norm_const)
                args_y = (n, m, "y", x, y, z, norm_const)
                args_z = (n, m, "z", x, y, z, norm_const)

                if func_integration == "Bulirsch-Stoer":
                    p_components, v_components = bulirsch_stoer_3d(p_components, v_components, timestep, gravitational_force, gravitational_force, gravitational_force, args_x, args_y, args_z, N, h, q, p)  
                elif func_integration == "Bulirsch-Stoer_optimized":
                    p_components, v_components = bulirsch_stoer_3d_optimized(p_components, v_components, timestep, gravitational_force, gravitational_force, gravitational_force, args_x, args_y, args_z, N, h, q, p)  
                elif func_integration == "Runga-Kutta":
                    p_components, v_components = runga_kutta_3d(p_components, v_components, timestep, gravitational_force, gravitational_force, gravitational_force, args_x, args_y, args_z)
                elif func_integration == "Runga-Kutta_optimized":
                    p_components, v_components = runga_kutta_3d_optimized(p_components, v_components, timestep, gravitational_force, gravitational_force, gravitational_force, args_x, args_y, args_z)

            # save new values in arrays
            x[n] = p_components[0]
            y[n] = p_components[1]
            z[n] = p_components[2]

            Vx[n] = v_components[0]
            Vy[n] = v_components[1]
            Vz[n] = v_components[2]
            
        # save values in files
        if t % step_save == 0:
            save_timestep(bodies, x, y, z, Vx, Vy, Vz, folder, t)   
                 
    exec_time = time.time() - start_time
    return n_bodies, bodies, exec_time

def create_empty_folder(folder):
    if os.path.exists(folder):
        files = glob.glob(folder + '/*')
        for f in files:
            os.remove(f)
    else:
        os.mkdir(folder)

def save_timestep(bodies, x, y, z, Vx, Vy, Vz, folder, timestep):
    # create dataframe
    df = pd.DataFrame()
    df["body"] = bodies
    df["x"] = x
    df["y"] = y
    df["z"] = z
    df["Vx"] = Vx
    df["Vy"] = Vy
    df["Vz"] = Vz
    
    df.to_csv(folder + "/t_%s.csv"%(str(timestep)), index = False)

