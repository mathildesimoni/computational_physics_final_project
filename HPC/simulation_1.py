import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import time
import os
import glob
from functions import *

# non-dimentionalize code
# universal gravitation constant
G = 6.67259e-20  # (km**3/kg/s**2)
mass_sun = 1.989e30 # kg
AU = 1.496e8 #km
year = 3.156e7 #s
norm_const = (G * mass_sun * (year**2))/(AU**3)

# initial conditions: store in arrays
# NASA JBL 01/01/2023 (vector table)
# observer: solar system barycenter 
# input_file = "/scratch/mps565/comp_phy_simulations/bodies2.csv"
# folder = "positions"

input_file = "/scratch/mps565/comp_phy_simulations/horizons_data_init_01_01_1750.csv"
folder = "positions_250_years_init_01_01_1750_2.5_day_timestep_runga_kutta"
n_days_total = 365 * 250 # total number of days I want to simulate

n_days_timestep = 2.5 # timestep in number of days
n_timesteps = int(n_days_total/n_days_timestep) # number of timesteps for simulation
# print("Number of timesteps:", n_timesteps)
timestep = (86400/year) * n_days_timestep # (86400/year) = 1 day in non dimensional
# print("Timestep:", timestep)

# variables for Bulirsch-Stoer method
N = 4
h = timestep/N
q = 2
p = 1

step_save = 2 # every ... timestep

func_integration =  "Runga-Kutta_optimized" #"Bulirsch-Stoer_optimized" #"Runga-Kutta" "Bulisrsch-Stoer"  
args_bulirsch_stoer = [N, h, q, p]

n_bodies, bodies, exec_time = run_simulation_save(input_file, n_timesteps, timestep, func_integration, args_bulirsch_stoer, folder, step_save)

print("folder:", folder)
print("n_days_total:", n_days_total)
print("step_save:", step_save)
print("n_bodies:", n_bodies)
print("exec time", exec_time)





