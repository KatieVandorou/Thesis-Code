
# coding: utf-8

# In[ ]:

import numpy as np
import astropy.units as u #To keep track of units
import matplotlib.pyplot as plt #For plotting
import plutokore as pk
import pyPLUTO as pp
import matplotlib as mpl
from plutokore import simulations
from plutokore import io
from plutokore import plot as _plot
from plutokore.environments.king import KingProfile
from astropy import units as u 
from astropy import cosmology


# # Calculating Jet Parameters

# In[ ]:

# Defining the environment properties:
     mass = (10 ** 14.5) * u.M_sun # The dark matter halo mass
        z =  0                     # The environment redshift
delta_vir = 200
    cosmo = cosmology.Planck15  
    
concentration_method = 'klypin-planck-relaxed' 

# Creating the King profile:
profile = KingProfile(mass, z, delta_vir=delta_vir, cosmo=cosmo, concentration_method=concentration_method)

from plutokore.jet import AstroJet

# Define the jet properties:
theta_deg = 15         # The half-opening angle in degrees, can be changed to 30 degrees
      M_x = 25         # The Mach number
        Q = 1e38 * u.W # The jet power, can be changed to 1e37

# Creating the jet
jet = AstroJet(theta_deg, M_x, profile.sound_speed, profile.central_density, Q, profile.gamma)

# Printing some environment properties calculated in plutokore:
print('Central density is {0}'.format(profile.central_density))
print('Virial radius is {0}'.format(profile.virial_radius))
print('Concentration is {0}'.format(profile.concentration))
print('Core radius is {0}'.format(profile.core_radius))
print('Sound speed is {0}'.format(profile.sound_speed))
print('External temperature is {}'.format(profile.T))

pk.utilities.printmd(jet.get_calculated_parameter_table())
unit_values = pk.jet.get_unit_values(profile, jet)
dict(unit_values._asdict()) # Display the unit values in the notebook

# Converting to physical units:
scaling_dict = dict(unit_values._asdict())
length_scaling = scaling_dict['length']
density_scaling = scaling_dict['density']
pressure_scaling = scaling_dict['pressure']
speed_scaling = scaling_dict['speed']

# Defining the run directory:
run_dir = '/Users/Katie/Desktop/PLUTO1/KatiesSims/Offset/M25_m14.5_OFF4R_Q38/' 



# # The Environment 

# In[ ]:

# Choosing the initial time step
tstep = 0  
initial = pp.pload(tstep,w_dir = run_dir) 

theta_index_first = 0
theta_index_last  = -1
       r_physical = curObject.x1*length_scaling.value


# loading the density data, and taking a slice along the x-axis
globalVar_density = getattr(initial,'rho').T
globalVar_density = globalVar_density * density_scaling.value
    density_slice = np.asarray(np.log10(globalVar_density))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.set_figheight(5)
f.set_figwidth(10)

ax1.plot(-r_physical, density_slice[theta_index_last,:])
ax2.plot(r_physical, density_slice[theta_index_first,:])

ax1.set_ylabel(r'$\log(\rho) (g/cm^3)$')
ax1.set_xlabel('X(kpc)')
ax2.set_xlabel('X(kpc)')
ax1.set_xlim(-1200, 0)
ax2.set_xlim(0,1200)

plt.savefig('density profile',bbox_inches = 'tight')
plt.show()


# # Imaging
# 

# In[ ]:

# choosing the output file abd loading it
l = 100
curObject = pp.pload(l,w_dir = run_dir) 

#create a meshgrid and changing to cartesian physical units
R, Theta = np.meshgrid(curObject.x1, curObject.x2) 
      X1 = R * np.cos(Theta)
      X2 = R * np.sin(Theta)  
      X1 = X1 * length_scaling.value
      X2 = X2 * length_scaling.value

# Create the figure and adjust the size
fig = plt.figure(figsize=(15,15)) 
  
density_profile = getattr(curObject,'rho').T #change 'rho' to 'prs' or 'vx1' or 'vx2' or 'tr1'    
density_profile = density_profile * density_scaling.value

#Plotting the colormesh 
plt.pcolormesh(X1,X2, np.log10(density_profile),shading='flat')    
plt.xlabel('X(kpc)',fontsize=10)
plt.ylabel('Y(kpc)',fontsize=10)
plt.xlim(-1200, 1200)
plt.ylim(0, 200)

# makes the plot have equal axis, so that the figure doesn't look distorted.
plt.gca().set_aspect('equal', adjustable='box')  

# colorbar
cbaxes = fig.add_axes([0.91, 0.43, 0.02, 0.15])
    cb = plt.colorbar( cax = cbaxes)   
    
cb.set_label(r'$\log(\rho) (g/cm^3)$',fontsize=10)
plt.clim(-26,-30) 

plt.savefig('M25-OFF4R-Q38-t310', bbox_inches = 'tight')       
plt.show()



# # Lobe Dynamics

# Pressure Slice - Recollimation Shocks:

# In[ ]:

# Loading pressure data
globalVar_pressure = getattr(curObject,'prs').T
globalVar_pressure = globalVar_pressure * pressure_scaling.value
    pressure_slice = np.asarray(globalVar_pressure)

ax1.plot(-r_physical, pressure_slice[theta_index_last,:])
ax2.plot(r_physical, pressure_slice[theta_index_first,:])

ax1.set_ylabel('Pressure(Pa)')
ax1.set_xlabel('X(kpc)')
ax2.set_xlabel('X(kpc)')
ax1.set_xlim(-700, 0)
ax2.set_xlim(0,700)
ax1.set_ylim(1*10**(-12), 4.5*10**(-12))

plt.savefig('pressureslices',bbox_inches = 'tight')
plt.show()


# Tracer Slices

# In[ ]:

tracerjet = getattr(curObject,'tr1').T
jet_tracer = np.asarray(tracerjet)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.set_figheight(4)
f.set_figwidth(8)
ax1.plot(-r_physical, jet_tracer[theta_index_last,:])
ax2.plot(r_physical, jet_tracer[theta_index_first,:])

ax1.set_ylabel('Tracer Value')
ax1.set_xlabel('X(kpc)')
ax2.set_xlabel('X(kpc)')
ax1.set_xlim(-1500,0)
ax2.set_xlim(0,1500)
plt.savefig('tracerslices', bbox_inches = 'tight')
plt.show()


# Tracer Thresholds

# In[ ]:

outputs_tr = np.arange(1,100,1)
results_tr = np.zeros((len(outputs_tr), 9))
 trc_cutoff = 0
trc_cutoff0 = 0.001
trc_cutoff1 = 0.01
trc_cutoff2 = 0.1
trc_cutoff3 = 0.3
trc_cutoff4 = 0.5
trc_cutoff5 = 0.7
trc_cutoff6 = 1

for ind, i in enumerate(outputs_tr):
    curObject = pp.pload(i,w_dir = run_dir)
     m = curObject.x1[np.where(curObject.tr1[:,0] >= trc_cutoff)[0][-1]]*length_scaling.value
    m0 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff0)[0][-1]]*length_scaling.value
    m1 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff1)[0][-1]]*length_scaling.value
    m2 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff2)[0][-1]]*length_scaling.value
    m3 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff3)[0][-1]]*length_scaling.value
    m4 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff4)[0][-1]]*length_scaling.value
    m5 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff5)[0][-1]]*length_scaling.value
    m6 = curObject.x1[np.where(curObject.tr1[:,0] == trc_cutoff6)[0][-1]]*length_scaling.value
    #n = curObject.x1[np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]]*length_scaling.value
    results_tr[ind, :] = [m,m0,m1,m2,m3,m4,m5,m6, curObject.SimTime * unit_values.time.value]
    
# Plotting:

fig = plt.figure(figsize=(7,7)) 
#plt.plot(results_tr[:,8], results_tr[:, 0], label = "tr cutoff = 0")
plt.plot(results_tr[:,8], results_tr[:, 1], label = "tr cutoff = 0.001")
plt.plot(results_tr[:,8], results_tr[:, 2], label = "tr cutoff = 0.01")
plt.plot(results_tr[:,8], results_tr[:, 3], label = "tr cutoff = 0.1")
plt.plot(results_tr[:,8], results_tr[:, 4], label = "tr cutoff = 0.3")
plt.plot(results_tr[:,8], results_tr[:, 5], label = "tr cutoff = 0.5")
plt.plot(results_tr[:,8], results_tr[:, 6], label = "tr cutoff = 0.7")
#plt.loglog(results[:,8], results[:, 7], label = "tr cutoff = 1")
plt.ylabel('Lobe Length (kpc)')
plt.xlabel('Simulation Time (Myr)')
plt.legend()
plt.savefig('tracer-thresholds', bbox_inches = 'tight')
plt.show()


# Lobe Length vs Time

# In[ ]:

trc_cutoff = 0.001

nrows,cols = np.shape(jet_tracer)
print('my tracer array is {} by {} in resolution units'.format(nrows,cols))

nrows, ncols = np.shape(jet_tracer)
lobe_cols = []

# creating a function that gives me the length of each lobe. 
def lobe_length(jet_number):           
    
    for col in range(ncols):
        row_values = jet_number[:,col]
        
        for row_value in row_values:
            if row_value > tracer_threshold:
                lobe_cols.append(col)
    return row_values


      jet2 = jet_tracer[0:800, :]
      jet1 = jet_tracer[801:-1, :]

# Finding lobe length at the end of its lifetime
lobe2 = lobe_length(jet2)
lobe2 = curObject.x1[lobe_cols[-1]]*length_scaling.value
print('The length of jet2 is {} kpc'.format(lobe2))

lobe1 = lobe_length(jet1)
lobe1 = curObject.x1[lobe_cols[-1]]*length_scaling.value
print('The length of jet1 is {} kpc'.format(lobe1))

#plotting the temporal evolution of lobe length: 
length_outputs = np.arange(1,100,1)
length_results = np.zeros((len(length_outputs), 3))


for ind, i in enumerate(length_outputs):
    curObject = pp.pload(i,w_dir = run_dir)
    j2 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]]*length_scaling.value
    j1 = curObject.x1[np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]]*length_scaling.value
    length_results[ind, :] = [j2, j1, curObject.SimTime * unit_values.time.value] 
    
#plotting: 
plt.plot(length_results[:,2], length_results[:, 0], label = "jet2")
plt.plot(length_results[:,2], length_results[:, 1], label = "jet1")
plt.ylabel('Lobe Length (kpc)')
plt.xlabel('Simulation Time (Myr)')
plt.legend()
plt.savefig('lobe length', bbox_inches = 'tight')
plt.show()


# Lobe Volume vs Time

# In[ ]:

volume_outputs = np.arange(1,100,1)
volume_results = np.zeros((len(volume_outputs), 3))

for ind, i in enumerate(volume_outputs):
    curObject = pp.pload(i,w_dir = run_dir)
    v = pk.simulations.calculate_cell_volume(curObject)
    t = v*(curObject.tr1 > trc_cutoff)
    tr_v2 = t[:, 0:800]
    tr_v1 = t[:, 800:]
    sum_v2 = np.sum(tr_v2) * (length_scaling.value ** 3)
    sum_v1 = np.sum(tr_v1) * (length_scaling.value ** 3)
    volume_results[ind, :] = [sum_v2, sum_v1, curObject.SimTime * unit_values.time.value]
    
#plotting 
plt.plot(volume_results[:,2], volume_results[:, 0], label = "jet2")
plt.plot(volume_results[:,2], volume_results[:, 1], label = "jet1")
plt.ylabel('Lobe Volume (kp$c^3$)')
plt.xlabel('Simulation Time (Myr)')
plt.legend()
plt.savefig('lobe volume', bbox_inches = 'tight')
plt.show()


# Density-Lobe length Asymmetry

# In[ ]:

# this can be adapted for low-powered jets and 30 degree jets
run_dir_NoOFF = '/Users/Katie/Desktop/PLUTO1/KatiesSims/NonOffset/M25_NoOFF_Q38/'  
run_dir_1R = '/Users/Katie/Desktop/PLUTO1/KatiesSims/Offset/M25_m14.5_OFF1R_Q38/' 
run_dir_4R = '/Users/Katie/Desktop/PLUTO1/KatiesSims/Offset/M25_m14.5_OFF4R_Q38/'


# In[ ]:

# This is calculated for each environment, and then plotted together.
asym_outputs_NoOFF = np.arange(1,100,1)
asym_results_NoOFF = np.zeros((len(asym_outputs_NoOFF), 3))
curObject0 = pp.pload(0, w_dir = run_dir_NoOFF) #Initial timestep where there is no jet

for ind, i in enumerate(asym_outputs_NoOFF):
    curObject = pp.pload(i,w_dir = run_dir_NoOFF) 
    D2 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]] #0 is the smaller side
    X2 = np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]
    D1 = curObject.x1[np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]]
    X1 = np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]
    D_ratio = (D1/D2)
    rho2 = curObject0.rho[X2, 0]
    rho1 = curObject0.rho[X1, -1]
    rho_ratio = (rho1/rho2)
    asym_results_NoOFF[ind, :] = [D_ratio, rho_ratio, curObject.SimTime*unit_values.time.value]

asym_outputs_1R = np.arange(1,100,1)
asym_results_1R = np.zeros((len(asym_outputs_1R), 3))
curObject0 = pp.pload(0, w_dir = run_dir_1R) #Initial timestep where there is no jet

for ind, i in enumerate(asym_outputs_1R):
    curObject = pp.pload(i,w_dir = run_dir_1R) 
    D2 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]] #0 is the smaller side
    X2 = np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]
    D1 = curObject.x1[np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]]
    X1 = np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]
    D_ratio = (D1/D2)
    rho2 = curObject0.rho[X2, 0]
    rho1 = curObject0.rho[X1, -1]
    rho_ratio = (rho1/rho2)
    asym_results_1R[ind, :] = [D_ratio, rho_ratio, curObject.SimTime*unit_values.time.value]

asym_outputs_4R = np.arange(1,100,1)
asym_results_4R = np.zeros((len(asym_outputs_4R), 3))
curObject0 = pp.pload(0, w_dir = run_dir_4R) #Initial timestep where there is no jet

for ind, i in enumerate(asym_outputs_4R):
    curObject = pp.pload(i,w_dir = run_dir_4R) 
    D2 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]] #0 is the smaller side
    X2 = np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]
    D1 = curObject.x1[np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]]
    X1 = np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]
    D_ratio = (D1/D2)
    rho2 = curObject0.rho[X2, 0]
    rho1 = curObject0.rho[X1, -1]
    rho_ratio = (rho1/rho2)
    asym_results_4R[ind, :] = [D_ratio, rho_ratio, curObject.SimTime*unit_values.time.value]
    
#plotting
log_D_NoOFF= np.log10(asym_results_NoOFF[:,0])
log_rho_NoOFF = np.log10(asym_results_NoOFF[:, 1])
   time1 = asym_results_NoOFF[:,2]
      p1 = plt.scatter(log_rho, log_D, c=time1, marker = "x", label='NoOFF')
    cb1 = plt.colorbar(p)
    
log_D_1R = np.log10(asym_results_1R[:,0])
log_rho_1R = np.log10(asym_results_1R[:, 1])
   time2 = asym_results_1R[:,2]
      p2 = plt.scatter(log_rho, log_D, c=time2, marker = "x", label='OFF1R')

log_D_4R = np.log10(asym_results_4R[:,0])
log_rho_4R = np.log10(asym_results_4R[:, 1])
   time3 = asym_results_4R[:,2]
      p3 = plt.scatter(log_rho, log_D, c=time3, marker = "x", label='NoOFF/OFF1R/OFF4R')
        
cb1.set_label(r'Jet Age (Myrs)')
plt.xlabel('log(Density ratio)')
plt.ylabel('log(Length Ratio)')
plt.legend()
plt.savefig('density-lobe length asymmetry', bbox_inches = 'tight')
plt.show()


# Axial Ratio

# In[ ]:

#outputs = np.arange(1,100,1)
#results = np.zeros((len(outputs), 3))


#for ind, i in enumerate(outputs):
#    curObject = pp.pload(i,w_dir = run_dir)
#    j2 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]]*length_scaling.value #jet 2
#    j1 = curObject.x1[np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]]*length_scaling.value #jet 1
#    v = pk.simulations.calculate_cell_volume(curObject)
#    t = v*(curObject.tr1 > trc_cutoff)
#    tr_v2 = t[:, 0:800]  #these have changed from the previous code eh
#    tr_v1 = t[:, 800:]
#    sum_v2 = np.sum(tr_v2) * (length_scaling.value ** 3) 
#    sum_v2 = np.sum(tr_v1) * (length_scaling.value ** 3)  
#    AR2 = ((j2**3)/sum_v2)**0.5
#    AR1 = ((j1**3)/sum_v1)**0.5
#    results[ind, :] = [AR2,AR1, curObject.SimTime * unit_values.time.value] 

#plt.plot(results1[:,2], results1[:, 0], label = "jet2")
#plt.plot(results1[:,2], results1[:, 1], label = "jet1")
#plt.ylabel('Axial Ratio')
#plt.xlabel('Simulation Time (Myr)')
#plt.legend()
#plt.savefig('Axial-Ratio-OFF4R-Q38', bbox_inches = 'tight')
#plt.show()


# # Radio Emission

# Luminosity vs Lobe length

# In[ ]:

lum_outputs = np.arange(1,100,1)
lum_results = np.zeros((len(lum_outputs), 4))
trc_cutoff = 0.001
redshift = 0.1
beam_width = 5 * u.arcsec

for ind, i in enumerate(lum_outputs):
    curObject = pp.pload(i,w_dir = run_dir)
    lum = pk.radio.get_luminosity(curObject, unit_values, redshift, beam_width)
    lum_tr = lum*(curObject.tr1 > trc_cutoff)
    j2 = curObject.x1[np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]]*length_scaling.value
    j1 = curObject.x1[np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]]*length_scaling.value
    lum_tr2 = lum_tr[:, 0:800]
    lum_tr1 = lum_tr[:, 800:]   #long jet
    l2 = np.sum(lum_tr2).value
    l1 = np.sum(lum_tr1).value 
    results[ind, :] = [j2, j1, l2, l1]
    
plt.loglog(results[:,0], results[:, 2], label = "jet2")
plt.loglog(results[:,1], results[:, 3], label = "jet1")
plt.ylabel('Luminosity (W $Hz^{-1}$)')
plt.xlabel('Lobe length (kpc)')
plt.legend()
plt.yticks([10**27, 10**28])
plt.savefig('pd-track-OFF4R-Q38-loglog', bbox_inches = 'tight')
plt.show()


# Density-Luminosity Asymmetry

# In[ ]:

# this can be adapted for low-powered jets and 30 degree jets
run_dir_NoOFF = '/Users/Katie/Desktop/PLUTO1/KatiesSims/NonOffset/M25_NoOFF_Q38/'  
run_dir_1R = '/Users/Katie/Desktop/PLUTO1/KatiesSims/Offset/M25_m14.5_OFF1R_Q38/' 
run_dir_4R = '/Users/Katie/Desktop/PLUTO1/KatiesSims/Offset/M25_m14.5_OFF4R_Q38/'


# In[ ]:

trc_cutoff = 0.001
redshift = 0.1
beam_width = 5 * u.arcsec

# This is calculated for each environment, and then plotted together.
asym_lum_outputs_NoOFF = np.arange(1,100,1)
asym_lum_results_NoOFF = np.zeros((len(asym_lum_outputs_NoOFF), 3))
curObject0 = pp.pload(0, w_dir = run_dir_noOFF)

for ind, i in enumerate(asym_lum_outputs_NoOFF):
    curObject = pp.pload(i,w_dir = run_dir_noOFF)
    lum = pk.radio.get_luminosity(curObject, unit_values, redshift, beam_width)
    lum_tr = lum*(curObject.tr1 > trc_cutoff)
    lum_tr2 = lum_tr[:, 0:800].T  #smaall
    lum_tr1 = lum_tr[:, 800:].T   #long jet
    l2 = np.sum(lum_tr1).value
    l1 = np.sum(lum_tr2).value
    l_ratio = (l1/l2)
    j2 = np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]
    j1 = np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]
    rho2 = curObject0.rho[j2, 0]   #small
    rho1 = curObject0.rho[j1, -1]
    rho_ratio = rho1/rho2
    asym_lum_results_NoOFF[ind, :] = [l_ratio, rho_ratio, curObject.SimTime*unit_values.time.value]
    
    
asym_lum_outputs_1R = np.arange(1,100,1)
asym_lum_results_1R = np.zeros((len(asym_lum_outputs_1R), 3))
curObject0 = pp.pload(0, w_dir = run_dir_1R)

for ind, i in enumerate(asym_lum_outputs_1R):
    curObject = pp.pload(i,w_dir = run_dir_1R)
    lum = pk.radio.get_luminosity(curObject, unit_values, redshift, beam_width)
    lum_tr = lum*(curObject.tr1 > trc_cutoff)
    lum_tr2 = lum_tr[:, 0:800].T  #smaall
    lum_tr1 = lum_tr[:, 800:].T   #long jet
    l2 = np.sum(lum_tr1).value
    l1 = np.sum(lum_tr2).value
    l_ratio = (l1/l2)
    j2 = np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]
    j1 = np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]
    rho2 = curObject0.rho[j2, 0]   #small
    rho1 = curObject0.rho[j1, -1]
    rho_ratio = rho1/rho2
    asym_lum_results_1R[ind, :] = [l_ratio, rho_ratio, curObject.SimTime*unit_values.time.value]
    

asym_lum_outputs_4R = np.arange(1,100,1)
asym_lum_results_4R = np.zeros((len(asym_lum_outputs_4R), 3))
curObject0 = pp.pload(0, w_dir = run_dir_4R)

for ind, i in enumerate(asym_lum_outputs_4R):
    curObject = pp.pload(i,w_dir = run_dir_4R)
    lum = pk.radio.get_luminosity(curObject, unit_values, redshift, beam_width)
    lum_tr = lum*(curObject.tr1 > trc_cutoff)
    lum_tr2 = lum_tr[:, 0:800].T  #smaall
    lum_tr1 = lum_tr[:, 800:].T   #long jet
    l2 = np.sum(lum_tr1).value
    l1 = np.sum(lum_tr2).value
    l_ratio = (l1/l2)
    j2 = np.where(curObject.tr1[:,0] > trc_cutoff)[0][-1]
    j1 = np.where(curObject.tr1[:,-1] > trc_cutoff)[0][-1]
    rho2 = curObject0.rho[j2, 0]   #small
    rho1 = curObject0.rho[j1, -1]
    rho_ratio = rho1/rho2
    asym_lum_results_4R[ind, :] = [l_ratio, rho_ratio, curObject.SimTime*unit_values.time.value]
 

 #plotting 
    
log_L_NoOFF = np.log10(asym_lum_results_NoOFF[:,0])
log_rho_NoOFF = np.log10(asym_lum_results_NoOFF[:, 1])
time1 = asym_lum_results_NoOFF[:,2]
p1 = plt.scatter(log_rho_NoOFF, log_L_NoOFF, c=time1, marker = "x", label='NoOFF')
cb1 = plt.colorbar(p1)

log_L_1R = np.log10(asym_lum_results_1R[:,0])
log_rho_1R = np.log10(asym_lum_results_1R[:, 1])
time2 = asym_lum_results_1R[:,2]
p1 = plt.scatter(log_rho_1R, log_L_1R, c=time2, marker = "x", label='OFF1R')

log_L_4R = np.log10(asym_lum_results_4R[:,0])
log_rho_4R = np.log10(asym_lum_results_4R[:, 1])
time3 = asym_lum_results_4R[:,2]
p1 = plt.scatter(log_rho_4R, log_L_1R, c=time3, marker = "x", label='OFF4R')

cb1.set_label(r'Jet Age (Myrs)')
plt.xlabel('log(Density ratio)')
plt.ylabel('log(Luminosity Ratio)')
plt.legend()
plt.savefig('Density-Luminosity Asymmetry', bbox_inches = 'tight')
plt.show()
    


# # Surface Brightness

# This has been adapted from Patrick Yates code: 

# In[ ]:


# Figure related methods
def figsize(scale, ratio=None):
    fig_width_pt = 240                              # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    if ratio is None:
        ratio = golden_mean
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def newfig(width, ratio=None):
    plt.clf()
    fig = plt.figure(figsize=figsize(width, ratio))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, fig, dpi=100, kwargs=None, png=False):
    if kwargs is None:
        kwargs = {}
    if png == True:
        fig.savefig('{}.png'.format(filename), dpi=dpi, **kwargs)
    else:
        fig.savefig('{}.eps'.format(filename), dpi=dpi, **kwargs)
        fig.savefig('{}.pdf'.format(filename), dpi=dpi, **kwargs)
        
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
        
def create_colorbar(im,
                    ax,
                    fig,
                    size='5%',
                    padding=0.05,
                    position='right',
                    divider=None,
                    use_ax=False):  #pragma: no cover
    if use_ax is False:
        if divider is None:
            divider = _make_axes_locatable(ax)
        cax = divider.append_axes(position, size=size, pad=padding)
    else:
        cax = ax
    ca = fig.colorbar(im, cax=cax)
    cax.yaxis.set_ticks_position(position)
    cax.yaxis.set_label_position(position)
    ca.solids.set_rasterized(True)
    return (ca, divider, cax)

# Surface brightness
def plot_surface_brightness(timestep,
                            unit_values,
                            run_dirs,
                            filename,
                            redshift=0.1,
                            beamsize=5 * u.arcsec,
                            showbeam=True,
                            xlim=(-200, 200),
                            ylim=(-750, 750),   #actually z in arcsec
                            xticks=None,
                            pixel_size=1.8 * u.arcsec,
                            beam_x=0.8,
                            beam_y=0.8,
                            png=True,
                            contours=True,
                            convolve=True,
                            half_plane=True,
                            vmin=-3.0,
                            vmax=2.0,
            
                            no_labels=False,
                            with_hist=False,   #set to false
                            ):
    from plutokore import radio
    from numba import jit
    from astropy.convolution import convolve, Gaussian2DKernel

    @jit(nopython=True)
    def raytrace_surface_brightness(r, theta, x, y, z, raytraced_values, original_values):
        phi = 0
        rmax = np.max(r)
        thetamax = np.max(theta)
        x_half_step = (x[1] - x[0]) * 0.5
        pi2_recip = (1 / (2 * np.pi))

        visited = np.zeros(original_values.shape)
        for x_index in range(len(x)):
            for z_index in range(len(z)):
                visited[:,:] = 0
                for y_index in range(len(y)):
                    # Calculate the coordinates of this point
                    ri = np.sqrt(x[x_index] **2 + y[y_index] ** 2 + z[z_index] ** 2)
                    if ri == 0:
                        continue
                    if ri > rmax:
                        continue
                    thetai = np.arccos(z[z_index] / ri)
                    if thetai > thetamax:
                        continue
                    phii = 0 # Don't care about φi!!

                    chord_length = np.abs(np.arctan2(y[y_index], x[x_index] + x_half_step) - np.arctan2(y[y_index], x[x_index] - x_half_step))

                    # Now find index in r and θ arrays corresponding to this point
                    r_index = np.argmax(r>ri)
                    theta_index = np.argmax(theta>thetai)
                    # Only add this if we have not already visited this cell (twice)
                    if visited[r_index, theta_index] <= 1:
                        raytraced_values[x_index, z_index] += original_values[r_index, theta_index] * chord_length * pi2_recip
                        visited[r_index, theta_index] += 1
        #return raytraced_values
        return

    fig, ax = newfig(1, 1.8)
    #fig, ax = figsize(10,50)

    # calculate beam radius
    sigma_beam = (beamsize / 2.355)

    # calculate kpc per arcsec
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift).to(u.kpc / u.arcsec)

    # load timestep data file
    d = pp.pload(timestep,w_dir = run_dir)

    X1, X2 = pk.simulations.sphericaltocartesian(d)
    X1 = X1 * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value
    X2 = X2 * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value

    l = radio.get_luminosity(d, unit_values, redshift, beamsize)
    f = radio.get_flux_density(l, redshift).to(u.Jy).value
    #sb = radio.get_surface_brightness(f, d, unit_values, redshift, beamsize).to(u.Jy)

    xmax = (((xlim[1] * u.arcsec + pixel_size) * kpc_per_arcsec) / unit_values.length).si
    xstep = (pixel_size * kpc_per_arcsec / unit_values.length).si
    zmax = (((ylim[1] * u.arcsec + pixel_size) * kpc_per_arcsec) / unit_values.length).si
    zstep = (pixel_size * kpc_per_arcsec / unit_values.length).si
    ymax = max(xmax, zmax)
    ystep = min(xstep, zstep)
    ystep = 0.5

    if half_plane:
        x = np.arange(-xmax, xmax, xstep)
        z = np.arange(-zmax, zmax, zstep)
    else:
        x = np.arange(0, xmax, xstep)
        z = np.arange(0, zmax, zstep)
    y = np.arange(-ymax, ymax, ystep)
    raytraced_flux = np.zeros((x.shape[0], z.shape[0]))

    # print(f'xlim in arcsec is {xlim[1]}, xlim in code units is {xlim[1] * u.arcsec * kpc_per_arcsec / unit_values.length}')
    # print(f'zlim in arcsec is {ylim[1]}, zlim in code units is {ylim[1] * u.arcsec * kpc_per_arcsec / unit_values.length}')
    # print(f'xmax is {xmax}, ymax is {ymax}, zmax is {zmax}')
    # print(f'x shape is {x.shape}; y shape is {y.shape}; z shape is {z.shape}')

    raytrace_surface_brightness(
        r=d.x1,
        theta=d.x2,
        x=x,
        y=y,
        z=z,
        original_values=f,
        raytraced_values=raytraced_flux
    )

    raytraced_flux = raytraced_flux * u.Jy

    # beam information
    sigma_beam_arcsec = beamsize / 2.355
    area_beam_kpc2 = (np.pi * (sigma_beam_arcsec * kpc_per_arcsec)
                      **2).to(u.kpc**2)
    beams_per_cell = (((pixel_size * kpc_per_arcsec) ** 2) / area_beam_kpc2).si
    #beams_per_cell = (area_beam_kpc2 / ((pixel_size * kpc_per_arcsec) ** 2)).si

    # radio_cell_areas = np.full(raytraced_flux.shape, xstep * zstep) * (unit_values.length ** 2)

    # n beams per cell
    #n_beams_per_cell = (radio_cell_areas / area_beam_kpc2).si

    raytraced_flux /= beams_per_cell

    stddev = sigma_beam_arcsec / beamsize
    beam_kernel = Gaussian2DKernel(stddev)
    if convolve:
        flux = convolve(raytraced_flux.to(u.Jy), beam_kernel, boundary='extend') * u.Jy
    else:
        flux = raytraced_flux.to(u.Jy)
    #flux = radio.convolve_surface_brightness(raytraced_flux, unit_values, redshift, beamsize)
    #flux = raytraced_flux
    
    #return (x, z, flux) # x_coords, z_coords, surfb = plot_surface_brightness(...)

    X1 = x * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value
    X2 = z * (unit_values.length / kpc_per_arcsec).to(u.arcsec).value
    
    return (X1, X2, flux) # x_coords, z_coords, surfb = plot_surface_brightness(...)

    # plot data keep
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    contour_color = 'k'
    contour_linewidth = 0.33
    #contour_levels = [-3, -1, 1, 2] 
    contour_levels = [-2, -1, 0, 1, 2] # Contours start at 10 μJy

    #with plt.style.context('flux-plot.mplstyle'): keep
    im = ax.pcolormesh(
            X1,
             X2,
            np.log10(flux.to(u.mJy).value).T,
            shading='flat',
            vmin=vmin,
            vmax=vmax)
    im.set_rasterized(True)
    if contours:
            ax.contour(X1, X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

    im = ax.pcolormesh(
            -X1,
            X2,
            np.log10(flux.to(u.mJy).value).T,
            shading='flat',
            vmin=vmin,
            vmax=vmax)
    im.set_rasterized(True)
    if contours:
            ax.contour(-X1, X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

    if not half_plane:
        im = ax.pcolormesh(
                X1,
                -X2,
                np.log10(flux.to(u.mJy).value).T,
                shading='flat',
                vmin=vmin,
                vmax=vmax)
        im.set_rasterized(True)
        if contours:
                ax.contour(X1, -X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

        im = ax.pcolormesh(
                -X1,
                -X2,
                np.log10(flux.to(u.mJy).value).T,
                shading='flat',
                vmin=vmin,
                vmax=vmax)
        im.set_rasterized(True)
        if contours:
                ax.contour(-X1, -X2, np.log10(flux.to(u.mJy).value).T, contour_levels, linewidths=contour_linewidth, colors=contour_color)

    if with_hist:
        div = make_axes_locatable(ax)   #from mpl_toolkits.axes_grid1 import make_axes_locatable
        ax_hist = div.append_axes('right', '30%', pad=0.0)
        s = np.sum(flux.to(u.mJy).value, axis=0)
        ax_hist.plot(np.concatenate([s, s]), np.concatenate([X2, -X2]))
        ax_hist.set_yticklabels([])

    if not no_labels:
        (ca, div, cax) = create_colorbar(
            im, ax, fig, position='right', padding=0.5)
        ca.set_label(r'$\log_{10}\mathrm{mJy / beam}$')

    circ = plt.Circle(
        (xlim[1] * beam_x, ylim[0] * beam_y),
        color='w',
        fill=True,
        radius=sigma_beam.to(u.arcsec).value,
        alpha=0.7)
    #circ.set_rasterized(True)

    if showbeam:
        ax.add_artist(circ)

    # reset limits
    if not no_labels:
        ax.set_xlabel('X ($\'\'$)')
        ax.set_ylabel('Y ($\'\'$)')
    ax.set_aspect('equal')

    if xticks is not None:
        ax.set_xticks(xticks)

    if no_labels:
        ax.set_position([0, 0, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

    ax.set_aspect('equal')

    if no_labels:
        savefig(filename, fig, png=png, kwargs={
            'bbox_inches': 'tight',
            'pad_inches': 0}
                )
    else:
        savefig(filename, fig, png=png, dpi = 300)
    plt.show()
    #plt.close();


# In[ ]:

filename = 'surf_bright'
timestep = 100
run_dirs = run_dir
SB = plot_surface_brightness(timestep, unit_values, run_dirs, filename)
print SB

x_coords, z_coords, surfb = plot_surface_brightness(timestep,unit_values,run_dirs,filename)

flat_index_1 = np.argmax(surfb[:,:surfb.shape[1]/2])
(xi_1,yi_1) = np.unravel_index(flat_index_1, surfb[:,:surfb.shape[1]/2].shape)

print x_coords[xi_1]
print z_coords[yi_1]

flat_index_2 = np.argmax(surfb[:,surfb.shape[1]/2:])
(xi_2,yi_2) = np.unravel_index(flat_index_2, surfb[:,surfb.shape[1]/2:].shape)

print x_coords[xi_2]
print z_coords[yi_2]

