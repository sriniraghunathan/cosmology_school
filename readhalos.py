import healpy as hp
from   cosmology import *

import os
totthreads = 6
os.putenv('OMP_NUM_THREADS',str(totthreads))

rho = 2.775e11*omegam*h**2 # Msun/Mpc^3

f=open('halos.pksc')
##f=open('halos-light.pksc')
N=np.fromfile(f,count=3,dtype=np.int32)[0]

# only take first five entries for testing (there are ~8e8 halos total...)
# comment the following line to read in all halos
N = 60000000 ###59890112  ##862923143 ##5000

catalog=np.fromfile(f,count=N*10,dtype=np.float32)
catalog=np.reshape(catalog,(N,10))

'''
catalog_full=np.fromfile(open('halos.pksc'),count=N*10,dtype=np.float32)
catalog_full=np.reshape(catalog_full,(N,10))

catalog_light=np.fromfile(open('halos-light.pksc'),count=N*10,dtype=np.float32)
catalog_light=np.reshape(catalog_light,(N,10))
print(catalog_light[0], catalog[0])
sys.exit()
catalog = catalog_light
'''

x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
vx = catalog[:,3]; vy = catalog[:,4]; vz = catalog[:,5] # km/sec
R  = catalog[:,6] # Mpc

# convert to mass, comoving distance, radial velocity, redshfit, RA and DEc
M200m    = 4*np.pi/3.*rho*R**3        # this is M200m (mean density 200 times mean) in Msun
chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
redshift = zofchi(chi)      

#srini adding - write new catalogue
import numpy as np, pandas as pd
'''
Mmin, Mmax = 1e13, 1e17
zmin, zmax = -1, 10000
inds = np.where( (M200m>=Mmin) & (M200m<=Mmax) & (redshift>=zmin) & (redshift<=zmax))[0]
#inds = np.where( (M200m>=Mmin) & (M200m<=Mmax) & (redshift>=zmin) & (redshift<=zmax) & ())[0]
print(len(inds))
'''

'''
### e.g. project to a map, matching the websky orientations
nside = 2048
HMAP   = np.zeros((hp.nside2npix(nside)))

pix = hp.vec2pix(nside, x, y, z)
#pix = hp.ang2pix(nside, theta, phi) does the same
'''

#weight = 1. #1 for number density, array of size(x) for arbitrary
#np.add.at(HMAP, pix, weight)

#add 3G mask
#MASK = hp.read_map('/data19/sri/3g_stuffs/MASK.fits')
#inds = np.where(MASK!=0.)[0]

#Mmin, Mmax = 1e13, 1e17
#zmin, zmax = -1, 10000

Mmin, Mmax = 2.5e13, 5e13
zmin, zmax = 0.4, 0.6

inds = np.where( (M200m>=Mmin) & (M200m<=Mmax) & (redshift>=zmin) & (redshift<=zmax))[0]
x,y,z = x[inds], y[inds], z[inds]
vx,vy,vz = vx[inds], vy[inds], vz[inds]
M200m = M200m[inds]
redshift = redshift[inds]


if (1):
    theta, phi  = hp.vec2ang(np.column_stack((x,y,z))) # in radians
    phi = np.degrees(phi)
    theta = np.degrees(theta)
    phi = np.round_(phi, 4)
    theta = np.round_(theta, 4)


M200m = M200m/1e14
M200m = np.round_(M200m, 4)
redshift = np.round_(redshift, 3)
vx = np.round_(vx, 3)
vy = np.round_(vy, 3)
vz = np.round_(vz, 3)

#headerval = ['phi', 'theta', 'z', 'M200m', 'vx', 'vy', 'vz']#, 'x', 'y', 'z']
headerval = ['phi', 'theta', 'z', 'M200m', 'vx', 'vy', 'vz', 'x', 'y', 'z']
#cat = [phi[inds], theta[inds], redshift[inds], M200m[inds]]

opfname = 'extracted_haloes_Mmin%g_Mmax%g_zmin%g_zmax%g.csv' %(Mmin/1e14, Mmax/1e14, zmin, zmax)
print('\n\toutputfile is %s: total inds = %s\n\n' %(opfname, len(inds)))
#np.save(opfname, cat, header=headerval)

df = pd.DataFrame({'phi' : phi, 'theta' : theta, 'redshift': redshift, 'M200m': M200m, 'vx': vx, 'vy': vy, 'vz': vz, 'x': x, 'y': y, 'z': z})
df.to_csv(opfname, index=False)


