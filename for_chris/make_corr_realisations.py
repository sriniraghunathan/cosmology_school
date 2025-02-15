import numpy as np, sys, os, warnings
import tools

from pylab import *
cmap = cm.RdYlBu_r

#----------------------------------------
#get params for sim generation
band_arr = [90, 150, 220]
noise_dic = {90: 3, 150: 2.8, 220: 8.8}
elknee, alphaknee = 3000., 5.
rho = 0.9 ##0. ##0.9 #xx per cent corr atm noise for the crosses.
lmin = 10 #remove some largest-scale modes

lmax = 10000
els = np.arange(lmax)
boxsize_arcmins, dx = 300., 1.0 #arcminutes
total_sims = 5
#----------------------------------------

#----------------------------------------
#get ra, dec or map-pixel grid
nx = int(boxsize_arcmins/dx)
mapparams = [nx, nx, dx, dx]
x1, x2 = -nx/2. * dx, nx/2. * dx
x1_deg, x2_deg = x1/60., x2/60.

ra_for_grid = np.linspace(x1,x2, nx)/60.
dec_for_grid = np.linspace(x1,x2, nx)/60.
ra_grid, dec_grid = np.meshgrid(ra_for_grid,dec_for_grid)
#----------------------------------------

#----------------------------------------
#define theory spectra now
nl_dic = {}

#autos first
nl_atm_dic = {}
for band in band_arr:

    #white
    delta_T_radians = noise_dic[band] * np.radians(1./60.)
    curr_nl_white = np.tile(delta_T_radians**2., len(els) )

    #atm
    curr_nl_atm = np.copy(curr_nl_white) * (elknee * 1./els)**alphaknee
    curr_nl_atm[np.isinf(curr_nl_atm) | np.isnan(curr_nl_atm) ] = 0.
    nl_atm_dic[(band, band)] = curr_nl_atm

    #total
    curr_nl = curr_nl_white + curr_nl_atm
    curr_nl[np.isinf(curr_nl) | np.isnan(curr_nl) ] = 0.
    nl_dic[(band, band)] = curr_nl

#crosses
for band1 in band_arr:
    for band2 in band_arr:
        if band1 == band2: continue
        curr_nl_atm_band1 = nl_atm_dic[(band1, band1)]
        curr_nl_atm_band2 = nl_atm_dic[(band2, band2)]
        curr_nl_atm_band1_band2 = rho * np.sqrt( curr_nl_atm_band1 * curr_nl_atm_band2 )
        ##print( curr_nl_atm_band1 ); 
        ##print( curr_nl_atm_band2 )
        ##print( curr_nl_atm_band1_band2 )
        ##sys.exit()

        nl_dic[(band1, band2)] = curr_nl_atm_band1_band2

for b1b2 in nl_dic: #remove nans, inf, and largest-scale modes
    curr_nl = nl_dic[b1b2]
    curr_nl[np.isnan(curr_nl) | np.isinf(curr_nl)] = 0.
    nl_dic[b1b2] = curr_nl
    nl_dic[b1b2][els<lmin] = 0.
#----------------------------------------

#----------------------------------------
#make sims now
nl_dic_for_sims = {}
for b1cntr, band1 in enumerate( band_arr ):
    for b2cntr, band2 in enumerate( band_arr ):
        if band2<band1: continue
        nl_dic_for_sims[(b1cntr, b2cntr)] = nl_dic[(band1, band2)]
print(nl_dic_for_sims.keys()); ##sys.exit()

nl_dic_sims = {}
for sim_no in range( total_sims ):
    print('Sim = %s of %s' %(sim_no+1, total_sims))
    noise_sim_arr = tools.make_gaussian_realisation(mapparams, els, nl_dic_for_sims, theory_is_1d_or_2D = '1d')
    
    if sim_no == 0:
        for cntr in range( len(noise_sim_arr) ):
            subplot(1, len(noise_sim_arr), cntr+1); imshow(noise_sim_arr[cntr]); colorbar()
        show(); ##sys.exit()
    print( '\t sim shape = ', noise_sim_arr.shape ); ##sys.exit()

    #get the sim spectra now.
    curr_sim_nl_dic = {}
    for (band1, map1) in zip( band_arr, noise_sim_arr ):
        for (band2, map2) in zip( band_arr, noise_sim_arr ):
            if band2<band1: continue
            els_sim, cl_sim = tools.map2cl(mapparams, map1, map2)
            curr_sim_nl_dic[(band1, band2)] = [els_sim, cl_sim]
    nl_dic_sims[sim_no] = curr_sim_nl_dic
#----------------------------------------

#----------------------------------------
#make plots
band_combs_for_plotting_autos = [[band, band] for band in band_arr]
band_combs_for_plotting_crosses = [[band1, band2] for band1 in band_arr for band2 in band_arr if (band1!=band2 and band1<band2)]
band_combs_for_plotting = band_combs_for_plotting_autos + band_combs_for_plotting_crosses

close('all')
clf()
fsval = 14
figure = figure( figsize = (11., 6.) )
tr, tc = 2, 3 #6 spectra in total
subplots_adjust( hspace = 0.2, wspace = 0.1)
sbpl = 1
dl_fac = els * (els+1)/2/np.pi
for b1b2 in band_combs_for_plotting:
    b1, b2 = b1b2
    curr_nl = nl_dic[(b1, b2)]
    
    ax = subplot( tr, tc, sbpl, yscale = 'log')

    #theory
    plot( els, dl_fac * curr_nl, color = 'black', label = r'Input')

    #sims
    colorval = 'orangered'
    curr_nl_arr = []
    for sim_no in nl_dic_sims:
        curr_el, curr_nl = nl_dic_sims[sim_no][(b1, b2)]
        curr_nl_arr.append( curr_nl )
        curr_dl_fac = curr_el * (curr_el+1)/2/np.pi
        plot( curr_el, curr_dl_fac * curr_nl, color = colorval, lw = 0.2, alpha = 0.5)
    curr_nl_mean = np.mean( curr_nl_arr, axis = 0 )
    plot( curr_el, curr_dl_fac * curr_nl_mean, color = colorval, label = r'Sim mean')


    xlim(0., lmax+10); ylim(0.1, 3e3)
    if (sbpl-1) % tc == 0:
        ylabel(r'Spectra $D_{\ell}$ [$\mu$K$^{2}$]', fontsize = fsval)
    else:
        setp(ax.get_yticklabels(), visible=False);# tick_params(axis='y',left='off')

    if sbpl == 1:
        legend(loc = 1, fontsize = fsval - 2)

    if sbpl>=tc:
        xlabel(r'Multipole $\ell$', fontsize = fsval)
    else:
        setp(ax.get_xticklabels(), visible=False);# tick_params(axis='y',left='off')

    sbpl += 1

show()



#----------------------------------------

