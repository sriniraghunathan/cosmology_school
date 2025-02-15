import numpy as np, sys, os, warnings
import tools
from pylab import *

################################################################################################
################################################################################################
################################################################################################

def make_gaussian_realisation(flatskymapparams, el, cl_dic, theory_is_1d_or_2D = '1d'):

    """
    return correlated realisations of flat sky maps.

    input:
    flatskymapparams = [ny, nx, dx] where ny, nx = flatskymap.shape; and dx is the pixel resolution in arcminutes.
    for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = 0.5 arcminutes.

    el: Multipole range over which the theory cls are defined.
    cl_dic: theory spectra. Either 1d or 2D. If 1d, 2D will be obtained using interpolation.
        Keys must be 00, 11, 01 for 2 maps; 
        Keys must be 00, 11, 22, 01, 02, 12 for 3 maps; and so on.
        For example, for Chris' noise sims, they will be
        00 - 90x90 noise auto.
        11 - 150x150 noise auto.
        22 - 220x220 noise auto.
        01 - 90x150 noise cross.
        02 - 90x220 noise cross.
        12 - 150x220 noise cross.
    theory_is_1d_or_2D: specificy if the theory cl_dic 1d or 2d.

    output:
    correlated sim maps: array
        returns N maps for N(N+1)/2 spectra.
    """

    #----------------------------------------
    #solve quadratic equation to get the number of maps
    """
    For "N" maps, we will have N (N +1)/2 = total spectra which is total_spec
    (N^2 + N)/2 = total_spec
    N^2 + N - 2 * total_spec = 0
    a = 1, b = 1, c = - (2 * total_spec)
    solution is: N = ( -b + np.sqrt( b**2 - 4 * a * c) ) / (2 * a)
    """
    total_spec = len( cl_dic )
    a, b, c = 1, 1, -2 * total_spec
    total_maps = int( ( -b + np.sqrt( b**2 - 4 * a * c) ) / (2 * a) )
    assert total_maps == int(total_maps)
    #----------------------------------------

    #----------------------------------------
    #map stuff
    nx, ny, dx, dy = flatskymapparams
    arcmins2radians = np.radians(1/60.)

    dx *= arcmins2radians
    dy *= arcmins2radians

    #norm stuff of maps
    norm = np.sqrt(1./ (dx * dy))
    #----------------------------------------

    #----------------------------------------
    #gauss reals
    gauss_reals_fft_arr = []
    for iii in range(total_maps):
        curr_gauss_reals = np.random.standard_normal([nx,ny])
        curr_gauss_reals_fft = np.fft.fft2( curr_gauss_reals )
        gauss_reals_fft_arr.append( curr_gauss_reals_fft )
    gauss_reals_fft_arr = np.asarray( gauss_reals_fft_arr )
    #----------------------------------------

    #----------------------------------------
    if theory_is_1d_or_2D == '1d': #1d to 2D spec
        cl_twod_dic = {}
        for ij in cl_dic:
            i, j = ij
            curr_cl_twod = cl_to_cl2d(el, cl_dic[(i,j)], flatskymapparams)
            cl_twod_dic[(i,j)] = cl_twod_dic[(j,i)] = np.copy( curr_cl_twod )
    else:
        cl_twod_dic = {}
        for ij in cl_dic:
            i, j = ij
            cl_twod_dic[(i,j)] = cl_twod_dic[(j,i)] = np.copy( cl_dic[(i,j)] )

    if (0):
        for ij in cl_twod_dic:
            imshow( np.fft.fftshift(cl_twod_dic[ij]) ); colorbar(); show()
        sys.exit()

    #----------------------------------------
    
    #----------------------------------------
    #get FFT amplitudes of reals now. Appendix of https://arxiv.org/pdf/0801.4380
    map_index_combs = []
    for i in range(total_maps):
        for j in range(total_maps):
            key = [j, i]
            key_rev = [i, j]
            if key_rev in map_index_combs: continue            
            map_index_combs.append( key )

    tij_dic = {}
    for ij in map_index_combs:
        i, j = ij
        kinds = np.arange(j)
        if i == j:
            t1 = cl_twod_dic[(i,j)]
            t2 = np.zeros( (ny, nx) )
            ##print(ij, i, j, kinds)
            for k in kinds:
                #print(i, j, k); sys.exit()
                t2 = t2 + tij_dic[(i,k)]**2.
            tij_dic[(i,j)] = tij_dic[(j,i)]= np.sqrt( t1-t2 )
        elif i>j:
            t1 = cl_twod_dic[(i,j)]
            t2 = np.zeros( (ny, nx) )
            for k in kinds: #range(j-1):
                t2 += tij_dic[(i,k)] * tij_dic[(j,k)]
            ##print( tij_dic.keys() ); sys.exit()
            t3 = tij_dic[(j,j)]
            #tij_dic[(i,j)] = tij_dic[(j,i)] = (t1-t2)/t3
            tij_dic[(i,j)] = (t1-t2)/t3 ###check this again: if (j,i) is not needed
    ##print( tij_dic.keys() ); sys.exit()
    for ij in tij_dic: #remove nans
        tij_dic[ij][np.isnan(tij_dic[ij])] = 0.
    ##print(tij_dic.keys()); sys.exit()

    #----------------------------------------
                
    #----------------------------------------
    #FFT amplitudes times gauss reals and ifft back
    sim_maps = []
    for i in range(total_maps):
        '''
        if i == 0:
            curr_map_fft = gauss_reals_fft_arr[i] * tij_dic[(i,i)]            
        else:
            curr_map_fft = np.zeros( (ny, nx) )
            for a in range(total_maps): #loop over tij_dic
                if a>i+1: continue
                curr_map_fft = curr_map_fft + gauss_reals_fft_arr[a] * tij_dic[(i,a)]
        curr_map_fft = curr_map_fft * norm
        curr_map = np.fft.ifft2( curr_map_fft ).real
        curr_map = curr_map - np.mean( curr_map )
        sim_maps.append( curr_map )
        '''
        if i == 0:
            curr_map_fft = gauss_reals_fft_arr[i] * tij_dic[(i,i)]            
        else:
            curr_map_fft = np.zeros( (ny, nx) )
            for a in range(total_maps): #loop over tij_dic
                #if a>i+1: continue #check this again
                if a>i: continue
                curr_map_fft = curr_map_fft + gauss_reals_fft_arr[a] * tij_dic[(i,a)]
        curr_map_fft = curr_map_fft * norm
        curr_map = np.fft.ifft2( curr_map_fft ).real
        curr_map = curr_map - np.mean( curr_map )
        sim_maps.append( curr_map )
    #----------------------------------------

    sim_maps = np.asarray( sim_maps )
    return sim_maps    


def get_lxly(flatskymapparams):

    """
    return lx, ly modes (kx, ky Fourier modes) for a flatsky map grid.
    """
    
    ny, nx, dx, dx = flatskymapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

def cl_to_cl2d(el, cl, flatskymapparams):
    
    """
    Interpolating a 1d power spectrum (cl) defined on multipoles (el) to 2D assuming azimuthal symmetry (i.e:) isotropy.
    """
    
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    cl2d = np.interp(ell.flatten(), el, cl, left = 0., right = 0.).reshape(ell.shape)
    return cl2d

def circular_apod_mask(ra_grid, dec_grid, mask_radius, apod_mask = 1, perform_apod = True):

    import scipy as sc
    import scipy.ndimage as ndimage

    radius = np.sqrt( (ra_grid**2. + dec_grid**2.) )

    if apod_mask:
        mask = np.zeros( dec_grid.shape )
    else:    
        mask = np.ones( dec_grid.shape )

    inds_to_mask = np.where((radius<=mask_radius)) #2arcmins - fix this for now
    if apod_mask:
        mask[inds_to_mask[0], inds_to_mask[1]] = 1.
    else:
        mask[inds_to_mask[0], inds_to_mask[1]] = 0.

    dx_grid = np.diff(ra_grid)[0][0]
    taper_radius = mask_radius * 6.
    if perform_apod:
        ##imshow(mask); colorbar(); show(); sys.exit()
        ker=np.hanning(taper_radius)
        ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )
        mask=ndimage.convolve(mask, ker2d)
        mask/=mask.max()

    return mask


def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, minbin = 0, maxbin = 15000, binsize = 100, mask = None, filter_2d = None):

    """
    map2cl module - get the power spectra of map/maps

    input:
    flatskymapparams = [ny, nx, dx] where ny, nx = flatskymap.shape; and dx is the pixel resolution in arcminutes.
    for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = 0.5 arcminutes.

    flatskymap1: map1 with dimensions (ny, nx)
    flatskymap2: provide map2 with dimensions (ny, nx) cross-spectra

    binsize: el bins. computed automatically if None

    cross_power: if set, then compute the cross power between flatskymap1 and flatskymap2

    output:
    auto/cross power spectra: [el, cl, cl_err]
    """

    ny, nx, dx, dx = flatskymapparams
    dx_rad = np.radians(dx/60.)

    lx, ly = get_lxly(flatskymapparams)

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)

    rad_prf = radial_profile(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = minbin, maxbin = maxbin, to_arcmins = 0)
    el, cl = rad_prf[:,0], rad_prf[:,1]

    if mask is not None:
        fsky = np.mean(mask**2.)
        cl /= fsky

    if filter_2d is not None:
        rad_prf_filter_2d = radial_profile(filter_2d, (lx,ly), bin_size = binsize, minbin = minbin, maxbin = maxbin, to_arcmins = 0)
        el, fl = rad_prf_filter_2d[:,0], rad_prf_filter_2d[:,1]
        cl /= fl

    return el, cl

################################################################################################################

def radial_profile(z, xy = None, bin_size = 1., minbin = 0., maxbin = 10., to_arcmins = 1):

    """
    get the radial profile of an image (both real and fourier space).
    """

    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(image.shape)
    else:
        x, y = xy

    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
    if to_arcmins: radius *= 60.

    binarr=np.arange(minbin,maxbin,bin_size)
    radprf=np.zeros((len(binarr),3))

    hit_count=[]

    for b,bin in enumerate(binarr):
        ind=np.where((radius>=bin) & (radius<bin+bin_size))
        radprf[b,0]=(bin+bin_size/2.)
        hits = len(np.where(abs(z[ind])>0.)[0])

        if hits>0:
            radprf[b,1]=np.sum(z[ind])/hits
            radprf[b,2]=np.std(z[ind])
        hit_count.append(hits)

    hit_count=np.asarray(hit_count)
    std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
    errval=std_mean/(hit_count)**0.5
    radprf[:,2]=errval

    return radprf

################################################################################################################
################################################################################################################
################################################################################################################
