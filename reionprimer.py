"""
Collection of the routines for calculating the reionization models.


by Alexander Kaurov
"""


import numpy as np
from numpy import sin, cos, pi
from scipy.interpolate import griddata
from scipy.stats import rankdata, norm


# FastPM related libraries to work with snapshots
from nbodykit.source.catalog.file import BigFileCatalog
from nbodykit.source.mesh import BigFileMesh

from nbodykit.cosmology import Planck15
from nbodykit.cosmology import EHPower

from nbodykit.lab import *
from nbodykit import setup_logging, style




# ------------------------------
# Wrapers around fastpm routines
# ------------------------------

def get_snapshot(fname = 'fastpm', a = 0.1250, N = 256):
    """ Load snapshot and project onto mesh
    TODO: documentation
    """
    print('Reading snapshot: %s_%0.04f'%(fname,a))
    part = BigFileCatalog('%s_%0.04f'%(fname,a), dataset='1/', header='Header')
    q = part.to_mesh(Nmesh=N)
    q = q.to_field()
    return q

def get_halo_field(fname = 'fastpm', a = 0.1250, N = 256, boxsize=10., filt_mass=1e7, f = lambda x: np.sqrt(x)):
    """ Load halos and project them onto mesh
    TODO: documentation
    """
    print('Reading halos from: %s_%0.04f/fof/0.200/'%(fname,a))
    halos = BigFileCatalog('%s_%0.04f/fof/0.200/'%(fname,a),header='Header')
    M = np.array(halos['Mass'])
    pos = np.array(halos['CMPosition'])
    # we want to take only halos above filt_mass in M_\Sun
    filt = (M>filt_mass)
    halo_field_m = np.zeros([N,N,N])
    for i,(m,x,y,z) in enumerate(zip(M[filt],
                                     (pos[filt,0])/boxsize*N, 
                                     (pos[filt,1])/boxsize*N, 
                                     (pos[filt,2])/boxsize*N)):
        halo_field_m[int(x),int(y),int(z)] += f(m)
    return halo_field_m


# ------------------------------
# Excursion set formalism routines
# ------------------------------

def sph_filter(r, N, mode, f = lambda x, y: x**2+y):
    ''' Constructs spherical filter in the real space.
    Args:
      r:     parameter
      N:     size of the grid
      mode:  tophat    Classical tophat (just a shpere), r is its radius
             gaussian  Gaussian profile, r is its STD
             r^-2      1/(x+r)^2, r is the softening scale
             custom    Provide your own lambda function f
      f:     function, i.e. (f = lambda x, y: x**2+y)
    
    Returns:
      NxNxN array of the filter in the real space normalized to Sum=1
    '''
    x,y,z = np.mgrid[:N,:N,:N]
    x[x>N/2] = N - x[x>N/2]
    y[y>N/2] = N - y[y>N/2]
    z[z>N/2] = N - z[z>N/2]
    r2 = (x**2+y**2+z**2)
    if mode == 'tophat':
        temp = r2<r**2
    elif mode == 'gaussian':
        temp = np.exp(-0.5*(r2/r**2))
    elif mode == 'r^-2':
        temp = 1/(r2**0.5+r)**2
    elif mode == 'custom':
        temp = f(np.sqrt(r2), r)
    return temp / np.sum(temp)

def ft_sph_filter(r, N, mode, f = lambda x, y: x**2+y):
    """ Constructs spherical filter in the Fourier space.
    Args:
      r:     parameter
      N:     size of the grid
      mode:  tophat    Classical tophat (just a shpere), r is its radius
             gaussian  Gaussian profile, r is its STD
             r^-2      1/(x+r)^2, r is the softening scale
             custom    Provide your own lambda function f
      f:     function, i.e. (f = lambda x, y: x**2+y)
    
    Returns:
      NxNxN/2 array of the filter in the Fourier space normalized to Sum=1
    """
    return np.fft.rfftn(sph_filter(r, N, mode, f))

def smooth(ft_data, r, N, mode, data_in_fourier_space = True, f = lambda x, y: x**2+y):
    """ Smoothing data.
    !!!! By default data is already in Fourier space !!!!
    Args:
      ft_data:   data (in the Fourier space or Real space)
      r:         smoothing parameter
      N:         size of the data cube
      mode:      
      data_in_fourier_space: 
      f: 
    """
    if data_in_fourier_space:
        return np.fft.irfftn(ft_sph_filter(r, N, mode, f) * ft_data)
    else:
        return np.fft.irfftn(ft_sph_filter(r, N, mode, f) * np.fft.rfftn(ft_data))

def get_trajectories(field, 
                     sm_scales=np.array([1,2,4,8,16,32,64,128,256], dtype=float), 
                     filter_mode='gaussian', 
                     normalization='None'):
    """Generating 'trajectories' for the 3D field

    Args:
        field (3d float array):  Original neutral fraction field of IGM.
        sm_scales (float array): List of smoothing scales.
        filter_mode     (str):   Coefficient.
        normalization (str):     For each smoothing scale we can normalize the 
                                 resulting smooth field. Options are: 
                                 'None' - do nothing.
                                 'rankorder' - assign to the cell its order 
                                              (result is the uniform distribution 
                                              between 0 and 1)
                                 'gaussian' - same as above, but the distribution 
                                              is normal with mean = 0 and std = 1

    Returns:
        res: [N x N x N x len(sm_scales)] array of trajectories.

    """
    f_field = np.fft.rfftn(field)
    N = field.shape[0]
    res = np.zeros([N,N,N,len(sm_scales)])
    for i,scale in enumerate(sm_scales):
        res[:,:,:,i] = smooth(f_field, scale, N, mode=filter_mode)
        if normalization == 'rankorder':
            res[:,:,:,i] = rankdata(res[:,:,:,i].reshape([N**3]), method='ordinal').reshape([N,N,N])
            res[:,:,:,i] = res[:,:,:,i]/(N**3-1)
        if normalization == 'gaussian':
            res[:,:,:,i] = rankdata(res[:,:,:,i].reshape([N**3]), method='ordinal').reshape([N,N,N])
            res[:,:,:,i] = res[:,:,:,i]/(N**3-1)
            res[:,:,:,i] = norm.ppf(res[:,:,:,i], loc=0, scale=1)
            #             res[:,:,:,i] -= res[:,:,:,i].mean()
            #             res[:,:,:,i] /= res[:,:,:,i].std()
    return res

def apply_reverse_barrier(trajectories, sm_scales, boxsize, r_barrier = lambda s,v: v-4*(np.log10(s)-.6)**2):
    """Apllying a 'reverse' barrier 
    
    TODO: documentation
    """
    N = trajectories.shape[0]
    h_field = np.zeros([N,N,N,len(sm_scales)])
    for i,scale in enumerate(sm_scales):
        h_field[:, :, :, i] = r_barrier(N/scale/boxsize, trajectories[:,:,:,i])
    h_field = h_field.max(3)
    return h_field

def convert_to_h_field(h_field, z_list, f_list):
    """Converting any ordered field to match given reionization field.
    
    Args:
      h_field (3d float array): Any ordered field
      z_list  (3d float array):  Redshifts of the ionization history
      f_list  (3d float array):  Ionizations of the ionization history
    
    
    Returns:
      f_field   (3d float array): ranked array.
      z_field   (3d float array): redshifts.
    """
    N = h_field.shape[0]
    f_field = 1.0-(rankdata(h_field) / N**3).reshape([N,N,N])
    z_field = np.interp(f_field.flat, f_list, z_list).reshape([N,N,N])
    return f_field, z_field

def reion_history(z):
    """ A naive eionization history
    
    Args:
      z (float): Redhsift
    
    Returns:
      Ionized fraction at z
    """
    res = np.exp((6-z)/1.)
    res[z<6] = 1.0
    #     res = np.arctan((8.-z)*5.)/np.pi+0.5
    return res

# ------------------------------
# Routines for adding filaments
# ------------------------------

def add_filaments_with_proximity(fHI_IGM, density, flux, n_c=2.5):
    """Adding filaments to the simulation

    Under the assumption that the filaments are resolved and optically
    thin to the ionizing background, we can apply a simple rule for
    estimating the neutral fraction. Naively you can think :

    Args:
        fHI_IGM (3d float array): Original neutral fraction field of IGM.
        density (3d float array): Density.
        flux    (3d float array or float): Ionization background if 3d array,
                                            constant bg in case of scalar.
        n_c     (float):          Coefficient.

    Returns:
        fHI_mod: Neutral fraction field that includes filaments.

    """
    fHI_mod = fHI_IGM.copy()
    fHI_mod += 1.0-n_c*q**-0.5*radiation_field**0.5
    fHI_mod[fHI_IGM==1]=1
    fHI_mod[fHI_mod>1] = 1
    fHI_mod[fHI_mod<0] = 0
    return fHI_mod

# ------------------------------
# Routines for adding stochasticity
# ------------------------------

def generate_deflection_potential(N, q=0, q_factor=1, mode='gaussian', parameters=[]):
    """Generating deflection potential
    
    Args:
        N (int):            Size of the field.
        q (3d float array): A manually added field.
        q_factor (float):   To normalize q.
        mode (str):         'gaussian' -- white noise smoothed 
        parameters (list):  
        
    Returns:
        (3d float array):   Potential normalized to M=0, STD=1
    """
    deflection_potential = np.fft.rfftn(np.random.normal(0,1,size=[N,N,N]))
    
    if mode == 'power law':
        x,y,z = np.mgrid[:N,:N,:N]
        x[x>N/2] = N - x[x>N/2]
        y[y>N/2] = N - y[y>N/2]
        z[z>N/2] = N - z[z>N/2]
        r2 = (x**2+y**2+z**2)**0.5
        deflection_potential *= np.fft.rfftn((r2+0.1)**parameters[0])
        deflection_potential = np.fft.irfftn(deflection_potential)
    elif mode == 'gaussian':
        deflection_potential = smooth(deflection_potential, parameters[0], N,'gaussian')
    else:
        raise('Mode "' +mode+ '" does not exist in the routine "generate_deflection_potential".' )
    
    deflection_potential += q * q_factor
    deflection_potential -= deflection_potential.mean()
    deflection_potential /= deflection_potential.std()
    
    return deflection_potential

def deflection_vector(h_field):
    """ The normalized gradient of the field.
    
    Args:
        h_field (3d float array): Any field.
    
    Returns:
        grad0, grad1, grad2 (3d float arrays): 3 vector components for each cell.
    """
    # The three components of the vector field are:
    grad0, grad1, grad2 = np.gradient(h_field)

    # We want to normalize it (which is not necessary, but we will do it here):
    norm = np.sqrt(grad0**2+grad1**2+grad2**2)
    norm[norm==0] = 1e10
    grad0 *= 1/norm
    grad1 *= 1/norm
    grad2 *= 1/norm
    
    return grad0, grad1, grad2

def perturb_field(h_field, grad0, grad1, grad2, deflection_potential, interpolation_method = 'nearest', n_steps=10, dt=.35):
    """ Perturbs field h_field allong given vector field grad0, grad1, grad2 and deflection_potential.
    
    Args:
        h_field (3d float array): Any field to perturb.
        grad0 (3d float array):   Components of the vector field.
        grad1 (3d float array):
        grad2 (3d float array):
        deflection_potential (3d float array): Potential
        interpolation_method (str): 'nearest' -- fast method,
                                    'linear' -- much-much slower but looks nicer.
        n_steps (int):            Number of steps for evolving the field.
        dt (float):               Step.
    
    Returns:
        (3d float array): Perturbed field.
    """
    N = h_field.shape[0]
    
    grid_0, grid_1, grid_2 = 0.5 + np.mgrid[0:N, 0:N, 0:N]

    grid_0_shifted = grid_0.copy()
    grid_1_shifted = grid_1.copy()
    grid_2_shifted = grid_2.copy()

    for i in range(n_steps):
        grad_0 = grad0[np.floor(grid_0_shifted).astype(int), 
                        np.floor(grid_1_shifted).astype(int),
                        np.floor(grid_2_shifted).astype(int)]
        grad_1 = grad1[np.floor(grid_0_shifted).astype(int), 
                        np.floor(grid_1_shifted).astype(int),
                        np.floor(grid_2_shifted).astype(int)]
        grad_2 = grad2[np.floor(grid_0_shifted).astype(int), 
                        np.floor(grid_1_shifted).astype(int),
                        np.floor(grid_2_shifted).astype(int)]

        norm = deflection_potential[np.floor(grid_0_shifted).astype(int), 
                    np.floor(grid_1_shifted).astype(int),
                    np.floor(grid_2_shifted).astype(int)]

        grid_0_shifted = grid_0_shifted + grad_0*dt*norm
        grid_1_shifted = grid_1_shifted + grad_1*dt*norm
        grid_2_shifted = grid_2_shifted + grad_2*dt*norm

        grid_0_shifted = np.mod(grid_0_shifted, N)
        grid_1_shifted = np.mod(grid_1_shifted, N)
        grid_2_shifted = np.mod(grid_2_shifted, N)

    
    values = h_field.flatten()
    pos = np.array([grid_0_shifted.flatten(), grid_1_shifted.flatten(), grid_2_shifted.flatten()]).T

    h_field_distorted = griddata(pos, values, (grid_0, grid_1, grid_2), method=interpolation_method, fill_value=np.nan)
    # TODO: Interpolator does not work perfectly near the borders of the grid.
    
    return h_field_distorted

# h_field_test_distorted = rankdata(f_field_test_distorted.flat).reshape([Nt,Nt,Nt])/Nt**3




# ------------------------------
# Routines for analysis
# ------------------------------

def pk(data, boxsize, k_list_phys):
    """Power spectrum of 3D box
    
    Args:
        data (3d float array):     3D field.
        boxsize (float):           Physical size of the box in Mpc/h.
        k_list_phys (float array): Bins in k space.

    Returns:
        P(k): The power spectrum binned by k_list_phys.    
    """
    # Detect number of cells
    N = data.shape[0]
    # Convert k_list_phys to k_list -- k-space of the box
    k_list = k_list_phys*boxsize/N
    # Real FFT of the input 3D field
    data=np.fft.rfftn(data)
    # Evaluating the k in each cell in the k-space
    kx, ky, kz = np.mgrid[:N, :N, :(N/2+1)]
    kx[kx > N/2-1] = kx[kx > N/2-1]-N
    ky[ky > N/2-1] = ky[ky > N/2-1]-N
    kz[kz > N/2-1] = kz[kz > N/2-1]-N
    k=2.0*np.pi*np.sqrt(kx**2+ky**2+kz**2)/N
    # We use np.histogram to quickly evaluate average value in each bin
    h1, dump = np.histogram(k.flat,weights=np.abs(data.flat)**2,bins=k_list)
    h2, dump = np.histogram(k.flat,bins=k_list)
    h2[h2==0] = 1.0
    res = h1/h2
    # Normalization
    res *= boxsize**3/N**6
    return res

def readifrit(path, nvar=0, moden=2, skipmoden=2):
    """ Reading IFRIT formal files.
    """
    openfile=open(path, "rb")
    dump = np.fromfile(openfile, dtype='i', count=moden)
    N1,N2,N3=np.fromfile(openfile, dtype='i', count=3)
    print(N1,N2,N3)
    dump = np.fromfile(openfile, dtype='i', count=moden)
    data = np.zeros([N1, N2, N3])
    j = 0
    for i in range(nvar):
        openfile.seek(4*skipmoden, 1)
        for j in range(4):
            openfile.seek(N1*N2*N3, 1)
        openfile.seek(4*moden, 1)
    openfile.seek(4*skipmoden, 1)
    data[:, :, :] = np.reshape(np.fromfile(openfile, dtype='f4', count=N1 * N2 * N3), [N1, N2, N3])
    openfile.close()
    return N1,N2,N3,data
