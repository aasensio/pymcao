import numpy as np
import matplotlib.pyplot as pl
import glob
import projection
import wavefront as wf
import uuid
import pathlib
import scipy.special as sp
from tqdm import tqdm
import logging
import phase_screens
import time
import fft
import pickle

def even(x):
    return x%2 == 0

def nearest(array, value):
    return (np.abs(array-value)).argmin()


class Atmosphere(object):
    def __init__(self, config, zernikes, mask):
        """
        This class defines an atmosphere that can be used to generate synthetic MCAO observations
        and also apply different tomography schemes. It will also be useful for training neural networks
        """

        # Logger
        self.logger = logging.getLogger("ATM  ")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.nHeight = len(config.turb_heights)
        self.n_stars = config.n_stars
        self.fov = config.fov / 206265.0        # in radians
        self.heights = config.turb_heights * 1e3
        self.telescope_diameter = config.telescope_diameter
        self.verbose = config.verbose
        self.MComputed = False
        self.M_sci_Computed = False
        self.numerical_projection = config.numerical_projection
        self.add_piston = config.add_piston
        self.noll0 = 1
        self.wavelength = config.wavelength
        self.fft_mode = config.fft_mode
        self.n_cpus = config.n_cpus                
        self.timestep = config.timestep
        self.Z = zernikes
        self.Z_normalized = zernikes / np.sum(zernikes**2, axis=(1,2))[:,None,None]
        self.n_science_fov = config.n_science_fov
        self.mask = mask
        self.mode = config.phasescreen_mode

        self.n_zernike, self.npixel_pupil, _ = self.Z_normalized.shape
        
        # Strength of each turbulent layer normalized to 1

        if (self.mode == 'computed-phase-screen'):
            self.strength = np.array(config.strength)
            self.strength /= np.sum(self.strength)            

            self.wind_direction = config.wind_direction
            self.wind_speed = config.wind_speed
            self.r0 = config.r0
            self.L0 = config.L0
            self.screen_size = config.screen_size
            self.strength = ( ((self.r0**(-5./3.)) * self.strength)**(-3./5.) )
            self.reuse_screens = config.reuse_screens

        self.n_heights = len(self.heights)                

        if (not self.add_piston):
            self.noll0 = 2

        # Diameter of the metapupils, which depend on the heights and the FOV
        self.DMetapupil = self.telescope_diameter + self.heights * self.fov

        # Pixel size of each metapupil, assuming a certain pixel size for the
        # telescope pupil. We rescale the pixel size to the ratio between the
        # metapupil and the diameter of the telescope
        self.pixel_size_metapupil = np.zeros(self.n_heights)
        self.n_move_pixels = np.zeros(self.n_heights)
        self.n_move_pixels_int = np.zeros(self.n_heights, dtype='int')
        for i in range(self.n_heights):  
            self.pixel_size_metapupil[i] = config.pixel_size_pupil * self.DMetapupil[i] / self.DMetapupil[0]
            if (self.mode == 'computed-phase-screen'):
                self.n_move_pixels[i] = (self.wind_speed[i] * self.timestep) / self.pixel_size_metapupil[i]
                self.n_move_pixels_int[i] = np.ceil(self.n_move_pixels[i])
            
        self.t = np.zeros((self.nHeight,self.n_stars))
        self.beta = np.zeros((self.nHeight,self.n_stars))
        self.angle = np.zeros((self.nHeight,self.n_stars))
        
        if (self.verbose):            
            self.logger.info("Zernike modes: {0}".format(self.n_zernike))
            self.logger.info("Number of heights : {0} -> {1} km".format(self.nHeight, self.heights * 1e-3))
            self.logger.info("FOV: {0} arcsec".format(206265.*self.fov))
            self.logger.info("Number of stars : {0}".format(self.n_stars))
            
        # Define the position of the line of sights so that we have one central direction and the rest
        # are divided with azimuthal symmetry around
        # t : radial position of the center of the footprint in units of the meta-pupil radius R<1
        # beta : scaling, so that the radius of the meta-pupil and of the footprint are related by beta=R(metapupil)/R(footprint)
        # angle : azimuthal angle of the line of sight
        for i in range(self.nHeight):
            for j in range(self.n_stars-1):
                self.t[i,j] = (self.heights[i] * self.fov) / self.DMetapupil[i]
                self.beta[i,j] = self.DMetapupil[i] / self.telescope_diameter
                self.angle[i,j] = j * 2.0 * np.pi / (self.n_stars - 1.0)
            self.t[i,-1] = 0.0
            self.beta[i,-1] = self.DMetapupil[i] / self.telescope_diameter
            self.angle[i,-1] = 0.0
        
        # Check if the projection matrices for this specific configuration exists. If it does, read it from the file
        # If not, compute them
        
        # First create the directory where all matrices will be saved
        p = pathlib.Path('matrices/')
        p.mkdir(parents=True, exist_ok=True)

        if (self.projection_exists() == 0):
            if (self.verbose):
                self.logger.info("Projection matrix for atmosphere (turbulence) does not exist. Computing a new one, which takes long but it is only done once.")
            self.compute_projection()

        self.aStack = {}
        self.a = {}        
                
        # Science FOV
        self.border_science_fov = self.fov * np.sin(np.pi/4.0)
        x_science = self.border_science_fov / self.n_science_fov * (0.5 + np.arange(self.n_science_fov)) - self.border_science_fov / 2.0
        y_science = self.border_science_fov / self.n_science_fov * (0.5 + np.arange(self.n_science_fov)) - self.border_science_fov / 2.0
        self.patch_size_science = self.border_science_fov / self.n_science_fov
                
        self.x_science, self.y_science = np.meshgrid(x_science, y_science)

        r = np.sqrt(self.x_science**2 + self.y_science**2)

        self.t_sci = np.zeros((self.nHeight,self.n_science_fov,self.n_science_fov))
        self.beta_sci = np.zeros((self.nHeight,self.n_science_fov,self.n_science_fov))
        self.angle_sci = np.zeros((self.nHeight,self.n_science_fov,self.n_science_fov))

        for i in range(self.nHeight):
            for j in range(self.n_science_fov):
                for k in range(self.n_science_fov):             
                    self.t_sci[i,j,k] = (self.heights[i] * r[j,k]) / self.DMetapupil[i]
                    self.beta_sci[i,j,k] = self.DMetapupil[i] / self.telescope_diameter
                    self.angle_sci[i,j,k] = np.arctan2(self.y_science[j,k], self.x_science[j,k])
                    
        self.t_sci = self.t_sci.reshape((self.nHeight, self.n_science_fov*self.n_science_fov))
        self.beta_sci = self.beta_sci.reshape((self.nHeight, self.n_science_fov*self.n_science_fov))
        self.angle_sci = self.angle_sci.reshape((self.nHeight, self.n_science_fov*self.n_science_fov))
        
        self.n_sci_directions = self.n_science_fov*self.n_science_fov

        # Science projection matrices
        p = pathlib.Path('matrices_science/')
        p.mkdir(parents=True,exist_ok=True)

        if (self.projection_science_exists() == 0):
            if (self.verbose):
                self.logger.info("Projection matrix for atmosphere (science) does not exist. Computing a new one, which takes long but it is only done once.")
            self.compute_projection_science()

        if (self.mode == 'computed-phase-screen'):

        # Compute phase screens
            self.screens_phase = np.zeros((self.n_heights, self.screen_size, self.screen_size))        

            # Instantiate an FFT routine 
            self.ifft = fft.FFT((self.screen_size, self.screen_size), mode=self.fft_mode, direction='backward', axes=(0,1), threads=self.n_cpus)


            # Compute the infinite phase screens
            start_time = time.time()

            # First check whether we want to use previous computed phase screens
            found = False

            if (self.reuse_screens):
                # Check if screens with the same parameters exist
                found = self.screens_exist()
                    
            if (not found):
                if (self.verbose):
                    self.logger.info("Phase screens do not exist for the combination of parameters. Computing a new one.")
                
                self.screens = [None] * self.n_heights        
                for i in range(self.n_heights):                

                    # All distances have to be given in m for the
                    # PhaseScreenVonKarman function
                    self.screens[i] = phase_screens.PhaseScreenVonKarman(self.screen_size, self.pixel_size_metapupil[i] * 0.01, self.strength[i] * 0.01, self.L0[i] * 0.01, \
                        n_columns=2, IFFT=self.ifft, wind_direction = self.wind_direction[i], verbose=self.verbose)
                    
                pickle.dump(self.screens, open( "screens/screen_{0}.pb".format(uuid.uuid4()), "wb" ) )                    
                
            # Compute the ranges of the phase screens that correspond to the
            # aperture of the telescope
            self.center_screens = self.screen_size // 2
            shift = self.npixel_pupil // 2
            if (self.npixel_pupil // 2 != self.npixel_pupil / 2):            
                self.range_screens = [self.center_screens - shift, self.center_screens + shift + 1]
            else:
                self.range_screens = [self.center_screens - shift, self.center_screens + shift]
                
            end_time = time.time()
            
            if (self.verbose):
                self.logger.info(f"Phase screens computation finalized (t={end_time-start_time:.4f} s)")

        if (self.mode == 'zernike-modes'):
            self.zernike_modes = config.zernike_modes
            if (self.verbose):
                self.logger.info("Using Zernike modes as phase")
            
                    
    def screens_exist(self):
        """
        Check whether the desired screens exist
        
        Returns:
            bool: True/False
        """
        
        # Go through all matrices and check if the parameters coincide with what we want
        files = glob.glob('screens/screen*.pb')
        for f in files:
            self.screens = pickle.load( open( f, "rb" ) )

            if (self.n_heights == len(self.screens)):
                conditions = []
                for i in range(self.n_heights):
                    conditions.append(self.screen_size == self.screens[i].nx_size)
                    conditions.append(self.pixel_size_metapupil[i] * 0.01 == self.screens[i].pixel_scale)
                    conditions.append(self.strength[i] * 0.01 == self.screens[i].r0)
                    conditions.append(self.L0[i] * 0.01 == self.screens[i].L0)
                    
                    if (all(conditions)):
                        
                    # We have found a dataset with the matrices we want. Read it.
                        if (self.verbose):
                            self.logger.info("Phase screens found : {0}".format(f))
                        return True                
        return False

    def move_screens(self):
        """
        Move screens. TODO!!: do reinterpolation for subpixel motion
        """
        if (self.mode == 'computed-phase-screen'):
            start_time = time.time()
            for i in range(self.n_heights):            
                for j in range(self.n_move_pixels_int[i]):                
                    self.screens[i].add_row()

            end_time = time.time()

            if (self.verbose):
                self.logger.info(f"Phase screens moved (t={end_time-start_time:.4f} s)")

    def projection_exists(self):
        """
        Check whether a projection matrix exists
        
        Returns:
            bool: True/False
        """
        
        # Go through all matrices and check if the parameters coincide with what we want
        files = glob.glob('matrices/transformationMatrices*.npz')
        for f in files:
            out = np.load(f)
            heights = out['arr_1']
            n_stars = out['arr_2']
            n_zernike = out['arr_3']
            fov = out['arr_4']
            telescope_diameter = out['arr_5']
            ind = np.where(np.in1d(heights, self.heights))[0]
            if (len(ind) == self.nHeight):
                if (n_stars == self.n_stars and n_zernike >= self.n_zernike and 
                    fov == self.fov and telescope_diameter == self.telescope_diameter):
                    self.M = out['arr_0'][0:self.n_zernike,0:self.n_zernike,ind,:]

                    # We have found a dataset with the matrices we want. Read it.
                    if (self.verbose):
                        self.logger.info("Projection matrix for atmosphere layers (turbulence) exists : {0}".format(f))
                        self.MComputed = True
                        self.stack_projection()
                    return True
                
        return False

    def projection_science_exists(self):
        """
        Check whether a projection matrix for science exists
        
        Returns:
            bool: True/False
        """
        
        # Go through all matrices and check if the parameters coincide with what we want
        files = glob.glob('matrices_science/transformationMatrices*.npz')
        for f in files:
            out = np.load(f)
            heights = out['arr_1']
            n_sci_directions = out['arr_2']
            n_zernike = out['arr_3']
            fov = out['arr_4']
            telescope_diameter = out['arr_5']
            n_science_fov = out['arr_6']
            ind = np.where(np.in1d(heights, self.heights))[0]
            if (len(ind) == self.nHeight):
                if (n_zernike >= self.n_zernike and fov == self.fov and \
                    telescope_diameter == self.telescope_diameter and n_science_fov == self.n_science_fov):
                    
                    self.M_sci = out['arr_0'][0:self.n_zernike,0:self.n_zernike,ind,:]

                    # We have found a dataset with the matrices we want. Read it.
                    if (self.verbose):
                        self.logger.info("Projection matrix for atmosphere (science) exists : {0}".format(f))
                        self.MComputed_sci = True
                        self.stack_projection_science()
                    return True
                
        return False

    def compute_projection(self):
        """
        Compute the projection matrix for the heights and number of stars defined
        
        Returns:
            None
        """
        if (not self.MComputed):
            self.M = np.zeros((self.n_zernike,self.n_zernike,self.nHeight,self.n_stars))
            for i in tqdm(range(self.nHeight), desc='Height'):                
                for j in tqdm(range(self.n_stars), desc='Stars'):                    
                    if (self.numerical_projection):
                        self.M[:,:,i,j] = projection.zernikeProjectionMatrixNumerical(self.n_zernike, self.beta[i,j], self.t[i,j], self.angle[i,j], verbose=True, radius=128, includePiston=self.add_piston)
                    else:
                        self.M[:,:,i,j] = projection.zernikeProjectionMatrix(self.n_zernike, self.beta[i,j], self.t[i,j], self.angle[i,j], verbose=True, includePiston=self.add_piston)
            np.savez('matrices/transformationMatrices_{0}.npz'.format(uuid.uuid4()), self.M, self.heights, self.n_stars, self.n_zernike, self.fov, self.telescope_diameter)
            self.stack_projection()

    def compute_projection_science(self):
        """
        Compute the projection matrix for the heights and the directions defined
        for the science camera
        
        Returns:
            None
        """
        if (not self.M_sci_Computed):
            self.M_sci = np.zeros((self.n_zernike,self.n_zernike,self.nHeight,self.n_sci_directions))
            for i in tqdm(range(self.nHeight), desc='Height'):                
                for j in tqdm(range(self.n_sci_directions), desc='Science directions'):                    
                    if (self.numerical_projection):
                        self.M_sci[:,:,i,j] = projection.zernikeProjectionMatrixNumerical(self.n_zernike, self.beta_sci[i,j], self.t_sci[i,j], self.angle_sci[i,j], verbose=True, radius=128, includePiston=self.add_piston)
                    else:
                        self.M_sci[:,:,i,j] = projection.zernikeProjectionMatrix(self.n_zernike, self.beta_sci[i,j], self.t_sci[i,j], self.angle_sci[i,j], verbose=True, includePiston=self.add_piston)
            np.savez('matrices_science/transformationMatrices_{0}.npz'.format(uuid.uuid4()), self.M_sci, self.heights, self.n_sci_directions, self.n_zernike, self.fov, self.telescope_diameter, self.n_science_fov)
            self.stack_projection_science()

    def stack_projection(self):
        """
        Stack the projection matrix to take all heights into account. This facilitates later calculations because
        we can use matrix operations. All Zernike coefficients will be stacked one after the other for all
        metapupils. When multiplied by the matrix, it will make the transformation to the footprints
        
        Returns:
            None
        """
        self.MStack = np.zeros((self.n_zernike*self.n_stars, self.n_zernike*self.nHeight))
        for i in range(self.nHeight):
            for j in range(self.n_stars):
                left = i*self.n_zernike
                right = (i+1)*self.n_zernike
                up = j*self.n_zernike
                down = (j+1)*self.n_zernike
                self.MStack[up:down,left:right] = self.M[:,:,i,j]


    def stack_projection_science(self):
        """
        Stack the projection matrix to take all heights into account. This facilitates later calculations because
        we can use matrix operations. All Zernike coefficients will be stacked one after the other for all
        metapupils. When multiplied by the matrix, it will make the transformation to the footprints
        
        Returns:
            None
        """
        self.MStack_sci = np.zeros((self.n_zernike*self.n_sci_directions, self.n_zernike*self.nHeight))
        for i in range(self.nHeight):
            for j in range(self.n_sci_directions):
                left = i*self.n_zernike
                right = (i+1)*self.n_zernike
                up = j*self.n_zernike
                down = (j+1)*self.n_zernike
                self.MStack_sci[up:down,left:right] = self.M_sci[:,:,i,j]
    

    def generate_turbulent_zernikes_kolmogorov(self, r0, keepOnly=None):
        """
        A utility to generate the random Zernike coefficients in the metapupils. It uses
        the covariance matrix for the Zernike coefficients for a given value of r0 using Kolmogorov statistics
        
        Args:
            r0 (float): Fried radius [m]
        
        Returns:
            None
        """
        self.covariance = np.zeros((self.n_zernike,self.n_zernike))
        for i in range(self.n_zernike):
            ni, mi = wf.nollIndices(i+self.noll0)
            for j in range(self.n_zernike):
                nj, mj = wf.nollIndices(j+self.noll0)
                if (even(i - j)):
                    if (mi == mj):
                        phase = (-1.0)**(0.5*(ni+nj-2*mi))
                        t1 = np.sqrt((ni+1)*(nj+1)) * np.pi**(8.0/3.0) * 0.0072 * (self.telescope_diameter / r0)**(5.0/3.0)
                        t2 = sp.gamma(14./3.0) * sp.gamma(0.5*(ni+nj-5.0/3.0))
                        t3 = sp.gamma(0.5*(ni-nj+17.0/3.0)) * sp.gamma(0.5*(nj-ni+17.0/3.0)) * sp.gamma(0.5*(ni+nj+23.0/3.0))
                        self.covariance[i,j] = phase * t1 * t2 / t3

        self.a['Original'] = np.random.multivariate_normal(np.zeros(self.n_zernike), self.covariance, size=(self.nHeight)).T

        # Since we might be using projection matrices that contain more heights than 
        # Keep only the heights that we want
        if (keepOnly != None):
            for i in range(self.nHeight):
                if (self.heights[i]/1e3 not in keepOnly):
                    self.a['Original'][:,i] = 0.0

        self.aStack['Original'] = self.a['Original'].T.flatten()

    def zernike_turbulent_metapupils(self):
        
        if (self.mode == 'computed-phase-screen'):
            for i in range(self.n_heights):
                self.screens_phase[i,:,:] = self.screens[i].screen

            # Project the screen to the Zernike modes
            self.turb_zernike = np.einsum('ijk,ljk->il', self.screens_phase[:, self.range_screens[0]:self.range_screens[1], self.range_screens[0]:self.range_screens[1]], self.Z_normalized)
            self.aStack['Original'] = self.turb_zernike.flatten()
            
            self.turb_shape_zernike = np.einsum('ij,jkl->ikl', self.turb_zernike, self.Z)
            self.turb_shape = self.screens_phase[:, self.range_screens[0]:self.range_screens[1], self.range_screens[0]:self.range_screens[1]] * self.mask[None,:,:]

        if (self.mode == 'zernike-modes'):
            self.turb_zernike = np.expand_dims(np.append(self.zernike_modes, np.zeros((self.n_zernike-len(self.zernike_modes)))), axis=0)
            self.aStack['Original'] = self.turb_zernike.flatten()
            self.turb_shape_zernike = np.einsum('ij,jkl->ikl', self.turb_zernike, self.Z)
            self.turb_shape = self.turb_shape_zernike * self.mask[None,:,:]            
        
    def generate_wfs(self):
        """
        This function uses the stacked alpha coefficients for all metapupils and 
        propagates that to the telescope and all directions
        
        Returns:
            float: array of Zernike coefficients measured in all WFS
        """
        start_time = time.time()
        out = (self.MStack @ self.aStack['Original']).reshape((self.n_stars, self.n_zernike))
        end_time = time.time()
        if (self.verbose):
            self.logger.info(f"WFS generated (t={end_time-start_time:.4f} s)")

        return out
        
    def generate_science_wf(self, dms=None):
        """
        This function uses the stacked alpha coefficients for all metapupils and 
        propagates that to the telescope and all directions from the science cameras
        
        Returns:
            float: array of Zernike coefficients measured in all directions of interest
        """
        if (dms is not None):
            atm_wf = self.MStack_sci @ self.aStack['Original']
            dms_wf = dms.MStack_sci @ dms.dms_zernike.flatten()

            return atm_wf.reshape((self.n_sci_directions, self.n_zernike)), (atm_wf - dms_wf).reshape((self.n_sci_directions, self.n_zernike))
        else:
            return (self.MStack_sci @ self.aStack['Original']).reshape((self.n_sci_directions, self.n_zernike))
        
    def lock_points(self):
        xy = np.zeros((2, self.n_stars))
        distance = self.t[-1,:] / np.max(self.t[-1,:])
        xy[0,:] = 206265. * 0.5 * self.fov * distance * np.cos(-self.angle[-1,:])
        xy[1,:] = 206265. * 0.5 * self.fov * distance * np.sin(-self.angle[-1,:])
        return xy

    def sci_pointings(self):
        xy = np.zeros((2, self.n_sci_directions))
        xy[0,:] = 206265 * self.x_science.flatten()
        xy[1,:] = 206265 * self.y_science.flatten()        
        return xy

    def plot_pupils(self):
        """
        Plot the pupils
        """
          
        cmap = pl.get_cmap('tab10')

        f, ax = pl.subplots(ncols=2, nrows=2, figsize=(10,10), constrained_layout=True)
        for i in range(2):
            ax[i,0].set_ylabel('Distance [cm]')
        for i in range(2):
            ax[-1,i].set_xlabel('Distance [cm]')
        ax = ax.flatten()        
        for i in range(self.nHeight):
            radiusMetapupil = self.DMetapupil[i] / 2.0
            circle = pl.Circle((0,0), radiusMetapupil, fill=False, linewidth=2, axes=ax[i])
            ax[i].add_artist(circle)
            ax[i].set_xlim([-0.7*self.DMetapupil[i],0.7*self.DMetapupil[i]])
            ax[i].set_ylim([-0.7*self.DMetapupil[i],0.7*self.DMetapupil[i]])
            ax[i].set_title('h={0} km'.format(self.heights[i] / 1e3))
            for j in range(self.n_stars):
                radiusCircle = radiusMetapupil / self.beta[i,j]
                xCircle = radiusMetapupil * self.t[i,j] * np.cos(self.angle[i,j])
                yCircle = radiusMetapupil * self.t[i,j] * np.sin(self.angle[i,j])
                circle = pl.Circle((xCircle,yCircle), radiusCircle, fill=False, axes=ax[i], linewidth=2, color=cmap(j/self.n_stars))
                ax[i].add_artist(circle)        

    # def to_wavefront(self, a):
    #     tmp = self.Z @ a
    #     return tmp.reshape((self.npix,self.npix))

    def plot_metapupil(self, index_height):
        """
        Plot the pupils                
        """
        self.generate_turbulent_zernikes_kolmogorov(5.0)
        
        cmap = pl.get_cmap('tab10')
        pl.close('all')

        f, ax = pl.subplots(ncols=self.n_stars, nrows=2, figsize=(19,6))
        
        metapupil = self.to_wavefront(self.a['Original'][:,index_height])

        for j in range(self.n_stars):
                    
            # M = projection.zernikeProjectionMatrixNumerical(self.n_zernike, self.beta[index_height,j], self.t[index_height,j], self.angle[index_height,j], includePiston=self.add_piston, radius=128)
            # beta = M @ self.a['Original'][:,index_height]

            beta = self.M[:,:,index_height,j] @ self.a['Original'][:,index_height]

            footprint = self.to_wavefront(beta)

            radiusMetapupil = self.DMetapupil[index_height] / 2.0
            radiusCircle = radiusMetapupil / self.beta[index_height,j]
            xCircle = radiusMetapupil * self.t[index_height,j] * np.cos(-self.angle[index_height,j])
            yCircle = radiusMetapupil * self.t[index_height,j] * np.sin(-self.angle[index_height,j])
            circle = pl.Circle((xCircle,yCircle), radiusCircle, fill=False, axes=ax[0,j], linewidth=2)

            ax[0,j].imshow(metapupil, cmap=pl.cm.jet, vmin=-1, vmax=1, extent=[-radiusMetapupil,radiusMetapupil,-radiusMetapupil,radiusMetapupil])
            ax[0,j].add_artist(circle)
            ax[0,j].set_title('Angle: {0:.2f}'.format(self.angle[index_height,j]))

            ax[1,j].imshow(footprint, cmap=pl.cm.jet, vmin=-1, vmax=1, extent=[-radiusCircle,radiusCircle,-radiusCircle,radiusCircle])

        pl.show()


if (__name__ == '__main__'):
    np.random.seed(123)
    n_stars = 7
    n_zernike = 30
    fov = 60.0
    telescope_diameter = 4.0

    # Compute the transformation matrices for all heights and directions. This takes some time and 
    # could be easily paralellized using MPI. For instance, 15 heights, 30 Zernike modes and 7 directions
    # took 8 min in my computer
    heights = np.arange(15)
    mcao = tomography(n_stars, n_zernike, fov, heights, telescope_diameter, add_piston=True)
    # mcao.plot_metapupil(14)
    mcao.plot_pupils()