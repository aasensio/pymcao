import numpy as np
import logging
import pathlib
import glob

__all__ = ['DM']

class DM(object):
    def __init__(self, config, zernikes=None):

        self.heights = config.dms_heights * 1e3
        self.nHeight = len(self.heights)
        self.n_stars = config.n_stars
        self.fov = config.fov / 206265.0        # in radians
        self.telescope_diameter = config.telescope_diameter
        self.Z = zernikes
        self.n_zernike = self.Z.shape[0]
        self.verbose = config.verbose
        self.n_science_fov = config.n_science_fov
        self.n_sci_directions = self.n_science_fov * self.n_science_fov
        self.svd_thresholding = config.svd_thresholding
        self.weight_wfs = config.weight_wfs

        # Build the weight matrix (diagonal) according to the configuration
        # http://mdav.ece.gatech.edu/ece-6250-fall2017/notes/10-notes-6250-f17.pdf
        self.weight_dm = np.zeros((self.n_stars, self.n_zernike))
        for i in range(self.n_stars):
            self.weight_dm[i,:] = self.weight_wfs[i]
        self.weight_dm = self.weight_dm.flatten()
        
        # Logger
        self.logger = logging.getLogger("DMS  ")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if (self.verbose):            
            self.logger.info("Number of heights : {0} -> {1} km".format(self.nHeight, self.heights * 1e-3))
            self.logger.info("FOV: {0} arcsec".format(206265.*self.fov))
            self.logger.info("Number of stars : {0}".format(self.n_stars))


        # First create the directory where all matrices will be saved
        p = pathlib.Path('matrices/')
        p.mkdir(parents=True,exist_ok=True)

        if (self.projection_exists() == 0):
            if (self.verbose):
                self.logger.info("Projection matrix for DMs (turbulence) does not exist. Computing a new one, which takes long but it is only done once.")
            self.compute_projection()

        # Science projection matrices
        p = pathlib.Path('matrices_science/')
        p.mkdir(parents=True,exist_ok=True)

        if (self.projection_science_exists() == 0):
            if (self.verbose):
                self.logger.info("Projection matrix for DMs (science) does not exist. Computing a new one, which takes long but it is only done once.")
            self.compute_projection_science()

        self.dms_zernike = np.zeros((self.nHeight, self.n_zernike))

        
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
                        self.logger.info("Projection matrix for DMs (turbulence) exists : {0}".format(f))
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
                        self.logger.info("Projection matrix for DMs (science) exists : {0}".format(f))
                        self.MComputed_sci = True
                        self.stack_projection_science()
                    return True
                
        return False

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

        if (self.verbose):
            self.logger.info("Computing DMs reconstruction matrix...")
        
        # Carry out inversion (regularize here)
        U, s, V = np.linalg.svd(np.diag(self.weight_dm) @ self.MStack, full_matrices=False)
        
        max_s = np.max(s)
        inv_s = 1.0 / s
        inv_s[s<self.svd_thresholding * max_s] = 0.0
        
        self.reconstruction_DMs = np.transpose(V) @ np.diag(inv_s) @ np.transpose(U)
            
        
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

    def actuate(self, wfs):
        if (self.verbose):
            self.logger.info("Acting on DMs")
        
        self.dms_zernike = self.reconstruction_DMs @ np.diag(self.weight_dm) @ wfs.flatten()
        
        self.dms_zernike = self.dms_zernike.reshape((self.nHeight, self.n_zernike))
        
        self.dms_zernike[:,0:2] = 0.0
        
        self.dms_shape = np.einsum('ij,jkl->ikl', self.dms_zernike, self.Z)