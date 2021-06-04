import numpy as np
import matplotlib.pyplot as pl
import logging
import time
import pymcao.fft as fft
from itertools import product

__all__ = ['Science']

class Science(object):
    """
    This class defines a science camera
    """
    def __init__(self, config, zernikes = None, aperture = None):

        self.n_science_fov = config.n_science_fov
        self.science_fov = config.science_fov / 206265.0  # in radians
        self.n_sci_directions = self.n_science_fov * self.n_science_fov
        self.npix_overfill = config.npix_overfill
        self.fft_mode = config.fft_mode
        self.n_cpus = config.n_cpus
        self.verbose = config.verbose
        self.Z = zernikes
        self.aperture = aperture
        self.simulation_pixel_size = config.simulation_pixel_size
        self.patch_size_science = config.patch_size_science
        self.patch_npixel = int(self.patch_size_science * 206265 / self.simulation_pixel_size)
        
        self.n_zernike, self.npix_pupil, _ = self.Z.shape

        self.border_patch = int((self.npix_pupil - self.patch_npixel) // 2)
        
        # Logger
        self.logger = logging.getLogger("SCI  ")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        ch = logging.StreamHandler(config.logfile)        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.illum = np.zeros((self.npix_overfill, self.npix_overfill))
        self.wfbig = np.zeros((self.n_sci_directions, self.npix_overfill, self.npix_overfill))
        
        # Generate Hamming window function for WFS correlation
        percentage = 12.0
        M = int(np.ceil(self.npix_pupil * percentage/100.0))
        win = np.hanning(M)

        winOut = np.ones(self.npix_pupil)
        winOut[0:M//2] = win[0:M//2]
        winOut[-M//2:] = win[-M//2:]

        self.window = np.outer(winOut, winOut)

        if (self.verbose):
            self.logger.info('Science camera ready')
            self.logger.info("FOV of science camera : {0} arcsec".format(206265.*self.science_fov))
            
    def init_fft(self):
        
        # Define the plans for the FFT        
        if (self.verbose):            
            self.logger.info("Preparing plans for science PSF generation")

        self.fft_forward_overfill = fft.FFT((self.n_sci_directions, self.npix_overfill, self.npix_overfill), mode=self.fft_mode, direction='forward', axes=(1,2), threads=self.n_cpus)        
        self.fft_forward_psf = fft.FFT((self.n_sci_directions, self.npix_pupil, self.npix_pupil), mode=self.fft_mode, direction='forward', axes=(1,2), threads=self.n_cpus)
        self.fft_forward_image = fft.FFT((self.n_sci_directions, self.npix_pupil, self.npix_pupil), mode=self.fft_mode, direction='forward', axes=(1,2), threads=self.n_cpus)
        self.fft_backward_multi = fft.FFT((self.n_sci_directions, self.npix_pupil, self.npix_pupil), mode=self.fft_mode, direction='backward', axes=(1,2), threads=self.n_cpus)

    def degrade_image(self, images, uncorrected_wavefront_zernike, wavefront_zernike=None):
        """
        Degrade the science image using the wavefronts obtained for many
        observing directions
        
        Parameters
        ----------
        wavefront_zernike : float
            Zernike coefficients of the wavefronts
        images : float arrays
            Original images from the simulation
        """

        if (self.verbose):
            self.logger.info("Computing science images")

        #-------------------------
        # DMs uncorrected image
        #-------------------------

        # First compute PSFs
        self.wavefronts = np.einsum('ij,jkl->ikl', uncorrected_wavefront_zernike, self.Z)

        half = int((self.npix_overfill - self.npix_pupil)/2)
        self.wfbig[:, half:half+self.npix_pupil,half:half+self.npix_pupil] = self.wavefronts
                
        self.illum[half:half+self.npix_pupil,half:half+self.npix_pupil] = self.aperture
                
        phase = np.exp(self.wfbig*(0.+1.j))
                
        ft = self.fft_forward_overfill(self.illum[None,:,:] * phase)
        
        powft = np.real(np.conj(ft) * ft)

        self.psfs = np.roll(np.roll(powft, self.npix_overfill//2, axis=1), self.npix_overfill//2, axis=2)                    
        
        self.psfs = self.psfs[:, half:half+self.npix_pupil,half:half+self.npix_pupil]
        
        # Now carry out the convolution
        psf_fft = self.fft_forward_psf(self.psfs)
        im_fft = self.fft_forward_image(images * self.window[None,:,:])

        images_final = np.fft.fftshift(np.real(self.fft_backward_multi(psf_fft * im_fft)), axes=(1,2))
        
        # Unpatch the images to reconstruct the final image
        self.science_degraded = self.unpatchify(images_final)
        self.science_original = self.unpatchify(images)

        #-------------------------
        # DMs corrected image
        #-------------------------

        if (wavefront_zernike is not None):

            # First compute PSFs
            self.wavefronts = np.einsum('ij,jkl->ikl', wavefront_zernike, self.Z)

            half = int((self.npix_overfill - self.npix_pupil)/2)
            self.wfbig[:, half:half+self.npix_pupil,half:half+self.npix_pupil] = self.wavefronts
                    
            self.illum[half:half+self.npix_pupil,half:half+self.npix_pupil] = self.aperture
                    
            phase = np.exp(self.wfbig*(0.+1.j))
                    
            ft = self.fft_forward_overfill(self.illum[None,:,:] * phase)
            
            powft = np.real(np.conj(ft) * ft)

            self.psfs = np.roll(np.roll(powft, self.npix_overfill//2, axis=1), self.npix_overfill//2, axis=2)                    
            
            self.psfs = self.psfs[:, half:half+self.npix_pupil,half:half+self.npix_pupil]
            
            # Now carry out the convolution
            psf_fft = self.fft_forward_psf(self.psfs)
            im_fft = self.fft_forward_image(images * self.window[None,:,:])

            images_final = np.fft.fftshift(np.real(self.fft_backward_multi(psf_fft * im_fft)), axes=(1,2))
            
            # Unpatch the images to reconstruct the final image
            self.science_degraded_corrected = self.unpatchify(images_final)
                        

    def unpatchify(self, patches):        
        left = self.border_patch
        right = left + self.patch_npixel
        images_final = patches[:,left:right,left:right]
        images_final = np.transpose(images_final.reshape((self.n_science_fov, self.n_science_fov, self.patch_npixel, self.patch_npixel)), axes=(0,3,1,2))
        return images_final.reshape((self.n_science_fov * self.patch_npixel, self.n_science_fov * self.patch_npixel))                        

    def set_wavefront(self, wavefront):
        """
        Generate a new wavefront from the Zernike coefficients
        
        Parameters
        ----------
        wavefront : float
            Wavefront in Zernike modes
        """
                
        # Compute pupil image
        self.zernike_wavefront = wavefront
        
        