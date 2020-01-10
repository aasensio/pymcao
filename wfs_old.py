import numpy as np
import matplotlib.pyplot as pl
import wavefront as wf
import scipy.special as sp
import skimage.transform
import time
import fft
from tqdm import tqdm
import scipy.ndimage
import multiprocessing
import atmosphere
import zern
import util
import logging
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

def even(x):
    return x%2 == 0


class WFS(object):

    def __init__(self, config):
        """
        Instantiate a wavefront sensor
        
        Parameters
        ----------
        config : Config class, optional
            Class containing the configuration
        """
        
        self.n_subapertures_horizontal = config.n_subapertures_horizontal
        self.npix_subaperture = config.npix_subaperture
        self.gap_subaperture = config.gap_subaperture
        self.verbose = config.verbose
        self.fft_mode = config.fft_mode
        self.wavelength = config.wavelength
        self.simulation_pixel_size = config.simulation_pixel_size
        self.remove_tiptilt = config.remove_tiptilt
        self.subapertures_geometry = config.geometry

        if (self.subapertures_geometry == 'shifted'):
            self.npix_pupil = int(self.npix_subaperture * (self.n_subapertures_horizontal + 1) + self.gap_subaperture * (self.n_subapertures_horizontal + 1))
        
        if (self.subapertures_geometry == 'square'):
            self.npix_pupil = int(self.npix_subaperture * self.n_subapertures_horizontal + self.gap_subaperture * self.n_subapertures_horizontal)

        self.telescope_diameter = config.telescope_diameter
        self.wfs_pixel_size = config.wfs_pixel_size
        self.wfs_subaperture_arcsec = config.wfs_subaperture_arcsec
        self.pupil_pixel_size_cm = self.telescope_diameter / self.npix_pupil
        
        self.npix_wfs_subaperture = int(self.wfs_subaperture_arcsec / self.wfs_pixel_size)
        
        if (self.subapertures_geometry == 'shifted'):
            self.npix_wfs = (self.n_subapertures_horizontal + 1) * self.npix_wfs_subaperture + (self.n_subapertures_horizontal + 1) * self.gap_subaperture
            
        if (self.subapertures_geometry == 'square'):
            self.npix_wfs = self.n_subapertures_horizontal * self.npix_wfs_subaperture + self.n_subapertures_horizontal * self.gap_subaperture
                
        x = np.arange(self.npix_pupil)
        y = np.arange(self.npix_pupil)
        self.X, self.Y = np.meshgrid(x,y)

        self.pupil = util.aperture(npix=self.npix_pupil, cent_obs = 0.2, spider=0)

        # Generate Hamming window function for WFS correlation
        x = np.arange(self.npix_wfs_subaperture)
        self.window = 0.53836 + (0.53836 - 1.0) * np.cos(2.0 * np.pi * x / (self.npix_wfs_subaperture-1))
        self.window = self.window[:,None] * self.window[None,:]

        self.overfill = self.psf_scale()
        if (self.overfill < 1.0):
            raise Exception(f"The pixel size of the simulation is not small enough to model a telescope with D={self.telescope_diameter} cm")
        self.npix_overfill = int(np.round(self.npix_pupil*self.overfill))

        self.n_cpus = config.n_cpus
        if (self.n_cpus is None):
            self.n_cpus = multiprocessing.cpu_count()

        # Logger
        self.logger = logging.getLogger("WFS  ")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if (self.verbose):
            self.logger.info(f"Telescope diameter : {self.telescope_diameter} cm")
            self.logger.info(f"Number of subapertures in horizontal : {self.n_subapertures_horizontal} (PUPIL)")
            self.logger.info(f"Number of pixels per subaperture : {self.npix_subaperture} (PUPIL)")
            self.logger.info(f"Gap in pixels : {self.gap_subaperture} (PUPIL)")
            self.logger.info(f"Total number of pixels of pupil : {self.npix_pupil} (PUPIL)")
            self.logger.info(f"Pixel size in pupil : {self.pupil_pixel_size_cm:.4f} cm (PUPIL)")
            self.logger.info(f"Pixel size in detector : {self.wfs_pixel_size} arcsec (WFS)")
            self.logger.info(f"Pixels in each subaperture : {self.wfs_subaperture_arcsec} arcsec (WFS)")
            self.logger.info(f"Number of pixels of detector : {self.npix_wfs} (WFS)")            
            if (self.remove_tiptilt):
                self.logger.info("Removing tip-tilt from wavefronts")
            else:
                self.logger.info("Tip-tilt is considered in wavefronts")
            self.logger.info(f"Geometry of subapertures in pupil plane : {self.subapertures_geometry}")
        
    def psf_scale(self):
        """
        Return the PSF scale appropriate for the required pixel size, wavelength and telescope diameter
        The aperture is padded by this amount; resultant pix scale is lambda/D/psf_scale, so for instance full frame 256 pix
        for 3.5 m at 532 nm is 256*5.32e-7/3.5/3 = 2.67 arcsec for psf_scale = 3

        https://www.strollswithmydog.com/wavefront-to-psf-to-mtf-physical-units/#iv
                
        """
        return 206265.0 * self.wavelength * 1e-8 / (self.telescope_diameter * self.simulation_pixel_size)

    def precompute_zernike(self, n_zernike=20):
        """
        Precompute Zernike polynomials
        
        Parameters
        ----------
        n_zernike : int, optional
            Number of Zernike modes to consider, by default 20
        """

        if (self.verbose):
            self.logger.info(f"Precomputing Zernike modes")
            
        self.n_zernike = n_zernike
        self.Z = np.zeros((self.n_zernike, self.npix_pupil, self.npix_pupil))
        self.Zx = np.zeros((self.n_zernike, self.npix_pupil, self.npix_pupil))
        self.Zy = np.zeros((self.n_zernike, self.npix_pupil, self.npix_pupil))
        
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, self.npix_pupil)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0
        
        for j in range(self.n_zernike):            
            n, m = zern.zernIndex(j+2)
            Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
            
            self.Z[j,:,:] = Z * aperture_mask
            self.Zx[j,:,:] = np.gradient(Z, self.pupil_pixel_size_cm, axis=0) * aperture_mask
            self.Zy[j,:,:] = np.gradient(Z, self.pupil_pixel_size_cm, axis=1) * aperture_mask
    
    def compute_subapertures(self, fill_fraction=1.0):
        """
        Generate all the necessary masks and position of the subapertures, both
        in the pupil plane and in the image plane of the Shack-Hartmann WFS
        
        Parameters
        ----------
        fill_fraction : float, optional
            Fraction of illumination of a subaperture to consider it active, by default 1.0
        """
                    
        start_t = time.time()

        self.mask = np.zeros((self.npix_pupil, self.npix_pupil))
        self.mask_active = []
        self.mask_subaperture = []
        self.mask_boundaries = []
        self.wfs_subaperture_boundaries = []
        
        # Compute first position of the subapertures in both planes to build the
        # displaced array
        if (self.subapertures_geometry == 'square'):
            bottom = int(0.5*self.gap_subaperture)
            bottom_wfs = int(0.5*self.gap_subaperture)
        
        if (self.subapertures_geometry == 'shifted'):
            bottom = int(-0.5 * self.npix_subaperture)
            bottom_wfs = int(-0.5 * self.npix_wfs_subaperture)

        index_subp = 1

        # Loop over computed number of horizontal subapertures
        for i in range(self.n_subapertures_horizontal+2):

            # Size of each subaperture in both planes
            top = bottom + self.npix_subaperture
            top_wfs = bottom_wfs + self.npix_wfs_subaperture
            
            # Square geometry
            if (self.subapertures_geometry == 'square'):
                left = int(0.5*self.gap_subaperture)
                left_wfs = int(0.5*self.gap_subaperture)

            # Take into account the 1/2 displacement of the subapertures
            if (self.subapertures_geometry == 'shifted'):
                left = int(0.5*self.gap_subaperture - (i % 2) * 0.5 * self.npix_subaperture)
                left_wfs = int(0.5*self.gap_subaperture - (i % 2) * 0.5 * self.npix_wfs_subaperture)
            
            
            for j in range(self.n_subapertures_horizontal+2):
                right = left + self.npix_subaperture
                right_wfs = left_wfs + self.npix_wfs_subaperture
                
                if (left < 0):
                    left = 0
                    left_wfs = 0

                if (bottom < 0):
                    bottom = 0
                    bottom_wfs = 0
                
                # Set all pixels in the current subaperture to 1
                mask2 = self.mask * 0
                mask2[bottom:top, left:right] = 1
                
                self.mask[bottom:top, left:right] = 1

                # If the subaperture is fully illuminated, take it into account
                if (np.sum(mask2 * self.pupil) >= fill_fraction * self.npix_subaperture**2):
                    
                    # Add this mask to the active masks
                    self.mask_active.append(mask2)                    

                    # Calculate the boundaries in the pupil plane
                    self.mask_boundaries.append([bottom, top, left, right])

                    centery = int(0.5*(top+bottom))
                    centerx = int(0.5*(right+left))        
                    r_subp = np.sqrt((self.X - centerx)**2 + (self.Y - centery)**2)
                    
                    # Compute circular pupils in each subaperture
                    mask_subaperture = np.zeros((self.npix_pupil, self.npix_pupil))
                    mask_subaperture[r_subp <= 0.5*self.npix_subaperture] = 1

                    self.mask_subaperture.append(mask_subaperture)

                    # Calculate the boundaries in the image plane
                    self.wfs_subaperture_boundaries.append([bottom_wfs, top_wfs, left_wfs, right_wfs])
                                        
                    index_subp += 1
                    
                left = right + self.gap_subaperture
                left_wfs = right_wfs + self.gap_subaperture

            bottom = top + self.gap_subaperture 
            bottom_wfs = top_wfs + self.gap_subaperture 
        
        self.mask_subaperture = np.array(self.mask_subaperture)
        self.mask_active = np.array(self.mask_active)
        self.n_subapertures_total, _, _, = self.mask_subaperture.shape

        self.wfbig = np.zeros((self.npix_overfill, self.npix_overfill))
        self.illum = np.zeros((self.n_subapertures_total, self.npix_overfill, self.npix_overfill))
        
        # Compute the reconstruction matrix
        self.reconstruction_S_Z = np.zeros((2*self.n_subapertures_total, self.n_zernike))
        
        for i in range(self.n_subapertures_total):
            bottom, top, left, right = self.mask_boundaries[i]
            self.reconstruction_S_Z[i,:] = np.mean(self.Zx[:, bottom:top, left:right], axis=(1,2))
            self.reconstruction_S_Z[i+self.n_subapertures_total,:] = np.mean(self.Zy[:, bottom:top, left:right], axis=(1,2))
            
        # Carry out inversion (regularize here)
        U, s, V = np.linalg.svd(self.reconstruction_S_Z, full_matrices=False)
        
        max_s = np.max(s)
        inv_s = 1.0 / s
        inv_s[s<1e-6 * max_s] = 0.0

        self.reconstruction_S_Z = np.transpose(V) @ np.diag(inv_s) @ np.transpose(U)

        end_t = time.time()

        if (self.verbose):
            self.logger.info(f"Number of subapertures with filling fraction >= {fill_fraction} : {self.n_subapertures_total}")
            self.logger.info(f"Subapertures and reconstruction matrix computed (t={end_t-start_t:.4f} s)")

        
    def init_fft(self):
        
        # Define the plans for the FFT
        if (self.verbose):                        
            self.logger.info(f"Using {self.fft_mode} for FFT computations")
            if (self.fft_mode == 'pyfftw'):
                self.logger.info(f"Using {self.n_cpus} CPUs")
            self.logger.info("Preparing plan for FFT in image plane")

        self.fft_forward_multi = fft.FFT((self.n_subapertures_total, self.npix_pupil, self.npix_pupil), mode=self.fft_mode, direction='forward', axes=(1,2), threads=self.n_cpus)                
        self.fft_backward_multi = fft.FFT((self.n_subapertures_total, self.npix_pupil, self.npix_pupil), mode=self.fft_mode, direction='backward', axes=(1,2), threads=self.n_cpus)        
        self.fft_forward_single = fft.FFT((1, self.npix_pupil, self.npix_pupil), mode=self.fft_mode, direction='forward', axes=(1,2), threads=self.n_cpus)
        
        if (self.verbose):            
            self.logger.info("Preparing plans for PSF generation")

        self.fft_forward_overfill = fft.FFT((self.n_subapertures_total, self.npix_overfill, self.npix_overfill), mode=self.fft_mode, direction='forward', axes=(1,2), threads=self.n_cpus)
        
        if (self.verbose):            
            self.logger.info("Preparing plans for WFS correlation")
            
        self.fft_forward_correlation = fft.FFT((self.npix_wfs_subaperture, self.npix_wfs_subaperture), mode=self.fft_mode, direction='forward', axes=(0,1), threads=self.n_cpus)                    
        self.fft_backward_correlation = fft.FFT((self.npix_wfs_subaperture, self.npix_wfs_subaperture), mode=self.fft_mode, direction='backward', axes=(0,1), threads=self.n_cpus)

    def covariance_kolmogorov(self, r0, nterms=20):
        """
        A utility to generate the random Zernike coefficients in the metapupils. It uses
        the covariance matrix for the Zernike coefficients for a given value of r0 using Kolmogorov statistics
        
        Args:
            r0 (float): Fried radius [m]
        
        Returns:
            None
        """
        self.covariance = np.zeros((self.n_zernike, self.n_zernike))
        for i in range(self.n_zernike):
            ni, mi = zern.zernIndex(i+2)
            for j in range(self.n_zernike):
                nj, mj = zern.zernIndex(j+2)
                if (even(i - j)):
                    if (mi == mj):
                        phase = (-1.0)**(0.5*(ni+nj-2*mi))
                        t1 = np.sqrt((ni+1)*(nj+1)) * np.pi**(8.0/3.0) * 0.0072 * (self.telescope_diameter / r0)**(5.0/3.0)
                        t2 = sp.gamma(14./3.0) * sp.gamma(0.5*(ni+nj-5.0/3.0))
                        t3 = sp.gamma(0.5*(ni-nj+17.0/3.0)) * sp.gamma(0.5*(nj-ni+17.0/3.0)) * sp.gamma(0.5*(ni+nj+23.0/3.0))
                        self.covariance[i,j] = phase * t1 * t2 / t3

        self.a_zernike = np.random.multivariate_normal(np.zeros(self.n_zernike), self.covariance)
        self.a_zernike[nterms:] = 0.0
        
    def generate_wavefront(self, r0=20.0, nterms=20):
        """
        Generate a new wavefront. Use the Kolmogorov covariance matrix to
        generate the wavefront and multiply by the Zernike polynomials        
        
        Parameters
        ----------
        r0 : float, optional
            Fried parameter [cm], by default 20.0
        nterms : int, optional
            Number of Zernike modes to consider in the wavefront (should be
            smaller or equal than the number of computed Zernike modes), by default 20
        """

        if (nterms > self.n_zernike):
            raise Exception('You are trying to generate a wavefront with more modes ({0}) than the number of Zernike modes computed ({1}).\nPlease decrease the number of modes of the wavefront or increase the number of Zernike modes'.format(nterms, self.n_zernike))
                
        self.r0 = r0
        self.covariance_kolmogorov(self.r0, nterms=nterms)

        self.wavefront = np.sum(self.a_zernike[:,None,None] * self.Z, axis=0)

        # Compute pupil image        
        self.pupil_image = np.sum(self.mask_active, axis=0) * self.pupil * self.wavefront
        
        # tmp = atmosphere.ft_phase_screen(self.r0 * 1e-2, self.npix_pupil, 1e-2
        # * self.telescope_diameter / self.npix_pupil, 1.0, 0.01)
        
    def set_wavefront(self, wavefront):
        """
        Generate a new wavefront. Use the Kolmogorov covariance matrix to
        generate the wavefront and multiply by the Zernike polynomials        
        
        Parameters
        ----------
        wavefront : float
            Wavefront in Zernike modes
        """
                
        # Compute pupil image
        self.zernike_wavefront = wavefront

        if (self.remove_tiptilt):
            self.zernike_wavefront[0:2] = 0.0

        self.wavefront = np.sum(wavefront[:,None,None] * self.Z, axis=0)

        self.pupil_image = np.sum(self.mask_active, axis=0) * self.pupil * self.wavefront
                
    def quadratic_interpolation_maximum(self, tmp):
        """
        Given an image, it first finds the location of the maximum in pixel
        praecision and then refines it with subpixel precision using 2D
        quadratic interpolation (see M. Löfdahl, arxiv:1009.3401)
        
        Parameters
        ----------
        tmp : float array
            Image
        
        Returns
        -------
        floats
            (x, y) position of the subpixel maximum
        """

        imax, jmax = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)

        a2 = 0.5*(tmp[imax+1,jmax] - tmp[imax-1,jmax])
        a3 = 0.5*(tmp[imax+1,jmax] - 2.0 * tmp[imax,jmax] + tmp[imax-1,jmax])
        a4 = 0.5*(tmp[imax,jmax+1] - tmp[imax,jmax-1])
        a5 = 0.5*(tmp[imax,jmax+1] - 2.0 * tmp[imax,jmax] + tmp[imax,jmax-1])
        a6 = 0.25*(tmp[imax+1,jmax+1] - tmp[imax-1,jmax+1] - tmp[imax+1,jmax-1] + tmp[imax-1,jmax-1])
        
        x2 = imax + (2*a2*a5 - a4*a6) / (a6**2 - 4*a3*a5)
        y2 = jmax + (2*a3*a4 - a2*a6) / (a6**2 - 4*a3*a5)

        return x2, y2

    def psf_from_wavefronts(self, apertures, wavefront):

        half = int((self.npix_overfill - self.npix_pupil)/2)
        self.wfbig[half:half+self.npix_pupil,half:half+self.npix_pupil] = wavefront
        
        self.illum[:, half:half+self.npix_pupil,half:half+self.npix_pupil] = apertures

        phase = np.exp(self.wfbig*(0.+1.j))
                
        ft = self.fft_forward_overfill(self.illum * phase[None,:,:])
        
        powft = np.real(np.conj(ft) * ft)

        sorted = np.roll(np.roll(powft, self.npix_overfill//2, axis=1), self.npix_overfill//2, axis=2)
                
        return sorted[:, half:half+self.npix_pupil,half:half+self.npix_pupil]

    def generate_subpupil_psf(self):
        """
        Generate the PSF of the subapertures
                
        """

        self.time_start_subpupil_psf = time.time()
                                        
        # Compute PSF of each subaperture by only activating one aperture at a
        # time. Do them in parallel if possible    
        self.psf = self.psf_from_wavefronts(self.mask_subaperture, self.wavefront) 
            
        # Energy normalization of the PSF
        self.psf /= np.sum(self.psf, axis=(1,2))[:,None,None]

        self.time_end_subpupil_psf = time.time()

        if (self.verbose):            
            self.logger.info(f"Subpupil PSFs generated (t={self.time_end_subpupil_psf - self.time_start_subpupil_psf:.4f} s)")
        
    def generate_wfs_images(self, noise_std = 0.0):
        """
        Generate images in the WFS

        TODO!!!!!! Add a window!!!!
        
        Parameters
        ----------
        noise_std : float, optional
            Noise standard deviation at the WFS, by default 0.0
        """

        self.noise_std = noise_std
        self.time_start_wfs_image = time.time()

        # Initialize the image of the WFS
        self.wfs_image = np.zeros((self.npix_wfs, self.npix_wfs))

        # Initialize the image with the correlations in the image plane
        self.correlate_image = np.zeros((self.npix_wfs, self.npix_wfs))

        # Initialize the image with the xy displacements
        self.displacement_image = np.zeros((2, self.npix_pupil, self.npix_pupil))
        
        # Initialize the image with the xy gradients
        self.gradient_image = np.zeros((2, self.npix_pupil, self.npix_pupil))

        # Find the ranges for cutting the images of each subaperture in the
        # image plane
        center = self.simulation_pixel_size / self.wfs_pixel_size * (self.npix_pupil // 2)  ##### FIX THIS!!!!!!!!!
        left_image = int(center - self.npix_wfs_subaperture // 2)
        right_image = int(center + self.npix_wfs_subaperture // 2)
        
        # Compute the 2D FFT of the PSF of each subpupil
        psf_fft = self.fft_forward_multi(self.psf)
        im_fft = self.fft_forward_single(self.image[None,:,:] / np.mean(self.image))

        image_final = np.real(self.fft_backward_multi(psf_fft * im_fft))
        
        # Rescale output to WFS pixel size
        im_final = scipy.ndimage.zoom(image_final, (1, self.simulation_pixel_size / self.wfs_pixel_size, self.simulation_pixel_size / self.wfs_pixel_size), order=1)                                        
        
        # im_final = skimage.transform.rescale(image_final, scale=self.simulation_pixel_size / self.wfs_pixel_size, order=1)
        im_final += np.random.normal(loc=0.0, scale=self.noise_std, size=im_final.shape)
                
        # Put all images in their appropriate location in the WFS image plane
        for i in range(self.n_subapertures_total):            
            bottom, top, left, right = self.wfs_subaperture_boundaries[i]

            breakpoint()
        
            self.wfs_image[bottom:top, left:right] = im_final[i, left_image:right_image, left_image:right_image]
            self.wfs_image[bottom:top, left:right] -= np.mean(self.wfs_image[bottom:top, left:right])
            
        self.time_end_wfs_image = time.time()

        if (self.verbose):
            self.logger.info(f"WFS images generated (t={self.time_end_wfs_image - self.time_start_wfs_image:4f} s)")
    
    
    def measure_wfs_correlation(self):
        """
        Measure WFS
        """
        

        self.time_start_wfs_correlation = time.time()

        # Set the first subaperture as reference for computing the shift
        bottom, top, left, right = self.wfs_subaperture_boundaries[0]
        im1 = self.wfs_image[bottom:top, left:right] * self.window

        # t1 = self.fft_forward_correlation(im1)

        t1 = np.fft.fft2(im1)
        
        s = np.zeros(2*self.n_subapertures_total)
                    
        for i in range(self.n_subapertures_total):
            
            bottom, top, left, right = self.wfs_subaperture_boundaries[i]
            im2 = self.wfs_image[bottom:top, left:right] * self.window
            
            # Correlate each subaperture with the first one. Apply a window to
            # avoid high frequencies in the Fourier plane         
            # 
            # TODO !!!!!!! Why this does not work??   Because it returns a
            # reference to the variable and they are overwritten. FIX IT!!!!!!!
            # t2 = self.fft_forward_correlation(im2)            
            # tmp = np.fft.fftshift(self.fft_backward_correlation(t1 * np.conj(t2))).real                                    
            # 
            t2 = np.fft.fft2(im2)
            tmp = np.fft.fftshift(np.fft.ifft2(t1 * np.conj(t2))).real
                                                
            self.correlate_image[bottom:top, left:right] = tmp
            
            # Compute peak with subpixel precision
            x2, y2 = self.quadratic_interpolation_maximum(tmp)

            bottom, top, left, right = self.mask_boundaries[i]

            # Save the displacements and the gradients. Substract the first
            # displacement and gradient as reference
            if (i == 0):
                ref_displacement_x = x2
                ref_displacement_y = y2
                
            self.displacement_image[0, bottom:top, left:right] = -(x2 - ref_displacement_x) * self.wfs_pixel_size
            self.displacement_image[1, bottom:top, left:right] = -(y2 - ref_displacement_y) * self.wfs_pixel_size
                        
            s[i] = -(x2 - ref_displacement_x) * self.wfs_pixel_size
            s[i+self.n_subapertures_total] = -(y2 - ref_displacement_y) * self.wfs_pixel_size
                        
            if (i == 0):
                ref_gradient_x = np.mean(np.gradient(self.pupil_image[bottom:top, left:right], axis=0))
                ref_gradient_y = np.mean(np.gradient(self.pupil_image[bottom:top, left:right], axis=1))
                
            self.gradient_image[0, bottom:top, left:right] = np.sqrt(np.pi) * np.mean(np.gradient(self.pupil_image[bottom:top, left:right], self.pupil_pixel_size_cm, axis=0)) - ref_gradient_x
            self.gradient_image[1, bottom:top, left:right] = np.sqrt(np.pi) * np.mean(np.gradient(self.pupil_image[bottom:top, left:right], self.pupil_pixel_size_cm, axis=1)) - ref_gradient_y
            

        # WHY THIS sqrt(pì)
        self.reconstructed_zernike = self.reconstruction_S_Z @ s

        if (self.remove_tiptilt):
            self.reconstructed_zernike[0:2] = 0.0

        self.reconstructed_wavefront = np.sum(self.reconstructed_zernike[:,None,None] * self.Z, axis=0)
                        
        self.time_end_wfs_correlation = time.time()

        if (self.verbose):            
            self.logger.info(f"WFS correlation computed (t={self.time_end_wfs_correlation - self.time_start_wfs_correlation:4f} s)")
            
    def set_image(self, image):
        """
        Read one of the simulation snapshots for the synthesis of the Shack-Hartmann
        
        Parameters
        ----------
        image : float
            Image
        """
        self.image = image

    def show_mini(self, ax):
        """
        Show the results of the calculation
        """
        ax[0].imshow(self.wavefront, cmap=pl.cm.viridis, extent=(-self.telescope_diameter/2, self.telescope_diameter/2, -self.telescope_diameter/2, self.telescope_diameter/2))
        ax[0].set_xlabel('x [cm]')
        ax[0].set_ylabel('y [cm]')
        ax[0].set_title('Original wavefront')

        ax[1].imshow(self.reconstructed_wavefront, cmap=pl.cm.viridis, extent=(-self.telescope_diameter/2, self.telescope_diameter/2, -self.telescope_diameter/2, self.telescope_diameter/2))
        ax[1].set_xlabel('x [cm]')
        ax[1].set_ylabel('y [cm]')
        ax[1].set_title('Wavefront from slopes')

        ax[2].imshow(self.wfs_image)
        ax[2].set_title('WFS image')
        ax[2].set_xlabel('x [pix]')
        ax[2].set_ylabel('y [pix]')

        ax[3].imshow(self.correlate_image)
        ax[3].set_title('Correlation')
        ax[3].set_xlabel('x [pix]')
        ax[3].set_ylabel('y [pix]')
        
    def show(self):
        """
        Show the results of the calculation
        """

        # pl.close('all')

        f, ax = pl.subplots(nrows=3, ncols=4, figsize=(15,8), constrained_layout=True)
        ax = ax.flatten()
            
        ax[0].imshow(self.mask)
        ax[0].set_title('mask')
        
        ax[1].imshow(self.mask * self.pupil, extent=(-self.telescope_diameter/2, self.telescope_diameter/2, -self.telescope_diameter/2, self.telescope_diameter/2))
        ax[1].set_title('mask * telescope pupil')
        ax[1].set_xlabel('x [cm]')
        ax[1].set_ylabel('y [cm]')
        
        ax[2].imshow(self.pupil_image, cmap=pl.cm.viridis, extent=(-self.telescope_diameter/2, self.telescope_diameter/2, -self.telescope_diameter/2, self.telescope_diameter/2))
        ax[2].set_title('Wavefront')
        circle = pl.Circle((0,0), self.telescope_diameter/2, color='b', fill=False)
        ax[2].add_artist(circle)
        ax[2].set_xlabel('x [cm]')
        ax[2].set_ylabel('y [cm]')
        
        ax[3].imshow(np.sum(self.mask_subaperture, axis=0) * self.pupil * self.wavefront, cmap=pl.cm.viridis, extent=(-self.telescope_diameter/2, self.telescope_diameter/2, -self.telescope_diameter/2, self.telescope_diameter/2))
        ax[3].set_title('Wavefront with lens pupils')
        circle = pl.Circle((0,0), self.telescope_diameter/2, color='b', fill=False)
        ax[3].add_artist(circle)
        ax[3].set_xlabel('x [cm]')
        ax[3].set_ylabel('y [cm]')
        
        ax[4].imshow(self.wfs_image)
        ax[4].set_title('WFS image')
        ax[4].set_xlabel('x [pix]')
        ax[4].set_ylabel('y [pix]')
        
        ax[5].imshow(self.correlate_image)
        ax[5].set_title('Correlation')
        ax[5].set_xlabel('x [pix]')
        ax[5].set_ylabel('y [pix]')
                
        ax[6].imshow(self.displacement_image[0,:,:], cmap=pl.cm.viridis)
        ax[6].set_title('x-shift')
        
        ax[7].imshow(self.displacement_image[1,:,:], cmap=pl.cm.viridis)
        ax[7].set_title('y-shift')

        ax[8].imshow(self.gradient_image[0,:,:], cmap=pl.cm.viridis)
        ax[8].set_title('x-gradient subpupils')

        ax[9].imshow(self.gradient_image[1,:,:], cmap=pl.cm.viridis)
        ax[9].set_title('y-gradient subpupils')

        # ax[10].plot(self.gradient_image[0,:,:].flatten(), self.displacement_image[0,:,:].flatten(),'.')
        # ax[10].plot(self.gradient_image[1,:,:].flatten(), self.displacement_image[1,:,:].flatten(),'.')
        # ax[10].set_xlabel('Average gradient in subapertures')
        # ax[10].set_ylabel('Displacement in WFS')

        ax[10].imshow(self.wavefront, cmap=pl.cm.viridis, extent=(-self.telescope_diameter/2, self.telescope_diameter/2, -self.telescope_diameter/2, self.telescope_diameter/2))
        ax[10].set_xlabel('x [cm]')
        ax[10].set_ylabel('y [cm]')
        ax[10].set_title('Original wavefront')

        ax[11].imshow(self.reconstructed_wavefront, cmap=pl.cm.viridis, extent=(-self.telescope_diameter/2, self.telescope_diameter/2, -self.telescope_diameter/2, self.telescope_diameter/2))
        ax[11].set_xlabel('x [cm]')
        ax[11].set_ylabel('y [cm]')
        ax[11].set_title('Wavefront from slopes')
        
        pl.show()

        pl.savefig('example.png')

    def time_summary(self):
        print('\nTIMING : ')
        print(f' - Generation of PSF : {self.time_end_subpupil_psf - self.time_start_subpupil_psf} s')
        print(f' - Generation of WFS : {self.time_end_wfs_image - self.time_start_wfs_image} s')
        print(f' - Correlation : {self.time_end_wfs_correlation - self.time_start_wfs_correlation} s')
        

if (__name__ == '__main__'):
    np.random.seed(123)
    wfs = WFS(telescope_diameter = 100.0, n_subapertures_horizontal = 9, npix_subaperture = 45, gap_subaperture = 4, wfs_pixel_size=0.2, verbose=True, fft_mode='pyfftw', \
        wavelength=5000.0, simulation_pixel_size=48.0/725.0)
    wfs.precompute_zernike(n_zernike=20)
    wfs.compute_subapertures(fill_fraction=1.0)
    wfs.init_fft()
    wfs.read_simulation(filename='I_out.191000')

    wfs.generate_wavefront(r0=20.0, nterms=20)
    wfs.generate_subpupil_psf()
    wfs.generate_wfs_images(noise_std=1e-2)
    wfs.measure_wfs_correlation()
    wfs.time_summary()
    wfs.show()