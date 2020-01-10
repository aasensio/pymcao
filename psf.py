import numpy as np
import matplotlib.pyplot as pl
import wavefront as wf
import poppy
import scipy.special as sp
import scipy.stats as stats

try:
    import progressbar as pb
    doProgress = True
except:
    doProgress = False
    pass

class psf(object):

    def __init__(self, telescope_diameter, pixel_size, lambda0, npix_psf, n_zernike=0):
        """
        Initialization of the class
        
        Args:			 
             telescopeDiameter (real): telescope diameter in m
             pixel_size (real): pixel size in arcsec
             lambda0 (real): observing wavelength
             npix_psf (int, optional): number of pixels where to sample the PSF
             n_zernike (int, optional): number of Zernike coefficients to be considered in the wavefront
        
        Returns:
             TYPE: Description
        """		
        self.lambda0 = lambda0
        self.telescope_diameter = telescope_diameter
        self.pixel_size = pixel_size		
        self.npix_psf = npix_psf
        self.n_zernike = n_zernike
        if (self.n_zernike != 0):
            self.precalculate_zernike()

    def compute_aperture(self, centralObs = 0, spider = 0):
        """
        Compute the aperture of the telescope
        
        Args:
             centralObs (int, optional): central obscuration
             spider (int, optional): spider size in pixel
        
        Returns:
             real: compute the aperture
        """
        self.aperture = wf.aperture(npix = self.npix_psf, cent_obs = centralObs, spider=spider)


    def _even(self, x):
        return x%2 == 0

    def _zernike_parity(self, j, jp):
        return self._even(j-jp)

    def init_covariance(self):
        """
        Fill the covariance matrix for Kolmogorov turbulence
        Args:
            r0 (float): Fried parameter (cm)
        Returns:
            N/A
        """
        self.covariance = np.zeros((self.n_zernike,self.n_zernike))

        for j in range(self.n_zernike):
            n, m = poppy.zernike.noll_indices(j+1)
            for jpr in range(self.n_zernike):
                npr, mpr = poppy.zernike.noll_indices(jpr+1)
                
                deltaz = (m == mpr) and (self._zernike_parity(j, jpr) or m == 0)
                
                if (deltaz):                
                    phase = (-1.0)**(0.5*(n+npr-2*m))
                    t1 = np.sqrt((n+1)*(npr+1)) 
                    t2 = sp.gamma(14./3.0) * sp.gamma(11./6.0)**2 * (24.0/5.0*sp.gamma(6.0/5.0))**(5.0/6.0) / (2.0*np.pi**2)

                    Kzz = t2 * t1 * phase
                    
                    t1 = sp.gamma(0.5*(n+npr-5.0/3.0))
                    t2 = sp.gamma(0.5*(n-npr+17.0/3.0)) * sp.gamma(0.5*(npr-n+17.0/3.0)) * sp.gamma(0.5*(n+npr+23.0/3.0))
                    self.covariance[j,jpr] = Kzz * t1 / t2

        self.covariance[0,0] = 1.0
        self.covariance[0,:] = 0.0
        self.covariance[:,0] = 0.0

        self.random_zernike = stats.multivariate_normal(mean=np.zeros(self.n_zernike), cov=self.covariance, allow_singular=True)

    def precalculate_zernike(self):
        """
        Precalculate the Zernike polynomials on the aperture
        """
        self.zernikes = np.zeros((self.n_zernike,self.npix_psf,self.npix_psf))
        for j in range(self.n_zernike):
            self.zernikes[j,:,:] = wf.zernike(j+1,npix=self.npix_psf)


    def get_psf_seeing(self, r0, no_piston=False, no_tiptilt=False, extra_defocus=0.0, no_seeing=False, output_diffraction=True):
        """
        Compute a seeing PSF
        
        Args:
             no_piston (boolean, optional): remove the piston
             no_tiptilt (boolean, optional): remove the tip-tilt
        
        Returns:
             real: the final PSF
        """		
        self.coeff = (self.telescope_diameter / (r0/100.0))**(5.0/6.0) * self.random_zernike.rvs()
        if (no_seeing):
            self.coeff *= 0.0

        if (no_piston):
            self.coeff[0] = 0.0
        if (no_tiptilt):
            self.coeff[1:3] = 0

        if (output_diffraction):
        
            self.wavefront = np.sum(self.coeff[:,None,None] * self.zernikes, axis=0) * 0.0

            # self.wavefront = wf.seeing(self.telescope_diameter * 100.0 / r0, npix = self.npix_psf, nterms = nterms, quiet=True)	
            self.diffraction = wf.psf(self.aperture, self.wavefront, overfill = wf.psfScale(self.telescope_diameter, self.lambda0, self.pixel_size))

    # Pad the PSF image to make it equal to the original image		
            self.diffraction = np.roll(self.diffraction, int(self.npix_psf/2), axis=0)
            self.diffraction = np.roll(self.diffraction, int(self.npix_psf/2), axis=1)

        self.wavefront = np.sum(self.coeff[:,None,None] * self.zernikes, axis=0)

        # self.wavefront = wf.seeing(self.telescope_diameter * 100.0 / r0, npix = self.npix_psf, nterms = nterms, quiet=True)	
        self.psf = wf.psf(self.aperture, self.wavefront, overfill = wf.psfScale(self.telescope_diameter, self.lambda0, self.pixel_size))

# Pad the PSF image to make it equal to the original image		
        self.psf = np.roll(self.psf, int(self.npix_psf/2), axis=0)
        self.psf = np.roll(self.psf, int(self.npix_psf/2), axis=1)

        self.coeff_defocus = np.copy(self.coeff)

        if (extra_defocus != 0):
            self.coeff_defocus[3] += extra_defocus * np.pi / np.sqrt(3.0)

            self.wavefront_defocus = np.sum(self.coeff_defocus[:,None,None] * self.zernikes, axis=0)

        # self.wavefront = wf.seeing(self.telescope_diameter * 100.0 / r0, npix = self.npix_psf, nterms = nterms, quiet=True)	
            self.psf_defocus = wf.psf(self.aperture, self.wavefront_defocus, overfill = wf.psfScale(self.telescope_diameter, self.lambda0, self.pixel_size))

# Pad the PSF image to make it equal to the original image		
            self.psf_defocus = np.roll(self.psf_defocus, int(self.npix_psf/2), axis=0)
            self.psf_defocus = np.roll(self.psf_defocus, int(self.npix_psf/2), axis=1)

            if (output_diffraction):
                return self.psf, self.psf_defocus, self.diffraction, self.wavefront
            else:
                return self.psf, self.psf_defocus, self.wavefront

        if (output_diffraction):
            return self.psf, self.diffraction, self.wavefront
        else:
            return self.psf, self.wavefront

    def get_psf_diffraction(self):
        """
        Compute a diffraction PSF
        
        Returns:
             real: the final PSF
        """
        self.wavefront = self.zernikes[0,:,:]
        self.psf = wf.psf(self.aperture, self.wavefront, overfill = wf.psfScale(self.telescope_diameter, self.lambda0, self.pixel_size))

# Pad the PSF image to make it equal to the original image		
        self.psf = np.roll(self.psf, int(self.npix_psf/2), axis=0)
        self.psf = np.roll(self.psf, int(self.npix_psf/2), axis=1)
        return self.psf

    def center_image(self, image):
        size = image.shape
        return np.roll(np.roll(image, int(size[0]/2), axis=0), int(size[1]/2), axis=1)

    def convolve_with_psf(self, index):
        self.psfFFT = np.fft.fft2(self.psf)
        return np.real(np.fft.ifft2(self.psfFFT * np.fft.fft2(self.cube[:,:,index])))

if (__name__ == '__main__'):
    
    # GREGOR
    telescopeDiameter = 1.440      # m
    secondaryDiameter = 0.404      # m
    pixel_size = 4.0 / 725.0              # arcsec
    lambda0 = 5000.0            # Angstrom
    diff_limit = 1.22 * lambda0 * 1e-8 / (telescopeDiameter * 100.0) * 206265

    pl.close('all')

    nPix = 65
    nRealizations = 500
    npix_psf = 101

    osys = psf(telescopeDiameter, pixel_size, lambda0, npix_psf, 40)
    osys.compute_aperture(centralObs = 0, spider = 0)

    osys.init_covariance()
    psf_focus, psf_defocus, psf_diffraction, wavefront = osys.get_psf_seeing(r0 = 15.0, no_piston=True, no_tiptilt=True, extra_defocus=1.0, no_seeing=False)   
    large_wf = wf.scaled_wavefront(wavefront, npix_wavefront=128)

    tmp = np.roll(psf_diffraction, npix_psf // 2, axis=(0,1))

    pl.semilogy(tmp[:,npix_psf // 2])
    pl.axvline(npix_psf // 2 + diff_limit / pixel_size)
    pl.show()



    # # Initialize the class with the telescope information
    # out = pyPSF(telescopeDiameter, pixel_size, lambda0, nPix)

    # # Compute aperture and return the diffraction PSF
    # out.computeAperture(centralObs = (secondaryDiameter/telescopeDiameter)**2, spider = 0)
    # psfDiffraction = out.centerImage(out.generateSeeingPSF(r0, nterms=0))

    # # Now generate many realizations of the seeing and average them out to
    # # mimick a long integration time observation. We use 20 random Zernike modes
    # psf = np.zeros((nPix,nPix))
    # if (doProgress):
    #     pbar = pb.ProgressBar(maxval=nRealizations).start()
    # for i in range(nRealizations):
    #     if (doProgress):
    #         pbar.update(i+1)
    #     psf += out.generateSeeingPSF(r0, nterms=20)	

    # if (doProgress):
    #     pbar.finish()

    # psf /= (1.0*nRealizations)
    # psfCenter = out.centerImage(psf)

    # f, ax = pl.subplots(ncols=2, nrows=1, figsize=(10,8))
    # ax[0].imshow(np.log(psfCenter))
    # ax[1].imshow(np.log(psfDiffraction))

    # # Save the PSF for later use
    # # np.save('psfVTTDiffraction_pix0.36.npy', psfDiffraction)
    # # np.save('psfVTT0.5arcsec_pix036.npy', psf)

    # np.save('psfGREGORDiffraction_10830.npy', psfDiffraction)
    # np.save('psfGREGORSeeing0.5_10830.npy', psf)