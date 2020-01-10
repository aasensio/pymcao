import numpy as np
import glob
import logging
import pathlib
from astropy.io import fits

__all__ = ['Sun']


class Sun(object):

    def __init__(self, config, lock_points, sci_pointings):
        """
        """

        self.filenames = glob.glob(config.filenames)
        self.n_filenames = len(self.filenames)
        self.npix = config.npix_pupil
        self.verbose = config.verbose
        self.lock_points = lock_points
        self.sci_pointings = sci_pointings
        self.simulation_pixel_size = config.simulation_pixel_size

        self.lock_points /= self.simulation_pixel_size
        self.lock_points = self.lock_points.astype('int')
        self.n_lock_points = self.lock_points.shape[1]

        self.sci_pointings /= self.simulation_pixel_size
        self.sci_pointings = self.sci_pointings.astype('int')
        self.n_sci_pointings = self.sci_pointings.shape[1]
        self.xoffset = config.xoffset
        self.yoffset = config.yoffset

        # Logger
        self.logger = logging.getLogger("SUN  ")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if (self.n_filenames != 0):
            self.new_file()
        else:
            self.logger.error('No available files with solar images.')
            raise Exception

        self.images = np.zeros((self.n_lock_points, self.npix, self.npix))
        self.images_sci = np.zeros((self.n_sci_pointings, self.npix, self.npix))
        

    def new_file(self):

        extension = pathlib.Path(self.filenames[np.random.choice(self.n_filenames)]).suffix

        self.filename = self.filenames[np.random.choice(self.n_filenames)]

        self.image = [None] * self.n_lock_points

        if (extension == '.fits'):
            tmp = fits.open(self.filename)
            self.solar_image = tmp[0].data
        
        if ('I_out' in self.filename):
            self.solar_image = np.memmap(self.filename, dtype='float32', offset=4*4, mode='r', shape=(1024, 1920))
        
        for i in range(self.n_lock_points):            
            self.image[i] = np.roll(self.solar_image, -self.lock_points[:,i], axis=(0,1))
                    
        self.image_sci = [None] * self.n_sci_pointings
        for i in range(self.n_sci_pointings):            
            self.image_sci[i] = np.roll(self.solar_image, -self.sci_pointings[:,i], axis=(0,1))

        if (self.verbose):
            self.logger.info(f"Solar image {self.filename} read")

    def new_image(self):
        bottom = np.random.randint(low=0, high=1024-self.npix)
        left = np.random.randint(low=0, high=1920-self.npix)

        bottom = self.xoffset
        left = self.yoffset

        for i in range(self.n_lock_points):
            self.images[i,:,:] = self.image[i][bottom:bottom+self.npix, left:left+self.npix]

        for i in range(self.n_sci_pointings):
            self.images_sci[i,:,:] = self.image_sci[i][bottom:bottom+self.npix, left:left+self.npix]
                                
        return self.images, self.images_sci