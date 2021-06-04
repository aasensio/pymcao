import numpy as np
import matplotlib.pyplot as pl
from configobj import ConfigObj
from astropy import units as u
import copy
import pymcao.wfs as wfs
import pymcao.atmosphere as atmosphere
import pymcao.sun as sun
import logging
import time
from tqdm import tqdm
import threading
import pymcao.comm as comm
import pymcao.dms as dms
import pymcao.plots as plots
import pymcao.science as science
import pymcao.fft as fft
import pymcao.config as config

try:
    from PyQt5.QtNetwork import QHostAddress, QTcpServer
    PYQT_VERSION = 5
except (ImportError ,RuntimeError):
    from PyQt4 import QtGui, QtCore
    QtWidgets = QtGui
    PYQT_VERSION = 4

__all__ = ['Simulator']


class Simulator(object):

    def __init__(self, configuration_file=None):
        self.configuration_file = configuration_file

        self.config = config.Config(configuration_file)

        # Logger
        self.logger = logging.getLogger("SIM  ")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        ch = logging.StreamHandler(self.config.logfile)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def init_simulation(self, init_server=False):
        self.operation_mode = self.config.operation_mode
        self.n_zernike = self.config.n_zernike
        self.n_stars = self.config.n_stars
        self.n_frame = 0
                
        # Instantiate all elements of the simulation

        #---------------------------------------------
        # Instantiate a single WFS
        #---------------------------------------------
        single_wfs = wfs.WFS(self.config)

        # Precompute Zernike basis
        single_wfs.precompute_zernike(n_zernike=self.n_zernike)
        single_wfs.compute_subapertures(fill_fraction=self.config.fill_fraction)
        single_wfs.init_fft()

        if (self.operation_mode == 'scao_single' or self.operation_mode == 'scao'):
            self.wfs = [single_wfs]

        if (self.operation_mode == 'mcao_single' or self.operation_mode == 'mcao'):
            self.wfs = [copy.copy(single_wfs) for i in range(self.config.n_stars)]
        
        
        self.config.pixel_size_pupil = self.wfs[0].pupil_pixel_size_cm
        self.config.npix_overfill = self.wfs[0].npix_overfill
        self.config.npix_pupil = self.wfs[0].npix_pupil

        #---------------------------------------------
        # Intantiate the atmosphere
        #---------------------------------------------
        self.atmosphere = atmosphere.Atmosphere(self.config, self.wfs[0].Z, self.wfs[0].pupil)

        # Find lock points for MCAO and observing points for later degrading the
        # observations
        self.lock_points = self.atmosphere.lock_points()
        self.sci_pointings = self.atmosphere.sci_pointings()

        #---------------------------------------------
        # Instantiate the Sun that provides the images
        #---------------------------------------------
        self.sun = sun.Sun(self.config, self.lock_points, self.sci_pointings)

        #---------------------------------------------
        # Instantiate the DMs
        #---------------------------------------------
        self.dms = dms.DM(self.config, self.wfs[0].Z)

        #---------------------------------------------
        # Instantiate science camera if present
        #---------------------------------------------
        if (self.config.compute_science):
            self.config.patch_size_science = self.atmosphere.patch_size_science
            self.sci = science.Science(self.config, self.wfs[0].Z, self.wfs[0].pupil)
            self.sci.init_fft()            
        
        #---------------------------------------------
        # Instantiate communication mode if necessary 
        #---------------------------------------------              
        if (init_server):
            if (self.operation_mode == 'mcao' or self.operation_mode == 'scao'):
                self.event_comm = threading.Event()
                self.comm = threading.Thread(target=comm.Comm, args=(self, self.config.cadence, self.event_comm))
                self.comm.start()
            
    def init_time(self):
        self.start_time = time.time()        

    def end_time(self):
        self.end_time = time.time()
    
    def print_time(self):
        dt = self.end_time - self.start_time
        fps = (self.n_frame+1) / dt
        print(f" Time : {dt:.4f} s - FPS : {fps:.4f} - Cadence : {1.0/fps:.4f}")

    def close_loggers(self):
        """
        Close all loggers after simulation ends
        """
        for logger in [self.logger, self.sun.logger, self.atmosphere.logger, \
            self.sci.logger, self.config.logger, self.dms.logger]:
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)

        for phase_screen in self.atmosphere.screens:
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)

        for wfs in self.wfs:            
            for handler in wfs.logger.handlers:
                handler.close()
                logger.removeHandler(handler)

    def finalize(self):
        self.close_loggers()

    def start_step(self, silence = False):
        if (silence):
            for wfs in self.wfs:
                wfs.logger.disabled = True
            self.logger.disabled = True

        self.logger.info("Frame")
            
    def frame_scao(self, silence = False, plot=False):
        """
        Run a single frame in SCAO mode
        
        Parameters
        ----------
        silence : bool, optional
            [description], by default False
        plot : bool, optional
            [description], by default False
        """

        if (silence):
            self.wfs[0].logger.disabled = True
            self.logger.disabled = True

        self.logger.info("Frame")

        # Generate new turbulent atmosphere
        # self.atmosphere.generate_turbulent_zernikes_kolmogorov(r0=20.0)
        
        self.atmosphere.zernike_turbulent_metapupils()
        self.atmosphere.move_screens()

        # Compute the wavefronts for all WFS
        self.wavefront = self.atmosphere.generate_wfs()
        self.images, self.images_sci = self.sun.new_image()
        
        # Set the image of the Sun in the WFS
        self.wfs[0].set_image(self.images[0,:,:])

        # Set wavefront
        self.wfs[0].set_wavefront(self.wavefront[0,:])

        # Compute subpupil PSFs and images
        self.wfs[0].generate_subpupil_psf()
        self.wfs[0].generate_wfs_images()

        # Measure correlation
        self.wfs[0].measure_wfs_correlation()

        self.uncorrected_wavefront_sci = self.atmosphere.generate_science_wf()            
        self.sci.degrade_image(self.images_sci, self.uncorrected_wavefront_sci)

        self.n_frame += 1

        if (silence):
            self.wfs[0].logger.disabled = False
            self.logger.disabled = False

        if (plot):
            plots.show_scao(self.wfs[0], save_results=True)

    def frame_mcao(self, silence = False, plot=False):

        if (silence):
            for wfs in self.wfs:
                wfs.logger.disabled = True
            self.logger.disabled = True

        self.logger.info("Frame")

        # Generate new turbulent atmosphere
        # self.atmosphere.generate_turbulent_zernikes_kolmogorov(r0=20.0)

        self.atmosphere.zernike_turbulent_metapupils()
        self.atmosphere.move_screens()

        # Compute the wavefronts for all WFS
        self.wavefront = self.atmosphere.generate_wfs()

        # Extract the images for the WFS and for the science camera
        self.images, self.images_sci = self.sun.new_image()

        self.wfs_zernike = np.zeros((self.n_stars, self.n_zernike))
        
        # Compute the wavefront for every direction
        for i in range(self.n_stars):
            self.wfs[i].set_image(self.images[i,:,:])

            # Set wavefront
            self.wfs[i].set_wavefront(self.wavefront[i,:])

            # Compute subpupil PSFs and images
            self.wfs[i].generate_subpupil_psf()
            self.wfs[i].generate_wfs_images()

            # Measure correlation
            self.wfs[i].measure_wfs_correlation()

            self.wfs_zernike[i,:] = self.wfs[i].reconstructed_zernike

        self.n_frame += 1

        # Actuate the DMs
        self.dms.actuate(self.wfs_zernike)
        
        # Compute science image if needed
        if (self.config.compute_science):
            self.uncorrected_wavefront_sci, self.wavefront_sci = self.atmosphere.generate_science_wf(self.dms)            
            self.sci.degrade_image(self.images_sci, self.uncorrected_wavefront_sci, self.wavefront_sci)
        
        if (plots):
            plots.show_mcao(self.wfs, self.dms, self.sci, self.atmosphere) 
            pl.show()        

        if (silence):
            for wfs in self.wfs:
                wfs.logger.disabled = False
            self.logger.disabled = False

        # Activate events to send data to GUI
        if (self.operation_mode == 'mcao' or self.operation_mode == 'scao'):
            self.event_comm.set()
            self.event_comm.clear()

        breakpoint()


if (__name__ == '__main__'):
    mcao = Simulator('gregor.ini')
    mcao.init_simulation()
    mcao.init_time()

    if (mcao.operation_mode == 'mcao'):
        for i in range(200):        
            mcao.frame_mcao(silence=False)            
    
    if (mcao.operation_mode == 'scao'):        
        for i in range(200):
            mcao.frame_scao(silence=True)

    if (mcao.operation_mode == 'mcao_single'):
        mcao.frame_mcao(silence=False, plot=True)

    if (mcao.operation_mode == 'scao_single'):
        mcao.frame_scao(silence=False, plot=True)

    mcao.end_time()
    mcao.print_time()
    mcao.finalize()