from configobj import ConfigObj
from astropy import units as u
import logging
import numpy as np

__all__ = ['Config']

def _lower_to_sep(string, separator='='):
    line=string.partition(separator)
    string=str(line[0]).lower()+str(line[1])+str(line[2])
    return string

class Config(object):

    def __init__(self, configuration_file=None):
        self.configuration_file = configuration_file

        # Logger
        self.logger = logging.getLogger("CNF  ")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if (self.configuration_file is None):
            print("Simulator initialized without configuration file. Use `read_configuration` to read a configuration file.")
        else:
            self.read_configuration(self.configuration_file)

        self.operation_mode = self.config_dict['global properties']['operation mode'].lower()
        
        # Arguments for WFS
        
        self.telescope_diameter = u.Quantity(self.config_dict['telescope']['diameter']).to(u.cm).value
        self.central_obscuration = u.Quantity(self.config_dict['telescope']['central obscuration']).to(u.cm).value
        self.wavelength = u.Quantity(self.config_dict['global properties']['wavelength']).to(u.Angstrom).value
        self.verbose = True if self.config_dict['global properties']['verbose'].lower() == 'true' else False 
        self.simulation_pixel_size = u.Quantity(self.config_dict['solar images']['pixel size']).to(u.arcsec).value
        self.n_subapertures_horizontal = int(self.config_dict['wfs']['number of subapertures in horizontal'])
        self.npix_subaperture = int(u.Quantity(self.config_dict['wfs']['number of pixels per subaperture in pupil plane']).to(u.pix).value)
        self.gap_subaperture = int(u.Quantity(self.config_dict['wfs']['gap between subapertures in pupil plane']).to(u.pix).value)
        self.wfs_pixel_size = u.Quantity(self.config_dict['wfs']['pixel size in image plane']).to(u.arcsec).value
        self.wfs_subaperture_arcsec = u.Quantity(self.config_dict['wfs']['size of subaperture in image plane']).to(u.arcsec).value
        self.fft_mode = self.config_dict['numerical']['fft mode']
        self.remove_tiptilt = self.config_dict['wfs']['remove tip-tilt']
        self.geometry = self.config_dict['wfs']['geometry']        
        self.n_cpus = None if self.config_dict['numerical']['number of cpus'] == 'None' else int(self.config_dict['numerical']['number of cpus'])
        self.enhancement = float(self.config_dict['wfs']['enhancement'])                
        

        # Arguments for atmosphere
        self.n_stars = int(self.config_dict['wfs']['number of directions'])
                        
        self.fov = u.Quantity(self.config_dict['wfs']['field of view of mcao']).to(u.arcsec).value

        self.phasescreen_mode = self.config_dict['atmosphere']['mode']

        self.numerical_projection = True
        self.add_piston = False

        if (self.phasescreen_mode == 'computed-phase-screen'):
                                                
            if (isinstance(self.config_dict['atmosphere']['strength'], list)):
                self.strength = np.array([float(f) for f in self.config_dict['atmosphere']['strength']])
            else:
                self.strength = np.array([float(self.config_dict['atmosphere']['strength'])])
            
            self.wind_direction = np.array([int(f) for f in self.config_dict['atmosphere']['wind direction']])

            if (isinstance(self.config_dict['atmosphere']['wind speed'], list)):
                self.wind_speed = np.array([u.Quantity(f).to(u.cm / u.s).value for f in self.config_dict['atmosphere']['wind speed']])
            else:
                self.wind_speed = np.array([u.Quantity(self.config_dict['atmosphere']['wind speed']).to(u.cm / u.s).value])

            self.r0 = u.Quantity(self.config_dict['atmosphere']['r0']).to(u.cm).value

            if (isinstance(self.config_dict['atmosphere']['l0'], list)):
                self.L0 = np.array([u.Quantity(f).to(u.cm).value for f in self.config_dict['atmosphere']['l0']])
            else:
                self.L0 = np.array([u.Quantity(self.config_dict['atmosphere']['l0']).to(u.cm).value])
        
            self.screen_size = int(self.config_dict['atmosphere']['screen size'])
            self.reuse_screens = True if self.config_dict['atmosphere']['reuse screens'].lower() == 'true' else False

        if (self.phasescreen_mode == 'zernike-modes'):
            self.zernike_modes = np.array([float(f) for f in self.config_dict['atmosphere']['zernikes']])

        if (isinstance(self.config_dict['atmosphere']['heights'], list)):
            self.turb_heights = np.array([u.Quantity(f).to(u.km).value for f in self.config_dict['atmosphere']['heights']])
        else:
            self.turb_heights = np.array([u.Quantity(self.config_dict['atmosphere']['heights']).to(u.km).value])

        self.timestep = u.Quantity(self.config_dict['global properties']['timestep']).to(u.s).value
        self.n_science_fov = int(self.config_dict['science']['number of horizontal patches in the fov for psf calculation'])
        self.fill_fraction = float(self.config_dict['wfs']['fill fraction of subaperture to be active'])

        self.svd_thresholding = float(self.config_dict['numerical']['svd thresholding'])

        self.weight_wfs = np.array([float(f) for f in self.config_dict['wfs']['weight of each direction']])

        self.cadence = u.Quantity(self.config_dict['visualization']['cadence']).to(u.s).value
        
        # Arguments for deformable mirrors        
        if (isinstance(self.config_dict['dm']['conjugation heights'], list)):
            self.dms_heights = np.array([u.Quantity(f).to(u.km).value for f in self.config_dict['dm']['conjugation heights']])
        else:
            self.dms_heights = np.array([u.Quantity(self.config_dict['dm']['conjugation heights']).to(u.km).value])

        self.compute_science = True if self.config_dict['science']['compute science camera'].lower() == 'true' else False                
        self.n_frame = 0
        self.n_dms = len(self.dms_heights)
        self.n_zernike = int(self.config_dict['numerical']['maximum zernike modes'])

        self.filenames = self.config_dict['solar images']['files']

        if ('xoffset' in self.config_dict['solar images']):
            self.xoffset = int(self.config_dict['solar images']['xoffset'])
        else:
            self.xoffset = 0

        if ('yoffset' in self.config_dict['solar images']):
            self.yoffset = int(self.config_dict['solar images']['yoffset'])
        else:
            self.yoffset = 0

        self.noise_std_wfs = float(self.config_dict['wfs']['noise std']) 

    def read_configuration(self, configuration_file):

        self.logger.info(f"Reading configuration file : {configuration_file}")
        f = open(configuration_file, 'r')
        tmp = f.readlines()
        f.close()

        self.configuration_txt = tmp

        input_lower = ['']

        for l in tmp:
            input_lower.append(_lower_to_sep(l)) # Convert keys to lowercase

        self.config_dict = ConfigObj(input_lower)