import numpy as np

try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except:
    PYFFTW_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

__all__ = ['FFT']

class FFT(object):

    def __init__(self, size_array, mode='pyfftw', direction='forward', axes=(0,1), threads=1):
        self.size_array = size_array
        self.direction = direction
        self.axes = axes

        if (mode == 'pyfftw'):
            if (PYFFTW_AVAILABLE):
                self.mode = mode
            else:
                print(" WARNING : PYFFTW not found, so numpy will be used for FFT.")
                self.mode = 'numpy'

        if (mode == 'torch'):
            if (TORCH_AVAILABLE):
                self.mode = mode
            else:
                print(" WARNING : PyTorch not found, so numpy will be used for FFT.")
                self.mode = 'numpy'

            
        if (mode == 'numpy'):
            self.mode = 'numpy'

        if (self.mode == 'pyfftw'):
            self.input_data = pyfftw.empty_aligned(size_array, dtype='complex64')
            self.output_data = pyfftw.empty_aligned(size_array, dtype='complex64')
            
            if (self.direction == 'forward'):
                self.fft_object = pyfftw.FFTW(self.input_data, self.output_data, axes=self.axes, direction='FFTW_FORWARD', threads=threads)
            if (self.direction == 'backward'):
                self.fft_object = pyfftw.FFTW(self.input_data, self.output_data, axes=self.axes, direction='FFTW_BACKWARD', threads=threads)
                
        if (mode == 'torch'):
            self.mode = 'torch'
            self.cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if self.cuda else "cpu")
            size_array = list(size_array)
            size_array.append(2)
            size_array = tuple(size_array)
            self.input_data = torch.zeros(size_array).to(self.device)
            self.output_data = torch.zeros(size_array).to(self.device)
            self.tmp = np.zeros(size_array)
                                    
    def __call__(self, data = None):

        return self.fft(data)

    def fft(self, data):

        if (self.mode == 'numpy'):
            if (self.direction == 'forward'):
                return np.fft.fft2(data, axes=self.axes)
            if (self.direction == 'backward'):
                return np.fft.ifft2(data, axes=self.axes)

        if (self.mode == 'pyfftw'):
            self.input_data[:] = data
            return self.fft_object()

        if (self.mode == 'torch'):            
            self.tmp[:,:,:,0] = data.real
            self.tmp[:,:,:,1] = data.imag
            self.input_data = torch.from_numpy(self.tmp).to(self.device)
            if (self.direction == 'forward'):
                self.output_data = torch.fft(self.input_data, 2).cpu()
            if (self.direction == 'backward'):
                self.output_data = torch.ifft(self.input_data, 2).cpu()
            return self.output_data.numpy()[:,:,:,0] + 1j * self.output_data.numpy()[:,:,:,1]            