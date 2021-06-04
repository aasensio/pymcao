import numpy as np

__all__ = ['aperture', 'nollIndices']

def aperture(npix=256, cent_obs=0.0, spider=0):
    """
    Compute the aperture image of a telescope
  
    Args:
        npix (int, optional): number of pixels of the aperture image
        cent_obs (float, optional): central obscuration fraction
        spider (int, optional): spider size in pixels
    
    Returns:
        real: returns the aperture of the telescope
    """
    illum = np.ones((npix,npix),dtype='d')
    x = np.arange(-npix/2,npix/2,dtype='d')
    y = np.arange(-npix/2,npix/2,dtype='d')

    xarr = np.outer(np.ones(npix,dtype='d'),x)
    yarr = np.outer(y,np.ones(npix,dtype='d'))

    rarr = np.sqrt(np.power(xarr,2) + np.power(yarr,2))/(npix/2)
    outside = np.where(rarr > 1.0)
    inside = np.where(rarr < cent_obs)

    illum[outside] = 0.0
    if np.any(inside[0]):
        illum[inside] = 0.0

    if (spider > 0):
        start = int(npix/2 - int(spider)/2)
        illum[start:start+int(spider),:] = 0.0
        illum[:,start:start+int(spider)] = 0.0

    return illum

def nollIndices(j):
    """
    Return the (n,m) pair of indices from a noll index j
    
    Args:
        j (int): Noll index
    
    Returns:
        int, int: (n,m) Zernike indices associated with the Noll index j
    """
    narr = np.arange(40)
    jmax = (narr+1)*(narr+2)/2
    wh = np.where(j <= jmax)
    n = wh[0][0]
    mprime = j - n*(n+1)/2
    if ((n % 2) == 0):
        m = 2*int(np.floor(mprime/2))
    else:
        m = 1 + 2*int(np.floor((mprime-1)/2))

    if ((j % 2) != 0):
        m *= -1

    return n, m