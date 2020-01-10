import numpy as np
import scipy.integrate as integ
import scipy.special as sp
import zern
from tqdm import tqdm

def _sign(x):
    if (x == 0):
        return 1
    else:
        return np.sign(x)

def _signGeneralized(mi, mj):
    if (mj == 0 or mi == 0):
        return 1
    else:
        return _sign(mi)

def _signEqual(x, y):
    if (x == 0 and y > 0):
        return True

    if (x > 0 and y == 0):
        return True

    if (np.sign(x) == np.sign(y)):
        return True

    return False

def cmask(a, b, radius, array):    
    nx,ny = array.shape
    y,x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius

    return mask

def _Aplus(ni, nj, mi, mj, alpha):
    summ = np.abs(mi) + np.abs(mj)
    phase = (-1.0+0J)**(0.5 * (3.0*(ni+nj) - np.abs(mi) - np.abs(mj) + np.abs(summ))) * (1J)**(np.abs(mj) - np.abs(mi))
    
    if (_signEqual(mi, mj)):
        return phase * _signGeneralized(mi,mj) * np.cos(summ * alpha)
    else:        
        return phase * np.sin(summ * alpha)

def _Aminus(ni, nj, mi, mj, alpha):
    diffm = np.abs(mi) - np.abs(mj)    
    phase = (-1.0+0J)**(0.5 * (3.0*(ni+nj) - np.abs(mi) - np.abs(mj) + np.abs(diffm))) * (1J)**(np.abs(mj) - np.abs(mi))

    if (_signEqual(mi, mj)):
        return phase * np.cos(diffm * alpha)
    else:
        return phase * _signGeneralized(mi,mj) * np.sin(np.abs(diffm) * alpha)

def _funcminus(k, ni, nj, mi, mj, beta, t):    
    return sp.jv(ni+1, 2.0*np.pi*k) * sp.jv(nj+1, 2.0*np.pi*k*beta) * sp.jv(np.abs(np.abs(mi) - np.abs(mj)), 2.0*np.pi*k*beta*t) / k

def _funcplus(k, ni, nj, mi, mj, beta, t):
    return sp.jv(ni+1, 2.0*np.pi*k) * sp.jv(nj+1, 2.0*np.pi*k*beta) * sp.jv(np.abs(mi) + np.abs(mj), 2.0*np.pi*k*beta*t) / k

def zernikeProjection(ni, nj, mi, mj, beta, t, alpha):    
    """
    Computes an element of the Zernike projection matrix for the Zernikes Z(ni,mi) and Z(nj,mj)
    The projection matrix relates the Zernike coefficients of an original meta-pupil with a footprint
    of smaller size inside the first one. The footprint is scaled and translated.

    beta -> scaling, so that the radius of the meta-pupil and of the footprint are related by beta=R(metapupil)/R(footprint) >= 1
    t -> radial position of the center of the footprint in units of the meta-pupil radius R<1
    alphat -> rotation angle of the center of the footprint [radians]
    
    Args:
        ni (int): radial degree of the first Zernike
        nj (int): radial degree of the second Zernike
        mi (int): azimuthal degree of the first Zernike
        mj (int): azimuthal degree of the second Zernike
        beta (float): scaling, so that the radius of the meta-pupil and of the footprint are related by beta=R(metapupil)/R(footprint)
        t (float): radial position of the center of the footprint in units of the meta-pupil radius R<1
        alpha (float): rotation angle of the center of the footprint [radians]
    
    Returns:
        float: matrix element Mij
    """
    Ap = np.real(_Aplus(ni, nj, mi, mj, alpha))
    Am = np.real(_Aminus(ni, nj, mi, mj, alpha))

    prefactor = 1.0
    if (mi != 0):
        prefactor *= np.sqrt(2.0)
    if (mj != 0):
        prefactor *= np.sqrt(2.0)

    prefactor *= beta * np.sqrt(ni+1) * np.sqrt(nj+1)

    out = 0.0

    if (np.abs(Am) > 1e-15):
        integminus, error1 = integ.quad(_funcminus, 1e-3, np.inf, args=(ni, nj, mi, mj, beta, t), limit=500)
        out += Am * integminus
        
    if (np.abs(Ap) > 1e-15):
        integplus, error2 = integ.quad(_funcplus, 1e-3, np.inf, args=(ni, nj, mi, mj, beta, t), limit=500)
        out += Ap * integplus

    return np.real(prefactor * out)

def zernikeProjectionMatrix(nMax, beta, t, alphat, verbose=True, includePiston=False):
    """
    Computes the matrix that relates the Zernike coefficients of an original meta-pupil with a footprint
    of smaller size inside the first one. The footprint is scaled and translated.
    
    b = M @ a
    
    with b the Zernike coefficients of the footprint in their local Zernike basis, M the transformation
    matrix, and a the original Zernike coefficients of the meta-pupil.
    It uses the analytical expression for the elements of the transformation matrix given in
    "Transformartion of Zernike coefficients: a Fourier-based method for scaled, translated and rotated wavefront apertures"
    Eric Tatulli, 2013, JOSA, 30, Issue 4, 726-732
    
    Args:
        nMax (int): maximum Noll index for the Zernikes
        beta (float): scaling, so that the radius of the meta-pupil and of the footprint are related by beta=R(metapupil)/R(footprint)
        t (float): radial position of the center of the footprint in units of the meta-pupil radius R<1
        alphat (float): angle of the footprint
        verbose (bool, optional): verbose
        includePiston (bool, optional): include the piston coefficient
    
    Returns:
        float: transformation matrix
    
    Deleted Parameters:
        alpha (float): rotation angle of the center of the footprint [radians]
        process (bool, optional): Description
    """
    M = np.zeros((nMax,nMax))
    loop = 0
    if (includePiston):
        delta = 1
    else:
        delta = 2
    for i in tqdm(range(nMax)):
        for j in range(nMax):
            ni, mi = zern.zernIndex(i+delta)
            nj, mj = zern.zernIndex(j+delta)            
            
            M[i,j] = zernikeProjection(ni, nj, mi, mj, beta, t, alphat)
            loop += 1
    return M

def zernikeProjectionMatrixNumerical(nMax, beta, t, alphat, radius=128, verbose=True, includePiston=False):
    """Computes the matrix that relates the Zernike coefficients of an original meta-pupil with a footprint
    of smaller size inside the first one. The footprint is scaled and translated.
    
    b = M @ a
    
    with b the Zernike coefficients of the footprint in their local Zernike basis, M the transformation
    matrix, and a the original Zernike coefficients of the meta-pupil.

    This routine carries out the numerical projection of the Zernike basis.
    
    Args:
        nMax (int): maximum Noll index for the Zernikes
        beta (float): scaling, so that the radius of the meta-pupil and of the footprint are related by beta=R(metapupil)/R(footprint)
        t (float): radial position of the center of the footprint in units of the meta-pupil radius R<1
        alphat (float): angle of the footprint
        verbose (bool, optional): verbose
        includePiston (bool, optional): include the piston coefficient
    
    Returns:
        float: transformation matrix
    
    Deleted Parameters:
        alpha (float): rotation angle of the center of the footprint [radians]
        process (bool, optional): Description
    """
    if (includePiston):
        delta = 1
    else:
        delta = 2

    radiusCircle = int(radius / beta)
    
    Z_machine = zern.ZernikeNaive(mask=[])
    
    x = np.linspace(-1, 1, int(2*radius))
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    aperture_mask = rho <= 1.0

    x = np.linspace(-1, 1, int(2*radiusCircle))
    xx, yy = np.meshgrid(x, x)
    rho2 = np.sqrt(xx ** 2 + yy ** 2)
    theta2 = np.arctan2(yy, xx)
    aperture_mask2 = rho2 <= 1.0
    
    zero = np.zeros((2*radius,2*radius))
        
    M = np.zeros((nMax,nMax))
    loop = 0

    zj = [None] * nMax

    for i in range(nMax):
        # zi = wf.zernike(i+delta, npix=int(2*radius))

        n, m = zern.zernIndex(i+delta)
        zi = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi') * aperture_mask
                
        xCircleMask = int(radius + radius * t * np.sin(alphat))
        yCircleMask = int(radius + radius * t * np.cos(alphat))
        mask = cmask(xCircleMask, yCircleMask, radiusCircle, zero)

        maskedMetaPupil = zi * mask
        maskedMetaPupil = maskedMetaPupil[xCircleMask-radiusCircle:xCircleMask+radiusCircle,yCircleMask-radiusCircle:yCircleMask+radiusCircle]
        
        for j in range(nMax):
            
            if (i == 0):
                n2, m2 = zern.zernIndex(j+delta)
                zj[j] = Z_machine.Z_nm(n2, m2, rho2, theta2, True, 'Jacobi') * aperture_mask2

            # zj = wf.zernike(j+delta, npix=int(2*radiusCircle))
            
            M[j,i] = np.sum(zj[j] * maskedMetaPupil) / np.sum(zj[j]**2)
            loop += 1

    return M
