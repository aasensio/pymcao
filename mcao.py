import numpy as np
import matplotlib.pyplot as pl
import tomography

# Define the properties of the telescope, FOV, etc.
np.random.seed(123)
nStars = 7
nZernike = 30
fov = 60.0
DTel = 4.0
npix = 100

# Define all potential heights that can harbor turbulence or DMs
heights = np.arange(15)

# Instantiate the tomography class. This reads the projection matrix and defines
# some useful functions
mcao = tomography.tomography(nStars, nZernike, fov, heights, DTel, addPiston=True, npix=npix)

# Indices of the height vector on which we have turbulence
index_h_turbulence = [0, 10]
n_layers_turbulence = len(index_h_turbulence)

# Indices of the height vector on which we have a DM
index_h_DMs = [0, 11]
n_layers_DMs = len(index_h_DMs)

# Generate some artificial random turbulence at the selected heights
alpha = np.random.randn(nZernike,n_layers_turbulence)

# Now define the Zernike coefficients for the DMs
d = np.copy(-alpha) #np.random.randn(nZernike,n_layers_DMs)

# Define the wavefronts array to do some plots
wfs = np.zeros((nStars, npix, npix))
loss = np.zeros(nStars)

mask = mcao.pupil == 1
n_pixel = np.sum(mask)

pl.close('all')
f, ax = pl.subplots(nrows=3, ncols=3, figsize=(13,8), constrained_layout=True)
ax = ax.flatten()

# Loop over directions
for i in range(nStars):
    
    # For each direction, we compute the Zernike coefficients of the wavefront by adding the
    # contribution of each footprint on each turbulent layer
    beta = np.zeros(nZernike)
    for j in range(n_layers_turbulence):
        # We project each turbulent metapupil with the appropriate projection matrix on the desired footprint
        beta += mcao.M[:,:,index_h_turbulence[j],i] @ alpha[:,j]

    # And now substract the same thing but with the DMs
    for j in range(n_layers_DMs):
        beta += mcao.M[:,:,index_h_DMs[j],i] @ d[:,j]

    # Finally, the resulting Zernike coefficient on each direction is multiplied
    # by the Zernike basis and summed
    wfs[i,:,:] = mcao.to_wavefront(beta)

    loss[i] = np.sum(wfs[i,mask]**2) / n_pixel

    im = ax[i].imshow(wfs[i,:,:], cmap=pl.cm.viridis)
    ax[i].set_title('L={0:.2f}'.format(loss[i]))
    pl.colorbar(im, ax=ax[i])

pl.show()