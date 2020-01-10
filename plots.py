import numpy as np
import matplotlib.pyplot as pl


def show_mcao(wfs, dms, sci, atm):
    """
    Show the results of the calculation
    """
    n_wfs = len(wfs)
    f, ax = pl.subplots(nrows=4, ncols=n_wfs, figsize=(18,10))

    for i in range(n_wfs):
        ax[0,i].imshow(wfs[i].wavefront, cmap=pl.cm.viridis, extent=(-wfs[i].telescope_diameter/2, wfs[i].telescope_diameter/2, -wfs[i].telescope_diameter/2, wfs[i].telescope_diameter/2))
        ax[0,i].set_xlabel('x [cm]')
        ax[0,i].set_ylabel('y [cm]')
        ax[0,i].set_title(f'Original WF{i}')

        ax[1,i].imshow(wfs[i].reconstructed_wavefront, cmap=pl.cm.viridis, extent=(-wfs[i].telescope_diameter/2, wfs[i].telescope_diameter/2, -wfs[i].telescope_diameter/2, wfs[i].telescope_diameter/2))
        ax[1,i].set_xlabel('x [cm]')
        ax[1,i].set_ylabel('y [cm]')
        ax[1,i].set_title('WF from slopes')

        ax[2,i].imshow(wfs[i].wfs_image)
        ax[2,i].set_title('WFS image')
        ax[2,i].set_xlabel('x [pix]')
        ax[2,i].set_ylabel('y [pix]')

        ax[3,i].imshow(wfs[i].correlate_image)
        ax[3,i].set_title('Correlation')
        ax[3,i].set_xlabel('x [pix]')
        ax[3,i].set_ylabel('y [pix]')

    pl.show()

    f, ax = pl.subplots(nrows=1, ncols=dms.nHeight)
    if (dms.nHeight == 1):
        ax = [ax]
    for i in range(dms.nHeight):
        im = ax[i].imshow(dms.dms_shape[i,:,:])
        pl.colorbar(im, ax=ax[i])
    ax[0].set_title('DMs')
    pl.show()

    f, ax = pl.subplots(nrows=2, ncols=atm.nHeight)
    if (atm.nHeight == 1):
        ax = [ax]
    for i in range(atm.nHeight):
        im = ax[0,i].imshow(atm.turb_shape[i,:,:])
        pl.colorbar(im, ax=ax[0,i])
        im = ax[1,i].imshow(atm.turb_shape_zernike[i,:,:])
        pl.colorbar(im, ax=ax[1,i])
        
    ax[0,0].set_title('ATM')
    pl.show()

    f, ax = pl.subplots(nrows=1, ncols=3, figsize=(12,6))
    ax[0].imshow(sci.science_original, cmap=pl.cm.gray)
    ax[1].imshow(sci.science_degraded, cmap=pl.cm.gray)
    ax[2].imshow(sci.science_degraded_corrected, cmap=pl.cm.gray)
    ax[0].set_title('Original')
    ax[1].set_title('Uncorrected')
    ax[2].set_title('Corrected')
    pl.show()

    # atm.plot_pupils()
    # pl.show()
        
def show_scao(wfs, save_results=False):
    """
    Show the results of the calculation
    """

    # pl.close('all')

    f, ax = pl.subplots(nrows=4, ncols=4, figsize=(15,8), constrained_layout=True)
    ax = ax.flatten()
        
    ax[0].imshow(wfs.mask)
    ax[0].set_title('mask')
    
    ax[1].imshow(wfs.mask * wfs.pupil, extent=(-wfs.telescope_diameter/2, wfs.telescope_diameter/2, -wfs.telescope_diameter/2, wfs.telescope_diameter/2))
    ax[1].set_title('mask * telescope pupil')
    ax[1].set_xlabel('x [cm]')
    ax[1].set_ylabel('y [cm]')
    
    ax[2].imshow(wfs.pupil_image, cmap=pl.cm.viridis, extent=(-wfs.telescope_diameter/2, wfs.telescope_diameter/2, -wfs.telescope_diameter/2, wfs.telescope_diameter/2))
    ax[2].set_title('Wavefront')
    circle = pl.Circle((0,0), wfs.telescope_diameter/2, color='b', fill=False)
    ax[2].add_artist(circle)
    ax[2].set_xlabel('x [cm]')
    ax[2].set_ylabel('y [cm]')
    
    ax[3].imshow(np.sum(wfs.mask_subaperture, axis=0) * wfs.pupil * wfs.wavefront, cmap=pl.cm.viridis, extent=(-wfs.telescope_diameter/2, wfs.telescope_diameter/2, -wfs.telescope_diameter/2, wfs.telescope_diameter/2))
    ax[3].set_title('Wavefront with lens pupils')
    circle = pl.Circle((0,0), wfs.telescope_diameter/2, color='b', fill=False)
    ax[3].add_artist(circle)
    ax[3].set_xlabel('x [cm]')
    ax[3].set_ylabel('y [cm]')
    
    ax[4].imshow(wfs.wfs_image)
    ax[4].set_title('WFS image')
    ax[4].set_xlabel('x [pix]')
    ax[4].set_ylabel('y [pix]')
    
    ax[5].imshow(wfs.correlate_image)
    ax[5].set_title('Correlation')
    ax[5].set_xlabel('x [pix]')
    ax[5].set_ylabel('y [pix]')
            
    ax[6].imshow(wfs.displacement_image[0,:,:], cmap=pl.cm.viridis)
    ax[6].set_title('x-shift')
    
    ax[7].imshow(wfs.displacement_image[1,:,:], cmap=pl.cm.viridis)
    ax[7].set_title('y-shift')

    ax[8].imshow(wfs.gradient_image[0,:,:], cmap=pl.cm.viridis)
    ax[8].set_title('x-gradient subpupils')

    ax[9].imshow(wfs.gradient_image[1,:,:], cmap=pl.cm.viridis)
    ax[9].set_title('y-gradient subpupils')

    ax[10].imshow(wfs.wavefront, cmap=pl.cm.viridis, extent=(-wfs.telescope_diameter/2, wfs.telescope_diameter/2, -wfs.telescope_diameter/2, wfs.telescope_diameter/2))
    ax[10].set_xlabel('x [cm]')
    ax[10].set_ylabel('y [cm]')
    ax[10].set_title('Original wavefront')

    ax[11].imshow(wfs.reconstructed_wavefront, cmap=pl.cm.viridis, extent=(-wfs.telescope_diameter/2, wfs.telescope_diameter/2, -wfs.telescope_diameter/2, wfs.telescope_diameter/2))
    ax[11].set_xlabel('x [cm]')
    ax[11].set_ylabel('y [cm]')
    ax[11].set_title('Wavefront from slopes')

    ax[12].imshow(wfs.ratio_image)
    ax[12].set_title('Ratio')
    ax[12].set_xlabel('x [pix]')
    ax[12].set_ylabel('y [pix]')

    pl.show()

    if (save_results):
        np.savez('results.npz', wfs_image=wfs.wfs_image, wfs_shifts=wfs.displacement_image, wfs_original=wfs.wavefront, wfs_reconstructed=wfs.reconstructed_wavefront)