# interpolation des écrans de phase
#%%
import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline
from astropy.io import fits
from utils_basic import WRITE_DAT, DSP
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#%%
fits_path = '/Users/lalyboyer/Desktop/code_stage/DATA_WF_small.fits'
with fits.open(fits_path) as hdul:
    phase = hdul[0].data

def interps(WF, x1, x2, n, z):
    # Interpolation des phases
    interp_funcs = [RectBivariateSpline(x1, x1, WF[:,:,i]) for i in range(n) if i%z==0]
        # TODO Faire en sorte qu'on puisse skip des écrans de phase

    phase_interps = [interp(x2, x2) for interp in interp_funcs]
    return phase_interps

nb_pix = 100
skip_factor = 100 #nombre d'écran de phase skipped
# si ça vaut 5 alors on aura 200 au final


WF = phase
x1 = np.linspace(-0.5,0.5,400) #écran OA
x2 = np.linspace(-39/38.542/2,39/38.542/2,nb_pix) #écran apodiseur standard
phase_interps = interps(WF, x1, x2, 1000, skip_factor) #1000 correspond au nombre d'écran considér, on pourrait en choisir moins

#%% 
pup = 1-(phase[:,:,0]==0)

pup_interp_func = RectBivariateSpline(x1, x1, pup,kx=1,ky=1)
x2_pup = np.linspace(-0.5, 0.5, nb_pix)
pup_interp = pup_interp_func(x2_pup, x2_pup)

plt.figure()
plt.imshow(pup_interp>=0.5)
plt.colorbar()

#%%
# ! enregistrement des fichier en .dat et .fits
#WRITE_DAT(f'fits/phase_interp_{nb_pix}', np.array(phase_interps).flatten())
#fits.writeto(f'fits/phase_interp_{nb_pix}.fits', np.array(phase_interps), overwrite=True)
pup_fin = (pup_interp >= 0.5).astype(np.float32)
WRITE_DAT(f'fits/Pupil_ELT_100', pup_fin)
fits.writeto(f'fits/Pupil_ELT_100.fits', pup_fin, overwrite=True)

# comment sauver en .dat correctement ?

#%%
for i, arr in enumerate(phase_interps):
    WRITE_DAT(f'fits/phase_interp_{nb_pix}_{i}', arr)

# %%
# regarder les DSP

dsp = DSP(phase[:,:,0], 39/38.542,pup,0,40)[1]
dsp_norm = dsp/dsp.max()
plt.figure()
plt.imshow(dsp_norm,norm=colors.LogNorm(vmin=1e-4),extent=[-35,35,-35,35])
plt.xlabel(r"$\lambda/D$")
plt.ylabel(r"$\lambda/D$")
plt.title("DSP d'une phase non interpolée")
plt.colorbar()

# Extraire une zone centrée de 100 pixels de large autour du centre
dsp_crop = dsp[151:249, 151:249]
dsp_cropped = dsp_crop / dsp_crop.max()
plt.figure()
plt.imshow(dsp_cropped,norm=colors.LogNorm(vmin=1e-4),extent=[-35//4,35//4,-35//4,35//4])
plt.xlabel(r"$\lambda/D$")
plt.ylabel(r"$\lambda/D$")
plt.title("Zoom sur la DSP d'une phase non interpolée")
plt.colorbar()

# ! j'ai mis une extent de 35 comme en fréquence on va de - N_act/2 à + N_act/2
dsp_intp = DSP(phase_interps[0], 39/38.542,pup_interp,0,10)[1]
dsp_intp_norm = dsp_intp/dsp_intp.max()
plt.figure()
plt.imshow(dsp_intp_norm,norm=colors.LogNorm(vmin=1e-4),extent=[-35//4,35//4,-35//4,35//4])
plt.xlabel(r"$\lambda/D$")
plt.ylabel(r"$\lambda/D$")
plt.colorbar()
plt.title("DSP d'une phase interpolée")

diff = dsp_cropped - dsp_intp_norm
plt.figure()
plt.imshow(diff, extent=[-35//4,35//4,-35//4,35//4],norm=colors.LogNorm(vmin=1e-4))
plt.xlabel(r"$\lambda/D$")
plt.ylabel(r"$\lambda/D$")
plt.title("Différence entre DSP non interpolée et interpolée (zoom)")
plt.colorbar()

# %%
