#%%
from amplpy import AMPL
from utils_basic import READ_DAT, ft_BASIC
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors

ampl = AMPL()
ampl.read("AMPL_MTF_phase_2.mod")

# %%
#test = READ_DAT('./TEST_multiple_defocus')
test = READ_DAT('./APOD_multiple_HDZ_test_OWA8_100ecr')

fits.writeto('APOD_multiple_HDZ_test_OWA8_100ecr.fits', test, overwrite=True)

pup = np.reshape(test,(100,100))

# transmission = np.sum(pup >= 0.9) / pup.size

# surf_tot = 166367 # surface totale de l'apodiseur, nombre de pixels qui composent l'apodiseur
# transmission = np.sum(self.apod.matrix() >= 0.9) / surf_tot

# print(f"Transmission: {transmission:.4f}")

plt.figure()
plt.imshow(pup>=0.9)
plt.colorbar()


iwa = 3
owa = 8# OWA de l'apodiseur considéré
N = 2 # résolution, valeur minimale pour etre a Nyquist
fov = int(2.5*owa) # champ de vue de la psf
nbr_pix = int(N * 2 * fov)
b = 1 #lambda/D
x_min = y_min = -b*fov/2
x_max = y_max = b*fov/2

psf = ft_BASIC(pup>=0.9, 39/38.452,fov,nbr_pix,direction=1).real**2


plt.figure()
plt.imshow(psf/psf.max(), norm=colors.LogNorm(vmin=1e-5),extent=[x_min,x_max,x_min,x_max])
plt.colorbar()
plt.xlabel(r"$\lambda/D$")
plt.xticks(rotation=45, ha="right")
plt.ylabel(r"$\lambda/D$")
plt.title(f'PSF longue pose')
circle_owa = plt.Circle((0, 0), owa, color='red', fill=False, linewidth=2, label='OWA')
circle_iwa = plt.Circle((0, 0), iwa, color='blue', fill=False, linewidth=2, label='IWA')
ax = plt.gca()
ax.add_patch(circle_owa)
ax.add_patch(circle_iwa)
ax.set_aspect('equal')
plt.show()


# %%
