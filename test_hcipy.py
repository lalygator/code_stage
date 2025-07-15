from hcipy import *
from numpy.fft import fftshift, fft2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from utils_OA import *

pupil_diameter = 0.003 # meter
wavelength = 1e-6 # meter

pupil_grid = make_pupil_grid(256, 1.2 * pupil_diameter)
aperture_circ = evaluate_supersampled(circular_aperture(pupil_diameter), pupil_grid, 8)
#aperture_circ = evaluate_supersampled(ap, pupil_grid, 8)

wf_circ = Wavefront(aperture_circ, wavelength)

propagation_distance = 0.01 # meter

fresnel = FresnelPropagator(pupil_grid, propagation_distance)

img_circ = fresnel(wf_circ)

plt.imshow(img_circ.intensity.shaped, extent=[-0.6, 0.6, -0.6, 0.6], cmap='inferno')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.colorbar(label='Intensity')

# Calcul de la TF de la PSF (OTF)

psf = img_circ.intensity.shaped
otf = fftshift(fft2(psf))

plt.figure()
plt.imshow(np.abs(otf), cmap='viridis')#, norm=colors.LogNorm())
plt.title('Module de la TF de la PSF (OTF)')
plt.xlabel('Fréquence spatiale x')
plt.ylabel('Fréquence spatiale y')
#plt.colorbar(label='Amplitude')

azav = AZAV(np.abs(otf),pupil_diameter/2, 1/pupil_diameter)
plt.figure()
plt.plot(azav)

plt.show()