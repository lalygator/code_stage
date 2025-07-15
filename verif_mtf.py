#%%
import numpy as np
import matplotlib.pyplot as plt
from utils_OA import ft_BASIC, PSF, AZAV
import matplotlib.colors as colors
import pickle
from skimage.transform import resize
import sys

# ! Calcul et affichage de l'ouverture circulaire 1132/1024

x_circ,y_circ=np.meshgrid(np.linspace(-1,1,1132),np.linspace(-1,1,1132))
circ = x_circ**2+y_circ**2 <= (1132/1024)/2
aper = circ

# shape aper = (1132,1132)
# shape disque = (1024,1024)
plt.figure()
plt.imshow(aper,cmap='gist_gray')
plt.title("Ouverture circulaire utilisée")
plt.axis('off')

# ! Paramètre de la PSF
# vent de 10 m/s
b = 2
owa = 5
fov = b*2.5*owa
N = 3
n = int(2*fov*N)

# ! Calcul de la PSF

plt.figure()
psf = (ft_BASIC(aper, 1132/1024, fov,n,direction=1)).real**2
plt.imshow(psf/psf.max(),norm=colors.LogNorm(vmin=1e-8,vmax=1),extent=[-fov/2,fov/2,-fov/2,fov/2])
plt.xlabel(r'$\lambda/D$')
plt.ylabel(r'$\lambda/D$')
plt.colorbar()
plt.title("PSF normalisé de la pupille circulaire")

plt.figure()
plt.semilogy(np.linspace(0,fov/2,n//2),AZAV(psf/psf.max(),owa,1/(owa))[0])
plt.xlabel(r'$\lambda/D$')
#plt.ylabel(r'$\lambda/D$')
plt.title("Moyenne azimutal de la PSF normalisé de la pupille circulaire")
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

#%%
# ! Calcul de la MTF initial

j = 1 #j=20 à noter#50 aussi pas mal
a = 2
size = a // j
mat = np.zeros((a, a), dtype=bool)
start = (a - size) // j
end = start + size
mat[start:end, start:end] = True
sq = mat

plt.figure()
otf = ft_BASIC(psf, 2*fov, 2*1132/1024,n,direction=1)#/ft_BASIC(sq, 2*fov, 2*1132/1024,n,direction=1)
mtf = np.abs(otf)
plt.imshow(mtf/mtf.max(),norm=colors.LogNorm(vmin=1e-8))
plt.colorbar()
plt.axis('off')
plt.title('MTF de la PSF de l\'ouverture circulaire')


shape = np.shape(mtf)
rows = []
for i in range(20):
    rows.append(mtf[:,i])
mtf_mean = np.stack(rows)

plt.figure()
plt.semilogy(np.mean(mtf_mean, axis=0))
plt.title("Moyenne de 20 tranches verticale sur la partie haute de la MTF")
plt.ylabel("Valeur contraste sur la MTF")
plt.show()

#sys.exit()

#%%
def hamming(x, y, T):
    r = np.sqrt(x**2 + y**2)
    mask = r <= T
    window = np.zeros_like(r)
    window[mask] = 0.5 + 0.5*np.cos(2 * np.pi * r[mask] / (2*T))
    # * Y'a un petit problème d'arrondi ou c'est pas exactement 1 le max mais 0.99998

    return window

R =n
grid_x, grid_y = np.meshgrid(np.linspace(-R, R, n), np.linspace(-R, R, n)) # correctino par 4000 pour éviter que ça fasse plus la meme taille en nombre de pixel que la PSF à cause du padding de 2000 ajouté
# pour avoir tout le cercle jusqu'au bord il faut T=2*R
# si R trop petit ça zoom trop à l'intéireur 
# si T plus petit que 2*R alors ça fait plus petit


fact = fov//2 # ! taille en lambda/D du rayon du filtre, en dehors le champ sera nul

T = 2*(2*N)*fact

hann = hamming(grid_x, grid_y, T) 
x_min = -b*fov//2
x_max = b*fov//2
plt.figure()
plt.imshow(hann, extent=[x_min,x_max,x_min,x_max])#,norm=colors.LogNorm())
plt.title('Fenêtre de Hann')
# plt.xticks([])
# plt.yticks([])

psf_new = psf*hann
otf = ft_BASIC(psf_new, 2*fov, 2*1132/1024,n,direction=1)#/ft_BASIC(sq, 2*fov, 2*1132/1024,n,direction=1)
mtf = np.abs(otf)

plt.figure()
plt.imshow(mtf/mtf.max(),norm=colors.LogNorm(vmin=1e-18))
plt.colorbar()
plt.axis('off')
plt.title('MTF de la PSF de l\'ouverture circulaire')


shape = np.shape(mtf)
rows = []
for i in range(20):
    rows.append(mtf[:,i])
mtf_mean = np.stack(rows)

plt.figure()
plt.semilogy(np.mean(mtf_mean, axis=0))
plt.title("Moyenne de 20 tranches verticale sur la partie haute de la MTF")
plt.ylabel("Valeur contraste sur la MTF")
plt.show()

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

#%%
# ! Calcul de l'AZAV de la MTF
plt.figure()
mtf_azav = AZAV(mtf/mtf.max(),owa,1/(3*owa))
plt.semilogy(mtf_azav[0]/mtf_azav[0].max())
# Création d'une image 2D à partir de la moyenne azimutale (symétrie circulaire)
#%%
plt.figure()
plt.semilogy(np.linspace(0, fov/2, len(mtf_azav[0])), mtf_azav[0], label='Moyenne azimutale')
plt.fill_between(
    np.linspace(0, fov/2, len(mtf_azav[0])),
    mtf_azav[0] - mtf_azav[1],
    mtf_azav[0] + mtf_azav[1],
    color='gray', alpha=0.3, label='Écart-type'
)
plt.xlabel(r'$\lambda/D$')
plt.legend()
#plt.gca().axes.get_xaxis().set_visible(False)
plt.title('Moyenne azimutal de la MTF de la PSF de l\'ouverture circulaire')

#%%

# ! Calcul de l'AZAV de la MTF projeté par symétrie circulaire

def ProjAzav(mtf,fov):
    size = mtf.shape[0]
    x = np.linspace(-fov/2, fov/2, size)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2 + yy**2)
    r_vals = np.linspace(0, fov/2, len(mtf_azav[0]))
    mtf_azav_img = np.interp(rr, r_vals, mtf_azav[0], left=0, right=0)
    return mtf_azav_img

mtf_azav_img = ProjAzav(mtf,fov)

plt.figure()
plt.imshow(mtf_azav_img/mtf_azav_img.max(), extent=[-fov/2, fov/2, -fov/2, fov/2], origin='lower',norm=colors.LogNorm())
plt.colorbar()
plt.title("Symétrie circulaire de la moyenne azimutale de la MTF")
plt.axis('off')


#%%

# ! Calcul de la PSF résultant de la nouvelle MTF calculé par symétrie

plt.figure()
psf_new = np.abs((ft_BASIC(mtf_azav_img, 4*1132/1024, fov, n, direction=-1)))
plt.imshow(psf_new, norm=colors.LogNorm(vmin=1e-8))
plt.colorbar()
plt.show()

#%%
# * Tentative de faire la déconvolution dans l'autre sens pour trouver la tete du filtre
plt.figure()
otf_filtre = ft_BASIC(psf, fov, 1132/1024,n,direction=1)/ft_BASIC(psf_new, fov, 1132/1024,n,direction=1)
mtf_filtre = np.abs(otf_filtre)
psf_filtre = np.abs((ft_BASIC(mtf_filtre, 3*1132/1024, fov, n, direction=-1)))
plt.imshow(psf_filtre/psf_filtre.max(),norm=colors.LogNorm(vmin=1e-6,vmax=1))
plt.colorbar()
plt.axis('off')
plt.title('PSF du filtre induit durant la TF de l\'ouverture circulaire')





#%%
# ! Affichage de la MTF théorique
s = np.linspace(0, 2, 150)
mtf_1d = np.zeros_like(s)
mask = s <= 1
mtf_1d[mask] = 2/np.pi * (np.arccos(s[mask]) - s[mask]*np.sqrt(1 - s[mask]**2))

size = 150
x = np.linspace(-1, 1, size)
xx, yy = np.meshgrid(x, x)
rr = np.sqrt(xx**2 + yy**2)
mtf_2d = np.zeros_like(rr)
m = rr <= 1
mtf_2d[m] = 2/np.pi * (np.arccos(rr[m]) - rr[m]*np.sqrt(1 - rr[m]**2))

new_size = 300
mtf_img = np.zeros((new_size, new_size))
c, h = new_size//2, size//2
mtf_img[c-h:c+h, c-h:c+h] = mtf_2d

plt.figure()
plt.imshow(mtf_img/mtf_img.max(), extent=[-1,1,-1,1], origin='lower')#,norm=colors.LogNorm())
plt.colorbar()
plt.axis('off')
plt.title("MTF théorique d'une tâche d'Airy")

plt.figure()
plt.plot(s, mtf_1d)
plt.title("MTF 1D théorique d'une tâche d'Airy")
plt.gca().axes.get_xaxis().set_visible(False)

plt.show()

#%%
plt.figure()
mtf_img_resized = resize(mtf_img, mtf.shape, mode='constant', preserve_range=True)
test = mtf - mtf_img_resized
plt.imshow(test/test.max(),norm=colors.LogNorm())
plt.colorbar()
plt.axis('off')
plt.title("Différence entre la MTF obtenu par simulation\n et la MTF théorique")
plt.figure()

# plt.figure()
# azav_mtf = AZAV(mtf,owa,1/owa)
# plt.semilogy(azav_mtf[2],azav_mtf[0])

plt.show()
# %%
