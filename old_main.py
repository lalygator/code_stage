from utils_OA import *
import matplotlib.colors as colors
import time
import numpy as np
import pickle
import sys
from astropy.io import fits

# TODO Refaire la fonction PSFs, y'a trop de paramètres
# TODO Mettre au propre MTFs_AVAZ et passer ça dans 'utils'

# * Différence
## * J'ai modifier pour ajouter les barres d'erreur sur les signaux
## * j'ai ajouter une normalisation au clacul des OTF

################################################ ! DÉFINITION DES PARAMÈTRES ! ################################################

#  * Choix du plan pupille : soit un apodiseur AMPL, soit HSP1/2, soit la pupille de l'ELT

# pupil = Pupil(8.0,20.0,0.85)
# pup_mat = pupil.matrix()  # quadrant pupille (ELT)
# P = SQ(pup_mat) # pupille entière (ELT)

# * circurlar aperture
x_circ,y_circ=np.meshgrid(np.linspace(-1,1,1132),np.linspace(-1,1,1132))
circ = x_circ**2+y_circ**2 <= (1024/1132)/2
P = circ

# fits_file = './HSP2.fits'
# with fits.open(fits_file) as hdul:
#     fits_data = hdul[0].data  # hdul[0] pour le premier HDU, .data pour l'array numpy

# P = fits_data # apodiseur HSP1 ou HSP2

plt.imshow(P)

owa = 38 # OWA de l'apodiseur considéré
N = 5 # résolution, valeur minimale pour etre a Nyquist
fov = int(2.5*owa)
nbr_pix = int(N * 2 * fov) # nombre de pixels, minimum 2 par lambda/D pour Nyquist à N=1
M = 500 # nombre d'écran de phase considéré

################################################ ! CALCUL AVEC ET SANS RÉSIDUS D'OA ! ################################################

# * Choix du fichier des écrans de phase des résidus d'OA

file_path_name = './phase_interps_0_500.pkl'
#file_path_name = './interp_funcs_500_600.pkl'
#file_path_name = './phase_interps_500_1000.pkl'

with open(file_path_name, 'rb') as file:
    loaded_interp_funcs = pickle.load(file)
phase_interps = loaded_interp_funcs

list_phase = np.array(phase_interps) # ? Étape vraiment utile de passer dans un np.array ?

########## !  Calcul de PSF, MTF et AZAV(MTF) avec et sans OA

# * Calcul des images sans résidus d'OA : A=P
im_pup = PSF(P,fov,nbr_pix)
# azav_im_pup = AZAV(im_pup, owa,1/owa)
# plt.imshow(im_pup,norm=colors.LogNorm())
# plt.figure()
# plt.plot(azav_im_pup[0])

otf_pup = (OTF(im_pup, fov, nbr_pix))
mtf_pup = np.abs(OTF(im_pup, fov, nbr_pix))
azav_pup = AZAV(mtf_pup, owa,1/owa)


# * Calcul des images avec résidus d'OA : A=P*exp(i*phi)
im = PSF(P,fov,nbr_pix)#PSFs(P,M,fov,N,nbr_pix,owa,list_phase) #### ! ATTENTION J'AI ENELVE LE ACLCUL D'ÉCRAN DE PHASE ICI  
mtf = np.abs(OTF(im, fov, nbr_pix))
azav = AZAV(mtf, owa, 1/owa)

########## ! Plot des PSF, OTF, DSP

# * Permet juste d'afficher les extent des plot correctement
b=1 #lambda/D
x_min=y_min=-b*fov/2
x_max=y_max=b*fov/2


#%%
#0.0280403x-0.253787 : fonction du temps de calcul en fonctin du temps d'écran de phase (diviser par deux pour avoir le nombre d'écran)


# correcteur de dispersion atmosphérique
# ! plot des PSF #

# ! À FAIRE !!!
#def plot_PSF(im)
#!###############
# fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))

# im0 = axs1[0].imshow(im_final / np.max(im_final), norm=colors.LogNorm(vmin= 1e-7, vmax=1),extent=(x_min, x_max, y_min, y_max))
# axs1[0].set_xlabel(r'$\lambda/D$')
# axs1[0].set_title(f'PSF avec résidus OA ({M*2}ms)')
# #axs1[0].axis('off')
# plt.colorbar(im0, ax=axs1[0], fraction=0.046, pad=0.04)

# im1 = axs1[1].imshow(im_pup / np.max(im_pup), norm=colors.LogNorm(vmin= 1e-7, vmax=1),extent=(x_min, x_max, y_min, y_max))
# axs1[1].set_title('PSF sans résidus OA')
# axs1[1].set_xlabel(r'$\lambda/D$')
# #axs1[1].axis('off')
# plt.colorbar(im1, ax=axs1[1], fraction=0.046, pad=0.04)

# plt.tight_layout()
#plt.show()

#sys.exit()
#%%

################################################ ! AFFICHAGE DES DONNÉES ! ################################################

########## ! Plot des MTF 
### * (c'est un subplot parce que je plottais les PSF avant)

fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))

### * MTF avec résidus OA
otf0 = axs2[0].imshow(mtf/mtf_pup.max(), norm=colors.LogNorm())
axs2[0].set_title(f'MTF de la PSF avec résidus OA ({M*2}ms)')
axs2[0].axis('off')
plt.colorbar(otf0, ax=axs2[0], fraction=0.046, pad=0.04)

### * MTF sans OA
otf1 = axs2[1].imshow(mtf_pup/mtf_pup.max(), norm=colors.LogNorm())
axs2[1].set_title('MTF de la PSF sans résidus OA')
axs2[1].axis('off')
plt.colorbar(otf1, ax=axs2[1], fraction=0.046, pad=0.04)

plt.tight_layout()


########## ! Plot des AZAV(MTF)

plt.figure()
plt.plot(np.linspace(0,1/x_max,len(azav[0])),azav[0]/azav_pup[0].max(), label=f'Avec résidus OA ({M*2}ms)') # * AZAV(MTF) avec résidus OA
plt.plot(np.linspace(0,1/x_max,len(azav[0])),azav_pup[0]/azav_pup[0].max(), label='Sans résidus OA') # * AZAV(MTF) sans OA
plt.xlabel(r'Fréquence angulaire ($D/\lambda$)') # ? Bonne unité ? Calculé comme étant 1/xmax
plt.grid()
plt.grid(which="minor", color="0.9")
plt.legend()
plt.title('Moyenne azimutal des MTF')


########## ! Plot de la DSP pour M écran de phase

dsp = DSP(im,im_pup,fov,2*nbr_pix) #*avec célian
#dsp = DSP(im,fov,nbr_pix,otf_pup)
plt.figure()
plt.imshow((np.abs(dsp)),extent=[x_min, x_max, x_min, x_max],norm=colors.LogNorm())
plt.colorbar()
plt.xlabel(r'Fréquence angulaire ($\lambda/D$)')
plt.ylabel(r'Fréquence angulaire ($\lambda/D$)')
#plt.colorbar(dsp, ax=axs2[1], fraction=0.046, pad=0.04)
plt.title(f"DSP pour un temps de pose de {2*M}ms")


########## ! Plot des AZAV(MTF) pour différents temps d'intégrations (grâce à la fonction)
mult_psf = np.zeros((3, int(2.5*owa*N*2),int(2.5*owa*N*2)))
plt.figure()
def MTFs_AZAV(step,rng): 
    #plt.figure()
    #x= np.linspace(0, 1/x_max, len(azav))
    for i in range(rng):
        psf = PSFs(P,5*step**(i),fov,N,nbr_pix,owa,list_phase)
        mult_psf[i]=psf
        # plt.figure()
        # plt.imshow(psf, norm=colors.LogNorm())
        #plt.show()
        mtf = np.abs(OTF(psf, fov, nbr_pix)) # * Refaire la fonction PSFs
        #plt.figure()
        #plt.imshow(mtf)
        azav, err, f = AZAV(mtf,owa,1)
        if i == 0: # * Permet de normaliser par le max de la meilleur transmission pour pouvoir comparer correctement
            max_azav = np.max(azav)
        else:
            current_max = np.max(azav)
            max_azav = current_max
        plt.plot(np.linspace(0,1/x_max,len(azav)), azav/max_azav, label=f'Avec résidus OA ({5*(step**i)}ms)')
        #plt.plot(f, azav - err, linestyle='dotted', color='gray')
        #plt.plot(f, azav + err, linestyle='dotted', color='gray')
    plt.title("AZAV pour différents temps de pose")
    plt.xlabel(r'Fréquence angulaire ($D/\lambda$)')
    plt.grid()
    plt.grid(which="minor", color="0.9")
    plt.legend()
    # plt.show()


#plot_otf = MTFs_AZAV(10,3) #tout combi de param ok tant que produit <= 500

# for i in range(3):
#     plt.figure()
#     plt.imshow(mult_psf[i]/mult_psf[i].max(),norm=colors.LogNorm(vmin=1e-6, vmax=1))
#     plt.colorbar()

plt.figure()
def test(psf_pup,dsp):
    tf_pup = ft_BASIC(psf_pup,fov, 1132/1024, nbr_pix, direction=1)
    kernel = np.exp(ft_BASIC(dsp, 1132/1024, fov, nbr_pix, direction=-1))
    psf_fin = ft_BASIC(tf_pup*kernel, 1132 / 1024, fov, nbr_pix, direction=-1)
    return psf_fin


# * PSF : 1132/1024 , fov


psf_pitie =np.abs(test(im_pup, dsp))

#%%
plt.figure()
plt.imshow(psf_pitie/psf_pitie.max(),norm=colors.LogNorm(vmin=1e-8,vmax=1))
plt.figure()
plt.imshow(im/im.max(),norm=colors.LogNorm(vmin=1e-8,vmax=1))
plt.figure()
plt.imshow(im_pup/im_pup.max(),norm=colors.LogNorm(vmin=1e-8,vmax=1))
plt.show()
# TODO Plot l'céart type pour voir l'enveloppe de la variatin des données
