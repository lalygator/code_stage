#%%
import numpy as np
from astropy.io import fits
from utils_OA import READ_DAT, SQ, timeit, ft_BASIC, AZAV
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
from matplotlib.path import Path
import matplotlib.colors as colors
from interp_phase import interps

#%%
class Aperture:
    """
    Permet de renvoyer ou afficher la matrice COMPLETE d'une aperture 
    """
    def __init__(self, name, iwa=0, owa=0, t=0):
        self.iwa = iwa
        self.owa = owa
        self.t = t
        self.name = name
        self.mat = None
    #@timeit
    def matrix(self):
        if self.mat is None:
            match self.name :
                case 'ELT':
                    self.mat = SQ(np.fromfile(f'./ELT_1132_M1=2_M4=4_ROT=5.dat', dtype=float, sep='\n').reshape((566, 566)))            
                case 'HSP1' | 'HSP2':
                    fits_file = f'./{self.name}.fits'
                    with fits.open(fits_file) as hdul:
                        fits_data = hdul[0].data  # hdul[0] pour le premier HDU, .data pour l'array numpy
                    self.mat = fits_data
                case 'ATM':
                    fits_file = f'./fits/APOD_1_QDZ.fits'
                    with fits.open(fits_file) as hdul:
                        fits_data = hdul[0].data
                    self.mat = fits_data
                case 'circ':
                    x_circ,y_circ=np.meshgrid(np.linspace(-1,1,1132),np.linspace(-1,1,1132))
                    circ = x_circ**2+y_circ**2 <= (1132/1024)/2
                    self.mat = circ
                case 'square':
                    size = 1132 // 2
                    mat = np.zeros((1132, 1132), dtype=bool)
                    start = (1132 - size) // 2
                    end = start + size
                    mat[start:end, start:end] = True
                    self.mat = mat
                case _:
                    A = READ_DAT(self.iwa, self.owa, self.t,name='ELT')
                    self.mat = SQ(A.reshape((566, 566)))
        return self.mat

    def affiche(self):
        if self.mat is None:
            self.matrix()
        plt.imshow(self.mat, cmap='binary_r')
        plt.axis('off')
        plt.show()

class PSF_utils:
    def __init__(self, aperture, fov, N, z, OAtime=None,norm=False):
        self.aperture = aperture
        self.fov = fov # champ de vue observée totale
        self.N = N # nombre de pixel
        self.z = z
        self.OAtime = OAtime
        self.norm = norm
        self.psf = None

    @timeit
    def PSF(self): 
        if self.psf is None:
            if self.OAtime is None:
                psf = (ft_BASIC(self.aperture.matrix(), 1132 / 1024, self.fov, self.N, direction=1).real)**2
                self.psf = psf
            else:
                fits_path = '/Users/lalyboyer/Desktop/code_stage/DATA_WF_small.fits'
                with fits.open(fits_path) as hdul:
                    phase = hdul[0].data

                x1 = np.linspace(-0.5,0.5,400) #écran OA
                x2 = np.linspace(-39/38.542/2,39/38.542/2,100) #écran apodiseur standard
                ecran = interps(phase, x1, x2, 1000, self.z)
                #ecran = np.empty((self.OAtime//2,1132,1132))

                m = self.OAtime//2 # nombre d'écran de phase considéré 
                mat = self.aperture.matrix() # matrice de la pupille/de l'apodiseur
                ecran = np.array(ecran)  # Ensure ecran is a NumPy array
                calc = ecran[:m,:,:]     # array des écrans de phases considéré
                phase = np.exp(2j*np.pi*calc/1.65e-6)
                A = np.multiply(phase,mat)
                im = np.zeros((m,self.N,self.N)) # allocation initial de la mémoire pour le calcul
                print(np.shape(A))
                for i, A_i in enumerate(A):
                    #print(i)
                    if i%(self.z) == 0 : 
                        print(i)
                        #im[i,:,:] = (ft_BASIC(A_i, 1132 / 1024, self.fov, self.N, direction=1).real)**2
                        im[i,:,:] = (ft_BASIC(A_i, 39 / 38.542, self.fov, self.N, direction=1).real)**2
                    #print(f'Image {i} fini')
                psf = np.mean(im, axis=0)
                self.psf = psf
        if self.norm == True:
            return self.psf/self.psf.max()
        return self.psf

# * Choix de la pupille
aper = Aperture('ATM')
# * Choix des paramètres
iwa = 3
owa = 8 # OWA de l'apodiseur considéré
N = 2 # résolution, valeur minimale pour etre a Nyquist
fov = int(2.5*owa) # champ de vue de la psf
nbr_pix = int(N * 2 * fov) # nombre de pixels, minimum 2 par lambda/D pour Nyquist à N=1
M = 2000 # temps d'OA considéré, en ms #//1min pour faire 1500 avec np multiply
psf_LP = PSF_utils(aper,fov,nbr_pix,2,M).PSF()
#%%
z = 200 # saut en ms d'écran de phase
psf_LP_skip = PSF_utils(aper,fov,nbr_pix,z,M).PSF()
#psf_tel = PSF_utils(aper,fov,nbr_pix).PSF()
#%%
# ! PSF longue pose pour différents temps de pose
psf_LP_500 = PSF_utils(aper,fov,nbr_pix,500).PSF()
psf_LP_1000 = PSF_utils(aper,fov,nbr_pix,1000).PSF()
psf_LP_1500 = PSF_utils(aper,fov,nbr_pix,1500).PSF()
psf_LP_2000 = psf_LP#PSF_utils(aper,fov,nbr_pix,2000).PSF()
#%%
fits.writeto('PSF_500_ELT_8.fits', psf_LP_500)#, overwrite=True)
fits.writeto('PSF_1000_ELT_8.fits', psf_LP_1000)#, overwrite=True)
fits.writeto('PSF_1500_ELT_8.fits', psf_LP_1500)#, overwrite=True)
fits.writeto('PSF_2000_ELT_8.fits', psf_LP_2000)#, overwrite=True)


#%%
# * Affichage de la PSF

b = 1 #lambda/D
x_min = y_min = -b*fov/2
x_max = y_max = b*fov/2
#%%
plt.figure()
plt.imshow(psf_LP/psf_LP.max(), norm=colors.LogNorm(vmin=1e-6),extent=[x_min,x_max,x_min,x_max])
plt.colorbar()
plt.xlabel(r"$\lambda/D$")
plt.xticks(rotation=45, ha="right")
plt.ylabel(r"$\lambda/D$")
plt.title(f'PSF longue pose')
plt.show()

plt.figure()
plt.imshow(psf_LP_skip/psf_LP_skip.max(), norm=colors.LogNorm(vmin=1e-6),extent=[x_min,x_max,x_min,x_max])
plt.colorbar()
plt.xlabel(r"$\lambda/D$")
plt.xticks(rotation=45, ha="right")
plt.ylabel(r"$\lambda/D$")
plt.title(f'PSF longue pose pour une image toutes les {z} ms \n soit {M/z} images')
plt.show()

# plt.figure()
# plt.imshow(psf_LP/psf_LP.max() - psf_LP_skip/psf_LP_skip.max(), norm=colors.LogNorm(vmin=1e-6,vmax=1),extent=[x_min,x_max,x_min,x_max])
# plt.colorbar()
# plt.xlabel(r"$\lambda/D$")
# plt.xticks(rotation=45, ha="right")
# plt.ylabel(r"$\lambda/D$")
# plt.title(f'Différence entre la PSF longue pose et \n PSF longue pose pour une image toutes les {z} ms \n soit {M/z} images')
# plt.show()


plt.figure()
azav_lp = AZAV(psf_LP/psf_LP.max(),owa,0.1)[0]
azav_lp_skip = AZAV(psf_LP_skip/psf_LP_skip.max(),owa,0.1)[0]
plt.semilogy(np.linspace(0,x_max,nbr_pix//2),azav_lp,label='PSF LP')
plt.semilogy(np.linspace(0,x_max,nbr_pix//2),azav_lp_skip,label='PSF LP skip')
plt.legend()
plt.xlabel(r"$\lambda/D$")
plt.ylabel("Contraste")
plt.title(f"AZAV de la PSF longue pose et \n PSF longue pose pour une image toutes les {z} ms \n soit {M/z} images")
plt.show()

#%%
# ! Calcul du contraste

def cr(im,iwa,owa,N,nbr_pix):
    y, x = np.indices((nbr_pix, nbr_pix))
    cx, cy = nbr_pix // 2, nbr_pix // 2

    r_int = iwa*2*N  
    r_ext = owa*2*N 

    # Calcul de la distance de chaque pixel au centre
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Masque : True pour les pixels entre les deux rayons
    mask = (r >= r_int) & (r <= r_ext)
    # Masque pour le quadrant supérieur droit (par exemple)
    # Quadrant selection masks
    mask_qua_ur = (r >= r_int) & (r <= r_ext) & (x >= cx) & (y <= cy)   # upper right
    mask_qua_ul = (r >= r_int) & (r <= r_ext) & (x <= cx) & (y <= cy)   # upper left
    mask_qua_lr = (r >= r_int) & (r <= r_ext) & (x >= cx) & (y >= cy)   # lower right
    mask_qua_ll = (r >= r_int) & (r <= r_ext) & (x <= cx) & (y >= cy)   # lower left

    # Example: choose which quadrant to use
    mask = mask_qua_ur  # Change to mask_qua_ul, mask_qua_lr, or mask_qua_ll as needed
    plt.figure()
    plt.imshow(im/im.max(), norm=colors.LogNorm(vmin=1e-6))
    plt.title("PSF avec masque quadrant")
    plt.colorbar()
    plt.contour(mask, colors='r', linewidths=0.5)
    return np.mean((im/im.max())[mask])


print(f'{cr(psf_LP_skip,iwa,owa,N,nbr_pix):2e}')



#%%
plt.figure()
plt.imshow(psf_aper/psf_aper.max(), norm=colors.LogNorm(vmin=1e-6),extent=[x_min,x_max,x_min,x_max])
plt.colorbar()
plt.xlabel(r"$\lambda/D$")
plt.xticks(rotation=45, ha="right")
plt.ylabel(r"$\lambda/D$")
plt.title('PSF coronographique')
plt.show()
#%%
class OTF:
    def __init__(self, psf):
        self.psf= psf
        self.fov = psf.fov
        self.N = psf.N

    def OTF_1(self): #soit le produit de OTF_telescope et OTF_atm, ou autocorrelation onde incidente capturée par le telescope
        otf = ft_BASIC(self.psf.PSF(), self.fov, 3*1132/1024,self.N, direction=1)
        #otf = np.fft.fft2(self.psf.PSF())
        #otf = np.fft.fftshift(otf)
        return otf/otf.max()

# f = 10
# j = 1 #j=20 à noter#50 aussi pas mal
# size = 950 // j
# mat = np.zeros((950, 950), dtype=bool)
# start = (950 - size) // j
# end = start + size
# mat[start:end, start:end] = True
# sq = mat
#%%
# fenetre de hamming : 0.54 - 0.46 cos(2pi*x/T)
# fenetre de hann : O.5 - 0.5cos(2pi*x/T)
def hamming(x, y, T):
    r = np.sqrt(x**2 + y**2)
    mask = r <= T
    window = np.zeros_like(r)
    window[mask] = 0.5 + 0.5*np.cos(2 * np.pi * r[mask] / (2*T))
    # * Y'a un petit problème d'arrondi ou c'est pas exactement 1 le max mais 0.99998

    return window

def sup_gauss(x,y,s,ord):
    r = np.sqrt(x**2 + y**2)
    mask = r <= s
    window = np.zeros_like(r)
    window[mask] = 0.5 + 0.5*np.cos(2 * np.pi * r[mask] / (2*T))
    # * Y'a un petit problème d'arrondi ou c'est pas exactement 1 le max mais 0.99998

    return window
    

R = nbr_pix
grid_x, grid_y = np.meshgrid(np.linspace(-R, R, nbr_pix), np.linspace(-R, R, nbr_pix)) # correctino par 4000 pour éviter que ça fasse plus la meme taille en nombre de pixel que la PSF à cause du padding de 2000 ajouté
# pour avoir tout le cercle jusqu'au bord il faut T=2*R
# si R trop petit ça zoom trop à l'intéireur 
# si T plus petit que 2*R alors ça fait plus petit


fact = 60#fov//2 # ! taille en lambda/D du rayon du filtre, en dehors le champ sera nul

T = 2*(2*N)*fact# nombre de pixel du diamètre du filtre criculaire
# * En gros c'est la conversion de lambda/D vers les pixels pour faire le calcul de la fenetre de Hann
hamm = hamming(grid_x, grid_y, T) 
plt.figure()
plt.imshow(hamm,extent=[x_min,x_max,x_min,x_max])#,norm=colors.LogNorm())
plt.xlabel(r"$\lambda/D$")
plt.ylabel(r"$\lambda/D$")
plt.title("Fenêtre de Hann")
plt.colorbar()
plt.show()
#%%
f = 3# facteur de largeur du champ de vue de l'OTF

def OTF(psf,fov,nbr_pix): #soit le produit de OTF_telescope et OTF_atm, ou autocorrelation onde incidente capturée par le telescopes
    otf = ft_BASIC(psf, fov, f*1132/1024, nbr_pix, direction=1)#/ft_BASIC(sq, fov, 1132/1024,N,direction=-1)
    return otf#/otf.max()

#%%
# * Choix de la PSF et permet bien l'affichage

psf_name = 'longue pose'

if psf_name == 'longue pose':
    psf = psf_LP*hamm
else:
    psf = psf_tel

# * Affichage de la PSF

plt.figure()
plt.imshow(psf/psf.max(), norm=colors.LogNorm(vmin=1e-6),extent=[x_min,x_max,x_min,x_max])
plt.xlabel(r"$\lambda/D$")
plt.ylabel(r"$\lambda/D$")
plt.colorbar()
plt.title('psf utilisé pour les calculs')


otf = OTF(psf,fov,nbr_pix)

# * Affichage de la mtf, normalisé par convention
plt.figure()
mtf = np.abs(otf)
plt.imshow(mtf/mtf.max(), norm=colors.LogNorm(vmin=1e-10),extent=[f/x_min,f/x_max,f/x_min,f/x_max])
plt.title(f"MTF 2D de la PSF {psf_name}")
#plt.axis('off')
plt.xlabel(r"$D/\lambda$")
plt.ylabel(r"$D/\lambda$")
plt.colorbar()
#%%
# ! ça galère a calculer l'azav de la mtf :(
# plt.figure()
# az=AZAV(mtf/mtf.max(),owa,owa)[0]
# plt.semilogy(np.linspace(0,1/x_max,nbr_pix//2),az) # affichage moy. azimutal MTF PSF longue pose
# plt.xlabel(r'$D/\lambda$')
# plt.title('Moyenne azimutal de la PSF longue pose')


#%%
# shape = np.shape(mtf)
# rows = []
# for i in range(200):
#     rows.append(mtf[i+750,:])
# mtf_mean = np.stack(rows)
# mean_row = np.zeros(200)
# for i, row in enumerate(mtf_mean):
#     mean_row[i] = np.mean(row)
#     #plt.scatter(i,np.mean(row))#, label=f'Tranche {i+1}')

# plt.figure()
# plt.plot(np.linspace(0, 3*x_max, 200),mean_row)
# plt.title("Valeur de chaque tranche verticale sur la partie droite de la MTF")
# plt.ylabel("Valeur contraste sur la MTF")
# plt.xlabel(r"$D/\lambda$")


# plt.figure()
# plt.semilogy(np.linspace(0, 3*x_max, 200), mean_row)
# plt.title("Moyenne de 200 tranches verticales sur la partie droite de la MTF")
# plt.ylabel("Valeur contraste sur la MTF")



# * Affichage de la moyenne azimutal de la MTF
# plt.figure()
# azav_test = AZAV(mtf, owa,1/owa)[0]
# plt.semilogy(azav_test)
# plt.title(f"Moyenne azimutal de la MTF de la PSF {psf_name}")

plt.show()

#%%
class DSP:
    def __init__(self, psf_LP, psf_pup,fov,N):
        self.psf_LP = psf_LP
        self.psf_pup= psf_pup
        self.fov=fov
        self.N=N

    def DSP(self): # ? Est ce que le plot de la DPS est correct ? -> Problème dès les OTF finalement
    # * Par contre comme on utilise les OTF, là ça doit effectivment être possible de récupérer
        f_gr = 2 #facteur de grandissement
        # ! ÊTRE SUR DE COMPRENDRE POURQUOI C'EST LE FACTEUR DE TAILE DE LA MTF QUI CHANGE TOUT 
        otf_lp = ft_BASIC(self.psf_LP, self.fov, f_gr*1132/1024, self.N, direction=1)# OTF PSF longue pose        
        #//a = np.where(a>=2e-4,a,7e-6)
        otf_tel = ft_BASIC(self.psf_pup, self.fov, f_gr*1132/1024, self.N, direction=1) # OTF PSF corono
        #//b = np.where(b>=2e-4,b,7e-6)
        plt.figure()
        plt.imshow(np.abs(otf_lp)/np.abs(otf_lp).max(),norm=colors.LogNorm(vmin=1e-6),extent=[f_gr/x_min,f_gr/x_max,f_gr/x_min,f_gr/x_max]) # affichage MTF PSF longue pose
        plt.colorbar()
        plt.title('MTF de la PSF longue pose')
        plt.xlabel(r'$D/\lambda$')
        plt.ylabel(r'$D/\lambda$')
        #plt.figure()
        #plt.semilogy(np.linspace(0,5/x_max,nbr_pix//2),AZAV(np.abs(otf_lp)/np.abs(otf_lp).max(),owa,1/owa)[0]) # affichage moy. azimutal MTF PSF longue pose
        #plt.xlabel(r'$D/\lambda$')
        #plt.title('Moyenne azimutal de la PSF longue pose')
        plt.show()
        #//r = ft_BASIC(self.psf_LP,7*self.fov, 1132/1024, self.N, direction=1)/ft_BASIC(self.psf_pup,7*self.fov, 1132/1024, self.N, direction=1) # * complexe
        r = otf_lp/otf_tel # * complexe
        Re = np.log(np.abs(r))*hamm
        Im = np.angle(r)*hamm
        tf_inv = (Re + 1j*Im)#*hamm
        #tf_inv = Re*np.exp(1j*Im)
        plt.figure()
        plt.imshow(Re)
        plt.title('Affichage partie réelle TF-1 DSP')
        plt.figure()
        plt.imshow(Im)
        plt.title('Affichage partie imaginaire TF-1 DSP')       
        plt.show()
        return ft_BASIC(tf_inv, f_gr*1132/1024, self.fov, self.N, direction=1)

# ! HSP1 n'est pas le bon ou il a été modifié entre temps


#%%
# PSF_aper = PSF_utils(aper,7*fov,nbr_pix,N,norm=True)#,OAtime=100)

# psf = PSF_aper.PSF()
# plt.imshow(psf, norm=colors.LogNorm())


# plt.figure()
# otf = OTF(PSF_aper).OTF()
# plt.imshow(np.abs(otf),norm=colors.LogNorm(vmin=1e-6,vmax=1))
# plt.figure()
# plt.semilogy(AZAV(np.abs(otf),owa,1/owa)[0])
# plt.show()



#%%
# * Calcul de la DSP
psf_LP_hamm = psf_LP*hamm
psf_tel_hamm = psf_tel*hamm
a = 0.5
dsp = DSP(psf_LP_hamm,psf_tel_hamm,fov,nbr_pix).DSP()
plt.figure()
plt.imshow(np.abs(dsp)/np.abs(dsp).max(),norm=colors.LogNorm(vmin=1e-10),extent=[a*x_min,a*x_max,a*x_min,a*x_max]) # ! pas encore correct, à faire attention
plt.colorbar()
plt.xlabel(r'$D/\lambda$')
plt.ylabel(r'$D/\lambda$')
plt.title('DSP')
plt.show()


#%%
def DSP2PSF(psf_pup,dsp):
    plt.figure()
    otf_pup = ft_BASIC(psf_pup,fov, 2*1132/1024, nbr_pix, direction=1)#*hamm
    plt.imshow(np.abs(otf_pup),norm=colors.LogNorm())
    plt.title("OTF pupille")
    ft_DSP = np.abs(ft_BASIC(dsp, 2*1132/1024, fov, nbr_pix, direction=1))
    kernel = np.exp(1e5*(ft_DSP-ft_DSP[nbr_pix//2,nbr_pix//2]))
    plt.figure()
    plt.imshow((kernel))#,norm=colors.LogNorm())
    plt.colorbar()
    plt.title("OTF Atmosphérique")
    plt.show()
    psf_fin = ft_BASIC(otf_pup*kernel,2*1132 / 1024, fov, nbr_pix, direction=-1)
    return psf_fin

# * Affichage de la PSF reconstruite depuis la DSP avec la fonction DSP2PSF au dessus
plt.figure()
psf_reconstr = np.abs(DSP2PSF(psf_tel, dsp))
plt.imshow(psf_reconstr/psf_reconstr.max(),norm=colors.LogNorm(vmin=1e-6),extent=[x_min,x_max,x_min,x_max])
plt.colorbar()
plt.title("PSF reconstruite depuis la DSP")
plt.xlabel(r'$\lambda/D$')
plt.ylabel(r'$\lambda/D$')

# * Affichage de la PSF initial, celle qu'on cherche à retrouver 
# plt.figure()
# plt.imshow(psf/psf.max(),norm=colors.LogNorm(vmin=1e-8))
plt.figure()
plt.imshow(psf_LP/psf_LP.max(),norm=colors.LogNorm(vmin=1e-6),extent=[x_min,x_max,x_min,x_max])
plt.colorbar()
plt.title("PSF longue pose initiale")
plt.xlabel(r'$\lambda/D$')
plt.ylabel(r'$\lambda/D$')
plt.show()

# * Affichage de la PSF coronographique
plt.figure()
plt.imshow(psf_tel/psf_tel.max(),norm=colors.LogNorm(vmin=1e-6),extent=[x_min,x_max,x_min,x_max])
plt.colorbar()
plt.title("PSF coronographique initiale")
plt.xlabel(r'$\lambda/D$')
plt.ylabel(r'$\lambda/D$')
plt.show()
#%%

#! Calcul de l'OTF et application d'un masque circulaire
apod = Aperture('HSP2')
psf_apod = PSF_utils(apod,fov,nbr_pix).PSF()

with fits.open(f'fits/PSF/PSF_500_ELT_8.fits') as hdul:
    psf_LP_500 = hdul[0].data
with fits.open(f'fits/PSF/PSF_1000_ELT_8.fits') as hdul:
    psf_LP_1000 = hdul[0].data
with fits.open(f'fits/PSF/PSF_1500_ELT_8.fits') as hdul:
    psf_LP_1500 = hdul[0].data
with fits.open(f'fits/PSF/PSF_2000_ELT_8.fits') as hdul:
    psf_LP = hdul[0].data


otf_atmo_500 = ft_BASIC(psf_LP_500,fov, 2,nbr_pix,direction=-1)/ft_BASIC(psf_tel,fov,2,nbr_pix,direction=-1)
otf_atmo_1000 = ft_BASIC(psf_LP_1000,fov, 2,nbr_pix,direction=-1)/ft_BASIC(psf_tel,fov,2,nbr_pix,direction=-1)
otf_atmo_1500 = ft_BASIC(psf_LP_1500,fov, 2,nbr_pix,direction=-1)/ft_BASIC(psf_tel,fov,2,nbr_pix,direction=-1)

#%%
otf_atmo_2000 = np.abs(ft_BASIC(psf_LP,fov, 1.8,nbr_pix,direction=-1)/ft_BASIC(psf_tel,fov,1.8,nbr_pix,direction=-1)) # comme le LP à été calculé pour 2000ms




#%%
# ! masque polygone, de N côté
def n(x,y,N):
    m = 0
    for k in range(0,N):
        v = np.abs(np.cos(2*np.pi*k/N)*x+np.sin(2*np.pi*k/N)*y)
        if v > m:
            m=v
    return m
n = np.vectorize(n)
grid_x,grid_y = np.meshgrid(np.linspace(-1000/964,1000/964,nbr_pix),np.linspace(-1000/964,1000/964,nbr_pix))
poly = n(grid_x,grid_y,12) <= 1

plt.imshow(poly)
#%%
R = 1
grid_x, grid_y = np.meshgrid(np.linspace(-R, R, nbr_pix), np.linspace(-R, R, nbr_pix))
mask = np.sqrt(grid_x**2+grid_y**2) <= 1.1

otf_atmo_filtr_500 = otf_atmo_500*mask
otf_atmo_filtr_1000 = otf_atmo_1000*mask
otf_atmo_filtr_1500 = otf_atmo_1500*mask
otf_atmo_filtr_2000 = otf_atmo_2000*mask
otf_atmo_filtr_dod_500 = otf_atmo_500*poly
otf_atmo_filtr_dod_1000 = otf_atmo_1000*poly
otf_atmo_filtr_dod_1500 = otf_atmo_1500*poly
otf_atmo_filtr_dod_2000 = otf_atmo_2000*poly

#%%
fits.writeto('fits/MTF/MTF_ELT_8.fits', otf_atmo_2000)
fits.writeto('fits/MTF/MTF_ELT_8_circ.fits', otf_atmo_filtr_2000)
fits.writeto('fits/MTF/MTF_ELT_8_dod.fits', otf_atmo_filtr_dod_2000)
#%%
plt.figure()
plt.imshow(np.abs(otf_atmo_2000),norm=colors.LogNorm())
plt.title('OTF atmosphérique')
plt.figure()
plt.imshow(np.abs(otf_atmo_filtr_2000),norm=colors.LogNorm())
plt.title('OTF atmosphérique filtrée')


# * c'est bien que l'autocorrelation de la pupille de l'ELT donne cette forme d'héxagone donc c'est normal :D

#%%
#! affichage de la MTF en valeur absolue
#mtf_atmo = np.abs(otf_atmo_filtr)

plt.figure()
plt.title("Moyenne azimutal de la MTF atmosphérique \npour différents temps d'OA")
plt.xlabel(r"$D/\lambda$")
plt.ylabel("u.a. (MTF non normalisé)")
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV((otf_atmo_filtr_500).real,owa,1)[0],label="500ms (réelle)",color='blue')
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV(np.abs(otf_atmo_filtr_500),owa,1)[0],label="500ms (abs)",linestyle='dashed', color='blue')
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV((otf_atmo_filtr_1000).real,owa,1)[0],label="1000ms (réelle)",color='orange')
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV(np.abs(otf_atmo_filtr_1000),owa,1)[0],label="1000ms (abs)",linestyle='dashed',color='orange')
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV((otf_atmo_filtr_1500).real,owa,1)[0],label="1500ms (réelle)",color='green')
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV(np.abs(otf_atmo_filtr_1500),owa,1)[0],label="1500ms (abs)",linestyle='dashed',color='green')
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV((otf_atmo_filtr_2000).real,owa,1)[0],label="2000ms (réelle)",color='red')
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV(np.abs(otf_atmo_filtr_2000),owa,1)[0],label="2000ms (abs)",linestyle='dashed',color='red')
plt.legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
im_list = [
    otf_atmo_filtr_500.imag,
    otf_atmo_filtr_1000.imag,
    otf_atmo_filtr_1500.imag,
    otf_atmo_filtr_2000.imag
]
titles = ["500ms", "1000ms", "1500ms", "2000ms"]

for ax, im, title in zip(axs.flat, im_list, titles):
    imshow = ax.imshow(im, extent=[2/x_min, 2/x_max, 2/x_min, 2/x_max])
    ax.set_title(f"Imag(MTF) {title}")
    ax.set_xlabel(r"$D/\lambda$")
    ax.set_ylabel(r"$D/\lambda$")
    plt.colorbar(imshow, ax=ax, fraction=0.046, pad=0.04)#, vmin=-0.11,vmax=0.11)
    imshow.set_clim(-0.11, 0.11)
plt.tight_layout()
plt.show()


#%%

plt.figure()
plt.imshow(np.abs(otf_atmo_filtr),norm=colors.LogNorm(),extent=[2/x_min,2/x_max,2/x_min,2/x_max])
plt.title("MTF atmosphérique (abs(MTF))")
plt.xlabel(r"$D/\lambda$")
plt.ylabel(r"$D/\lambda$")
plt.colorbar()

plt.figure()
plt.imshow((otf_atmo_filtr.real),norm=colors.LogNorm(),extent=[2/x_min,2/x_max,2/x_min,2/x_max])
plt.title("MTF atmosphérique (real(MTF))")
plt.xlabel(r"$D/\lambda$")
plt.ylabel(r"$D/\lambda$")
plt.colorbar()

plt.figure()
plt.imshow((otf_atmo_filtr.imag),extent=[2/x_min,2/x_max,2/x_min,2/x_max])
plt.title("MTF atmosphérique (imag(MTF))")
plt.xlabel(r"$D/\lambda$")
plt.ylabel(r"$D/\lambda$")
plt.colorbar()

plt.figure()
plt.title("Moyenne azimutal de la MTF atmosphérique (real(MTF))")
plt.xlabel(r"$D/\lambda$")
plt.ylabel("u.a. (MTF non normalisé)")
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV((otf_atmo_filtr).real,owa,1)[0])
plt.show()


#! affichage de l'azav
plt.figure()
#azav_mtf = AZAV(mtf_atmo, owa, 1)[0]
plt.title("Moyenne azimutal de la MTF atmosphérique")
plt.xlabel(r"$D/\lambda$")
plt.ylabel("u.a. (MTF non normalisé)")
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV(np.abs(otf_atmo_filtr),owa,1)[0],label="Valeur absolue")
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV((otf_atmo_filtr).real,owa,1)[0],label="Partie réelle")
plt.legend()
plt.show()

plt.figure()
plt.semilogy(np.linspace(0,2/x_max,nbr_pix//2),AZAV(np.abs(otf_atmo_filtr),owa,1)[0]-AZAV((otf_atmo_filtr).real,owa,1)[0])
plt.title("Différence entre les moyennes azimutal\nde la valeur absolue et la partie réelle de la MTF atmosphérique")

#%%
def OTF2PSF(otf_atmo,psf_tel): # en remplacement de DSP2PSF
    # ! pour l'instant on suppose que otf_atmo = mtf_atmo
    otf_rec = otf_atmo*ft_BASIC(psf_tel,fov,2,nbr_pix,direction=1)
    psf_rec = np.abs(ft_BASIC(otf_rec, 2,fov,nbr_pix,direction=1))
    return psf_rec
#%%
# mtf_atmo_filtr_2000 = np.abs(otf_atmo_filtr_2000)
# plt.figure()
# plt.imshow(mtf_atmo_filtr_2000, norm=colors.LogNorm())
# plt.colorbar()
# plt.title("MTF atmosphérique (2s d'OA)")

#%%
# ! Recréation de la PSF à partir d'une ouverture et d'une MTF atmosphérique

choix_MTF = "ELT_8_circ" #HSP2_circ, #HSP2_dod, #MTF_ELT, #MTF_ELT_20_
choix_aper = "ELT_8" #ELT #ELT_20

# * Permet de calculer correctement la PSF longue pose par les deux MTF
if choix_aper == "ELT_8":
    aper = psf_tel
else:
    aper = psf_apod

with fits.open(f'fits/MTF/MTF_{choix_MTF}.fits') as hdul:
    MTF = hdul[0].data


#%%
# TODO Faire le découpage du quart en bas à droite de la MTF



MTF_ud = np.flipud(MTF)
MTF_sym = np.fliplr(MTF_ud)

MTF_sum = MTF+MTF_sym
MTF_mean = MTF_sum/2

plt.figure()
plt.imshow(MTF_mean,norm=colors.LogNorm())
plt.colorbar()
plt.title("Quadrant de la MTF utilisé pour AMPL")
plt.show()

quad_MTF = MTF_mean[nbr_pix//2:, nbr_pix//2:]

# * Permet d'afficher le quadrant de MTF considéré
plt.figure()
plt.imshow(quad_MTF,norm=colors.LogNorm())
plt.colorbar()
plt.title("Quadrant de la MTF utilisé pour AMPL")
plt.show()

#%%
with open(f"AMPL/quad_MTF_{choix_MTF}_edge.dat","w") as file:
    quad_MTF_txt = np.array2string(quad_MTF)
    file.write(quad_MTF_txt)
    file.close()
#%%
fits.writeto(f"quad_MTF_{choix_MTF}_edge.fits", quad_MTF, overwrite=True)


#%%

# * Permet d'afficher la MTF atmosphérique utilisée
fig, ax = plt.subplots()
imshow = ax.imshow(MTF,extent=[2/x_min,2/x_max,2/x_min,2/x_max])
imshow.set_clim(vmin=0.7, vmax=0.9) # * limitaton de la colorbar pour correctement voir l'intérieur
plt.title(f"MTF atmosphérique considéré : {choix_MTF}")
plt.xlabel(r"$D/\lambda$")
plt.ylabel(r"$D/\lambda$")
plt.colorbar(imshow, ax=ax)

# * Permet de calculer la PSF reconstruite à partir des deux MTF
psf_rec = OTF2PSF(MTF, aper)

plt.figure()
plt.imshow(psf_rec/psf_rec.max(),norm=colors.LogNorm(vmin=1e-8),extent=[x_min,x_max,x_min,x_max])
plt.xlabel(r"$\lambda/D$")
plt.ylabel(r"$\lambda/D$")
plt.colorbar()
plt.title(f"PSF reconstruite (APER : {choix_aper} - MTF : {choix_MTF})")

#%%
# ! Différence entre la PSF reconstruite et la PSF longue pose

# * Ouverture de la PSF longue pose associé à l'ouverture

with fits.open(f'fits/PSF/PSF_2000_{choix_aper}.fits') as hdul:
    psf = hdul[0].data

# * Affichage de la différence entre les PSF
# plt.figure()
# plt.imshow(psf_rec/psf_rec.max()-psf/psf.max(),norm=colors.LogNorm(vmin=1e-8),extent=[x_min,x_max,x_min,x_max])
# plt.xlabel(r"$\lambda/D$")
# plt.ylabel(r"$\lambda/D$")
# plt.colorbar()
# plt.title(f"Différence entre la PSF reconstruite\net la PSF longue pose (APER : {choix_aper} - MTF : {choix_MTF})")

azav_rec = AZAV(psf_rec/psf_rec.max(), owa, 1)[0]
azav_ini = AZAV(psf/psf.max(), owa, 1)[0]

plt.figure()
plt.semilogy(np.linspace(0,x_max,nbr_pix//2),azav_rec, label='AZAV PSF reconstruite')
plt.semilogy(np.linspace(0,x_max,nbr_pix//2),azav_ini, label='AZAV PSF longue pose')
plt.xlabel(r"$\lambda/D$")
plt.ylabel("Contraste")
plt.legend()
plt.title("Comparaison des AZAV des PSF reconstruite et LP")


#%%
# ! Affichage de la PSF longue pose et la PSF coronographique associé à l'ouverture 

# * PSF longue pose
plt.figure()
plt.imshow(psf/psf.max(), norm=colors.LogNorm(vmin=1e-8),extent=[x_min,x_max,x_min,x_max])
plt.title(f"PSF longue pose ({choix_aper})")
plt.xlabel(r"$\lambda/D$")
plt.ylabel(r"$\lambda/D$")
plt.colorbar()

# * PSF coronographique
plt.figure()
plt.imshow(aper/aper.max(), norm=colors.LogNorm(vmin=1e-8),extent=[x_min,x_max,x_min,x_max])
plt.xlabel(r"$\lambda/D$")
plt.ylabel(r"$\lambda/D$")
plt.title(f"PSF coronographique ({choix_aper})")
plt.colorbar()
#%%
half_psf_rec = psf_rec[:nbr_pix//2,:]
half_psf_LP = psf_LP[nbr_pix//2:,:]
matrix_compar = np.zeros((nbr_pix, nbr_pix))
matrix_compar[:nbr_pix//2]=half_psf_rec
matrix_compar[nbr_pix//2:]=half_psf_LP 

plt.figure()
plt.imshow(matrix_compar,norm=colors.LogNorm(vmin=1e-8))


#%%
half_psf_rec = psf_rec#[:nbr_pix//2,:]
half_psf_LP = psf_LP#[:nbr_pix//2,:]
matrix_diff = np.zeros((nbr_pix//2, nbr_pix))
matrix_diff=half_psf_rec/half_psf_rec.max() - half_psf_LP/half_psf_LP.max()

plt.figure()
plt.imshow(matrix_diff,norm=colors.LogNorm(vmin=1e-8))
plt.colorbar()
plt.title("Différence entre les deux PSF (rec - LP)")
# ! essayer de bien interpreter la différence entre les deux PSF

#%%