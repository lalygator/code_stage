#%%
import numpy as np
from astropy.io import fits
from utils_OA import READ_DAT, SQ, timeit, ft_BASIC, AZAV
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
from matplotlib.path import Path
import matplotlib.colors as colors


# outils de selectrion des apertures
## contiendra apodiseur, pupille elt et une ouverture circulaire opur des test, mais a ne pas utiliser avec les résidus car sont suelement adapté à 'louvertue de l'elt

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
    def __init__(self, aperture, fov, N, OAtime=None,norm=False):
        self.aperture = aperture
        self.fov = fov # champ de vue observée totale
        self.N = N # nombre de pixel
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
                ecran = np.empty((self.OAtime//2,1132,1132))
                if self.OAtime <= 200:
                    file_path_name = './interp_funcs_500_600.pkl'
                    with open(file_path_name, 'rb') as file:
                        loaded_interp_funcs = pickle.load(file)
                    phase_interps = loaded_interp_funcs
                    ecran = np.array(phase_interps)
                elif self.OAtime <= 1000:
                        file_path_name = './phase_interps_0_500.pkl'
                        with open(file_path_name, 'rb') as file:
                            loaded_interp_funcs = pickle.load(file)
                        phase_interps = loaded_interp_funcs
                        ecran = np.array(phase_interps)
                        #file_path_name = './phase_interps_500_1000.pkl'
                elif self.OAtime<=2000:
                    file_path_name_1 = './phase_interps_0_500.pkl'
                    with open(file_path_name_1, 'rb') as file:
                        loaded_interp_funcs = pickle.load(file)
                    ecran[:500,:,:] = loaded_interp_funcs
                    file_path_name_2 = './phase_interps_500_1000.pkl'
                    with open(file_path_name_2, 'rb') as file:
                        loaded_interp_funcs = pickle.load(file)
                    ecran[500:self.OAtime//2,:,:] = loaded_interp_funcs[:np.abs(self.OAtime//2-500),:,:]
                else:
                    raise ValueError("Ce temps d'écran de phase n'est pas disponible")
                
                # with open(file_path_name, 'rb') as file:
                #     loaded_interp_funcs = pickle.load(file)
                # phase_interps = loaded_interp_funcs
            
                m = self.OAtime//2 # nombre d'écran de phase considéré 

                mat = self.aperture.matrix() # matrice de la pupille/de l'apodiseur
                calc = ecran[:m,:,:] # array des écrans de phases considéré
                phase = np.exp(2j*np.pi*calc/1.65e-6)
                A = np.multiply(phase,mat)
                im = np.zeros((m,self.N,self.N)) # allocation initial de la mémoire pour le calcul
                for i, A_i in enumerate(A):
                    im[i,:,:] = (ft_BASIC(A_i, 1132 / 1024, self.fov, self.N, direction=1).real)**2
                    #print(f'Image {i} fini')
                psf = np.mean(im, axis=0)
                self.psf = psf
        if self.norm == True:
            return self.psf/self.psf.max()
        return self.psf

# * Choix de la pupille
aper = Aperture('HSP2')

# * Choix des paramètres
owa = 38 # OWA de l'apodiseur considéré
N = 2 # résolution, valeur minimale pour etre a Nyquist
fov = int(5*2.5*owa) # champ de vue de la psf
nbr_pix = int(N * 2 * fov) # nombre de pixels, minimum 2 par lambda/D pour Nyquist à N=1
M = 2000 # temps d'OA considéré, en ms #//1min pour faire 1500 avec np multiply
psf_LP = PSF_utils(aper,fov,nbr_pix,M).PSF()
psf_tel = PSF_utils(aper,fov,nbr_pix).PSF()
#%%
psf_LP_500 = PSF_utils(aper,fov,nbr_pix,500).PSF()
psf_LP_1000 = PSF_utils(aper,fov,nbr_pix,1000).PSF()
psf_LP_1500 = PSF_utils(aper,fov,nbr_pix,1500).PSF()
psf_LP_2000 = psf_LP#PSF_utils(aper,fov,nbr_pix,2000).PSF()
fits.writeto('psf_LP_2000_HSP2.fits', psf_LP_2000, overwrite=True)


#%%
# * Affichage de la PSF

b=1 #lambda/D
x_min=y_min=-b*fov/2
x_max=y_max=b*fov/2
#%%
plt.figure()
plt.imshow(psf_LP/psf_LP.max(), norm=colors.LogNorm(vmin=1e-6),extent=[x_min,x_max,x_min,x_max])
plt.colorbar()
plt.xlabel(r"$\lambda/D$")
plt.xticks(rotation=45, ha="right")
plt.ylabel(r"$\lambda/D$")
plt.title('PSF longue pose')
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

#! Calcul de l'OTF et application d'un masque

otf_atmo_500 = ft_BASIC(psf_LP_500,fov, 2,nbr_pix,direction=-1)/ft_BASIC(psf_tel,fov,2,nbr_pix,direction=-1)
otf_atmo_1000 = ft_BASIC(psf_LP_1000,fov, 2,nbr_pix,direction=-1)/ft_BASIC(psf_tel,fov,2,nbr_pix,direction=-1)
otf_atmo_1500 = ft_BASIC(psf_LP_1500,fov, 2,nbr_pix,direction=-1)/ft_BASIC(psf_tel,fov,2,nbr_pix,direction=-1)
otf_atmo_2000 = np.abs(ft_BASIC(psf_LP,fov, 2,nbr_pix,direction=-1)/ft_BASIC(psf_tel,fov,2,nbr_pix,direction=-1)) # comme le LP à été calculé pour 2000ms


R = 1
grid_x, grid_y = np.meshgrid(np.linspace(-R, R, nbr_pix), np.linspace(-R, R, nbr_pix))
mask = np.sqrt(grid_x**2+grid_y**2) <= 1

#%%
def n(x,y,N):
    m = 0
    for k in range(0,N):
        v = np.abs(np.cos(2*np.pi*k/N)*x+np.sin(2*np.pi*k/N)*y)
        if v > m:
            m=v
    return m
n = np.vectorize(n)
grid_x,grid_y = np.meshgrid(np.linspace(-1000/964,1000/964,1900),np.linspace(-1000/964,1000/964,1900))
boule = n(grid_x,grid_y,12) <= 1

plt.imshow(boule)
#%%
otf_atmo_filtr_500 = otf_atmo_500*mask
otf_atmo_filtr_1000 = otf_atmo_1000*mask
otf_atmo_filtr_1500 = otf_atmo_1500*mask
otf_atmo_filtr_2000 = otf_atmo_2000*boule


fits.writeto('mtf_atmo_filtr_dod_2000_apod.fits', otf_atmo_filtr_2000, overwrite=True)
fits.writeto('mtf_atmo_2000_apod.fits', otf_atmo_2000, overwrite=True)




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
    
mtf_atmo_filtr_2000 = np.abs(otf_atmo_filtr_2000)
plt.figure()
plt.imshow(mtf_atmo_filtr_2000, norm=colors.LogNorm())
plt.colorbar()
plt.title("MTF atmosphérique (2s d'OA)")
#%%

apod = Aperture('HSP2')
psf_apod = PSF_utils(apod,fov,nbr_pix).PSF()

#%%
#!!!!!
# Open the FITS file and read the MTF data
with fits.open('mtf_atmo_filtr_dod_2000_apod.fits') as hdul:
    mtf_atm = hdul[0].data

psf_rec = OTF2PSF(mtf_atm, psf_apod)
plt.figure()
plt.imshow(psf_rec/psf_rec.max(),norm=colors.LogNorm(vmin=1e-6))
plt.colorbar()
plt.title("PSF reconstruite")
#!!!!!


#%%
plt.figure()
plt.imshow(psf_LP/psf_LP.max(), norm=colors.LogNorm(vmin=1e-6))
plt.title("PSF longue pose")

plt.colorbar()

plt.figure()
plt.imshow(psf_apod/psf_apod.max(), norm=colors.LogNorm(vmin=1e-6))
plt.title("PSF coronographique")
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
# plt.figure()
# test = pitie - psf_tel
# plt.imshow(test,norm=colors.LogNorm())

# outil pour PSF
# - aclul de la psf
# - azav
# - affichage

# outils pour OTF
# - aclul de l'otf
# - azav
# - affichage

# outils pour DSP
# - calcul de dsp
# - verification de la PSF retrived de la dsp
# - affichage