import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import convolve
from numpy import linalg
import matplotlib.colors as colors
from utils_basic import HALO

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')
        return result
    return wrapper

class Pupil:
    def __init__(self, iwa, owa, t):
        self.iwa = iwa
        self.owa = owa
        self.t = t

    def matrix(self):
        A = READ_DAT(self.iwa, self.owa, self.t,name='ELT')
        A = A.reshape((566, 566))
        return A

    def affiche(self, save=False):
        A = READ_DAT(self.iwa, self.owa, self.t)
        A = A.reshape((566, 566))
        plt.imshow(SQ(A), cmap='binary_r')
        plt.axis('off')
        plt.show()

def READ_DAT(iwa=0, owa=0, T=0, angle=0.0, name='ELT'):
    if name == 'ELT' :
        return np.fromfile(f'./ELT_1132_M1=2_M4=4_ROT=5.dat', dtype=float, sep='\n')
    else:
        return np.fromfile(f'./{iwa}_{owa}_{T}.dat', dtype=float, sep='\n')    

def SQ(A):
    A = np.concatenate((np.fliplr(A), A), axis=1)
    return np.concatenate((np.flipud(A), A), axis=0)


def ft_BASIC(Ein, W1, W2, n2, direction):
    nx, ny = np.shape(Ein)
    dx = W1 / nx
    du = W2 / n2
    x = (np.linspace(-nx / 2, nx / 2, nx, endpoint=False) + 0.5) * dx
    u = (np.linspace(-n2 / 2, n2 / 2, n2, endpoint=False) + 0.5) * du
    x_u = np.outer(x, u)
    Eout = np.dot(np.transpose(np.exp(-direction * 2. * np.pi * 1j * x_u)), Ein)
    Eout = np.dot(Eout, np.exp(-direction * 2. * np.pi * 1j * x_u))
    Eout = Eout * dx * dx
    #print(Eout.shape)
    return Eout

#%%
# def ft_BASIC_phase(Ein, W1, W2, n2, direction, lamb, phase_interp):
#     Ein = Ein * np.exp(2j*np.pi * phase_interp / lamb)
#     #np.multiply(Ein, np.exp(2j*pi * phase_interp / lamb)) 7sec oh
#     nx, ny = np.shape(Ein)
#     dx = W1 / nx
#     du = W2 / n2
#     x = (np.linspace(-nx / 2, nx / 2, nx, endpoint=False) + 0.5) * dx
#     u = (np.linspace(-n2 / 2, n2 / 2, n2, endpoint=False) + 0.5) * du
#     x_u = np.outer(x, u)
#     #print((np.exp(-direction * 2. * pi * 1j * x_u * phase_interp / lamb)).shape)
#     m = np.exp(-direction * 2. * np.pi * 1j * x_u )
#     Eout = np.dot(np.transpose(m), Ein)
#     #Eout = np.dot(np.exp(phase_interp / lamb ), Ein)
#     #print(Eout.shape) 
#     Eout = np.dot(Eout, m)
#     #Eout = np.dot(np.exp( phase_interp / lamb ), Eout)
#     Eout = Eout * dx * dx
#     return Eout
#%%
def PSF(Ein, fov, N): #optique = pupil, # N = nbr_pix
    return (ft_BASIC(Ein, 1132 / 1024, fov, N, direction=1).real)**2

def PSFs(p,m,fov,Ny,N,ecran):#permet d'avoir la psf moyenne de m écran de phase combiné à la pupille p
    size = int(fov*2*Ny) # taille de la matrice de signal
    A = np.exp(2j*np.pi*ecran[:m]/1.65e-6)*p
    print(np.shape(A))
    im = np.zeros((m,size,size)) # allocation initial de la mémoire pour le calcul
    for i, A_i in enumerate(A):
        im[i,:,:] = PSF(A_i,fov,N)
    return np.mean(im, axis=0)

def AZAV(I,OWA,PhotAp): #photap c'est la largeur des anneaux pris pour faire l'azav
    L = len(I)
    u,U,V,R,T = VECT(L,2*OWA)
    f = u[int(L/2):L]
    Lf = len(f)
    I_m = np.zeros(Lf)
    I_s = np.zeros(Lf)
    for k in range(Lf):
        REG = (R<(f[k]+PhotAp/2))*(R>(f[k]-PhotAp/2))*(I != 0)
        I_m[k] = np.nanmean(I[REG==True])
        I_s[k] = np.nanstd(I[REG==True])
    return (I_m,I_s,f)

#%%
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def VECT(N, D=38):
    x = np.linspace(-D/2.0, D/2.0, N, endpoint=False) + D/(2.0*N)
    X , Y = np.meshgrid(x, x, sparse = False)
    R , T = cart2pol(X,Y)
    return (x, X, Y, R, T)
#OTF_tot = OTF(PSF())
#%%

# * TF[PSF]
def OTF(psf,fov,N): #soit le produit de OTF_telescope et OTF_atm, ou autocorrelation onde incidente capturée par le telescope
    return ft_BASIC(psf, fov, 2*1132/1024, N, direction=1)
    

# def DSP(psf_LP,fov,N,otf): # ? Est ce que le plot de la DPS est correct ?
#     dsp = ft_BASIC(np.log(ft_BASIC(psf_LP,fov,2*1132 / 1024, N,direction=-1)/otf),2*1132 / 1024, fov, N,direction=1)
#     dsp -= np.max(dsp)
#     return dsp

### ! Test d'une réecriture différent de la DSP

@timeit
def DSP(psf_LP,psf_pup,fov,N): # ? Est ce que le plot de la DPS est correct ?
    r = ft_BASIC(psf_LP,fov, 2*1132/1024, N, direction=1)/ft_BASIC(psf_pup,fov, 2*1132/1024, N, direction=1) # * complexe
    Re = np.log(np.abs(r))
    Im = np.angle(r)
    tf_inv = Re + 1j*Im
    return ft_BASIC(tf_inv, 1132/1024, 2*fov, N, direction=1)

#%%
#L'OTF du télescope est la TF de la PSF du télescope lui-même

#L'OTF atmosphérique s'écrit comme l'exponentielle de la fonction de structure de phase multipliée par un facteur -1/2, i..e., exp(- 1/2*Phase_structure)). 
# np.exp(-0.5*fsp)

#La fonction de structure de phase dépend linéairement de l'autocorrelation de la phase B_phi[rho], corrigée d'un offset qui est la valeur de l'autocorrelation de la phase à l'origine: 2*(B_phi[0]-B_phi[rho])

#L'autocorrelation de la phase est - à un scalaire multiplicatif près - la TF de la DSP.


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]#on divise par deux comme par symétrie ça va juste refaire les meme calcul mais "à l'envers"
# potentiellement va falloir utiliser scipy.signal.correlate si c'est trop lent


def FSP(sig): #fonction de structure de phase # ! pour l'instant pas utile, comme on utilise le fait que c'est la TF(DPS)
    return 2*(autocorr(sig)-np.max(autocorr(sig)))
#2*(B_phi[0]-B_phi[rho]) # * autocorrelation spatial de la phase - autocorrelation spatial phase a l'origine


