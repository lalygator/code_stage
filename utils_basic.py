import numpy as np
import os
import time
from scipy.ndimage import rotate, shift, gaussian_filter, convolve
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import curve_fit
from scipy.special import j0, j1
from scipy.signal import convolve2d
from numpy import pi
from math import gamma
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/carlotal/Downloads/proper_v3.1.1_python_3.x_15july19/')
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import warnings
warnings.simplefilter('ignore', category=AstropyWarning)
import imageio
from poppy.zernike import arbitrary_basis

import proper
import scipy.io
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
from datetime import datetime

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

WORKPATH = '/Users/carlotal/HARMONI/HC-PSF/common_files/'

def NA2FD(NA):
    #Converts the numerical aperture of a fiber into a F/D value
    FD_ratio = 1/(2*np.arctan(np.arcsin(NA)))
    return FD_ratio

def FD2NA(FD):
    #Converts the F/D ratio into the numerical aperture of a fiber
    NA = np.sin(np.tan(1/(2*FD)))
    return NA

def TIME4ADI_old(dec,ha,wavelength,separation,tel_diam=38.542,lat=-24.5892):
    #dec: declination [degree]
    #ha: hour angle [hour]
    #wavelength [m]
    #tel_diam: telescope diameter [m] ; default_value = 38.542m
    #separation [mas]
    #returns the time [min] it takes to perform ADI so that the planet has moved by 1 lambda/D
    #That is assuming a linear evolution of the parang, which is false!!!
    vitesse = np.abs(PARANG(dec,ha-1/120,lat)-PARANG(dec,ha+1/120,lat)) #deg/min
    angle = np.tan(LD2MAS(1,tel_diam,wavelength)/separation)*180/np.pi
    time = angle/(vitesse+1e-10)

def TIME4ADI(dec,ha_start,wavelength,separation,tel_diam=38.542,lat=-24.5892, separation_lbd=1):
    #dec: declination [degree]
    #ha_start: starting hour angle [hour], time will be computed assuming we move to further time
    #wavelength [m]
    #separation [mas]
    #tel_diam: telescope diameter [m] ; default_value = 38.542m
    #lat: latitude of the observation site [deg] ; default value is the one of ELT
    #separation_lbd: separation between the PSF in the first and last frames [lambda/D] ; default value is 1
    #returns the time [min] it will take to perform ADI so that the planet will have moved by 1 lambda/D
    #Note that the time that is returned is by definition as large or larger than the exact time.
    #For more precision you can compute the parallactic angle change that corresponds to 1s instead of 1min, but more iterations will be needed to converge
    #
    #Important notice: this is the MINIMUM amount of time so that the FIRST image can be subtracted from the LAST image without subtracting the planet.
    #If you cant to have more pairs of images that can be subtracted from each other, you need to expose over a longer time!
    max_angle = np.tan(LD2MAS(separation_lbd,tel_diam,wavelength)/separation)*180/np.pi
    
    angle = 0
    time = 0
    while angle < max_angle:
        angle+= np.abs(PARANG(dec,ha_start+time,lat)-PARANG(dec,ha_start+1/60+time,lat)) #angle for 1min change
        time+=1/60
    
    return time*60

def TIME4ADI_SEQ(dec,ha,wavelength,separation,N,tel_diam=38.542,lat=-24.5892,separation_lbd=1):
    #dec: declination [degree]
    #ha: hour angle [hour]
    #wavelength [m]
    #separation [mas]
    #N: number of epochs in sequence
    #tel_diam: telescope diameter [m] ; default_value = 38.542m
    #lat: latitude of the observation site [deg] ; default value is the one of ELT
    #
    #returns the time [min] it takes to perform ADI so that the planet has moved by 1 lambda/D
    time_sequence=np.array([ha*60])
    for k in range(N):
        time = TIME4ADI(dec,time_sequence[k]/60,wavelength,separation,tel_diam,lat,separation_lbd)
        time_sequence=np.append(time_sequence, time_sequence[k]+time)
    return time_sequence

def ZELDA(P,PHI,lbd_vect,k_lbd,delta,D_win,D_tel,zelda_diam_mas,zelda_height,ref_index):
    #P:                 pupil
    #PHI:               phase screen (m)
    #lbd:               wavelength vector (m)
    #k_lbd:             index within wavelength vector
    #delta:             dispersion from lbd_0 (mas)
    #D_win:             physical diameter of input array (m)
    #D_tel:             physical diameter of telescope (m)
    #zelda_diam_mas:    diameter of zelda mask (mas)
    #zelda height:      height of zelda mask (m)
    #ref_index:         refraction index at lbd
    
    N = len(P)
    x = VECT1(N,D_win)
    X,Y = np.meshgrid(x,x)
    
    lbd = lbd_vect[k_lbd]
    lbd_0 = np.mean(lbd_vect)
    N_lbd = len(lbd_vect)
    dispersion = np.exp(2*1j*np.pi*MAS2LD(delta,D_tel,lbd)*Y/D_tel) #phase due to dispersion
    D_ratio = D_win/D_tel
    lbd_ratio = lbd/lbd_0
    FOV = 1.1*MAS2LD(zelda_diam_mas,D_tel,lbd)
    # computation of electric fields in image plane
    E_zelda_pre = ft(P*np.exp(2*1j*np.pi*PHI/lbd)*dispersion,1,D_ratio,D_ratio,FOV,FOV,lbd_ratio,N,N)
    E_zelda_full = ft(P*np.exp(2*1j*np.pi*PHI/lbd)*dispersion,1,D_ratio,D_ratio,N/2,N/2,lbd_ratio,N,N)
    E_zelda_noab = ft(P*dispersion,1,D_ratio,D_ratio,FOV,FOV,lbd_ratio,N,N) #no aberrations here; used to infer the diffraction due to the mask
    # zelda mask model
    u = VECT1(N,FOV)
    U,V = np.meshgrid(u,u)
    R = np.sqrt(U**2+V**2)
    zelda_shape = (R<MAS2LD(zelda_diam_mas,D_tel,lbd)/2)
    zelda_mask = np.exp(2*1j*np.pi*zelda_height*(ref_index-1)/lbd)*zelda_shape
    # plt.figure()
    # plt.imshow(zelda_shape)
    # plt.figure()
    # plt.imshow(np.abs(E_zelda_pre)**2)
    E_zelda_post = E_zelda_pre*zelda_mask
    E_zelda_diff = E_zelda_pre*zelda_shape
    # computation of electric field in second pupil plane
    # P_zelda = ft(E_zelda_post,1,FOV,FOV,D_ratio,D_ratio,lbd_ratio,N,N)
    # P_zelda_diff = ft(E_zelda_diff,1,FOV,FOV,D_ratio,D_ratio,lbd_ratio,N,N)
    # P_zelda_full = ft(E_zelda_full,1,N/2,N/2,D_ratio,D_ratio,lbd_ratio,N,N)
    P_zelda = ft(E_zelda_post,1,FOV,FOV,D_ratio,D_ratio,lbd_ratio,N,N,True)
    P_zelda_diff = ft(E_zelda_diff,1,FOV,FOV,D_ratio,D_ratio,lbd_ratio,N,N,True)
    P_zelda_full = ft(E_zelda_full,1,N/2,N/2,D_ratio,D_ratio,lbd_ratio,N,N,True)
    P_camera = P_zelda_full + P_zelda - P_zelda_diff 
    # intensity in pupil plane
    I = np.abs(P_camera)**2/N_lbd
    # b factor (diffraction of zelda mask)
    # b = -np.real(ft(E_zelda_noab*zelda_shape,1,FOV,FOV,D_ratio,D_ratio,lbd_ratio,N,N))/N_lbd #for later calibration 
    b = np.real(ft(E_zelda_noab*zelda_shape,1,FOV,FOV,D_ratio,D_ratio,lbd_ratio,N,N,True))/N_lbd #for later calibration    
    return (I,b)

def ZELDA_RECT(P,PHI,lbd_vect,k_lbd,delta,DX_win,DY_win,D_tel,zelda_diam_mas,zelda_height,ref_index,FieldStopDiameter):
    #P:                 pupil
    #PHI:               phase screen (m)
    #lbd:               wavelength vector (m)
    #k_lbd:             index within wavelength vector
    #delta:             dispersion from lbd_0 (mas)
    #D_win:             physical diameter of input array (m)
    #D_tel:             physical diameter of telescope (m)
    #zelda_diam_mas:    diameter of zelda mask (mas)
    #zelda height:      height of zelda mask (m)
    #ref_index:         refraction index at lbd
    #FieldStopDiameter: diameter of a field stop (mas)
    
    NY,NX = np.shape(P)
    x = VECT1(NX,DX_win)
    y = VECT1(NY,DY_win)
    X,Y = np.meshgrid(x,y)
    
    lbd = lbd_vect[k_lbd]
    lbd_0 = np.mean(lbd_vect)
    N_lbd = len(lbd_vect)
    dispersion = np.exp(2*1j*np.pi*MAS2LD(delta,D_tel,lbd)*Y/D_tel) #phase due to dispersion
    DX_ratio = DX_win/D_tel
    DY_ratio = DY_win/D_tel
    lbd_ratio = lbd/lbd_0
    FOV = 1.1*MAS2LD(zelda_diam_mas,D_tel,lbd)
    FieldStop_LD = MAS2LD(FieldStopDiameter,D_tel,lbd)
    # computation of electric fields in image plane
    E_zelda_pre = ft(P*np.exp(2*1j*np.pi*PHI/lbd)*dispersion,1,DX_ratio,DY_ratio,FOV,FOV,lbd_ratio,NY,NX)
    E_zelda_full = ft(P*np.exp(2*1j*np.pi*PHI/lbd)*dispersion,1,DX_ratio,DY_ratio,NX/2,NY/2,lbd_ratio,NY,NX)
    E_zelda_noab = ft(P*dispersion,1,DX_ratio,DY_ratio,FOV,FOV,lbd_ratio,NY,NX) #no aberrations here; used to infer the diffraction due to the mask
    # zelda mask model
    u = VECT1(NX,FOV)
    v = VECT1(NY,FOV)
    U,V = np.meshgrid(u,v)
    R = np.sqrt(U**2+V**2)
    zelda_shape = (R<MAS2LD(zelda_diam_mas,D_tel,lbd)/2)
    zelda_mask = np.exp(2*1j*np.pi*zelda_height*(ref_index-1)/lbd)*zelda_shape
    # plt.figure()
    # plt.imshow(zelda_shape)
    # plt.figure()
    # plt.imshow(np.abs(E_zelda_pre)**2)
    E_zelda_post = E_zelda_pre*zelda_mask
    E_zelda_diff = E_zelda_pre*zelda_shape
    # computation of electric field in second pupil plane
    # P_zelda = ft(E_zelda_post,1,FOV,FOV,D_ratio,D_ratio,lbd_ratio,N,N)
    # P_zelda_diff = ft(E_zelda_diff,1,FOV,FOV,D_ratio,D_ratio,lbd_ratio,N,N)
    # P_zelda_full = ft(E_zelda_full,1,N/2,N/2,D_ratio,D_ratio,lbd_ratio,N,N)
    P_zelda = ft(E_zelda_post,1,FOV,FOV,DX_ratio,DY_ratio,lbd_ratio,NY,NX,True)
    P_zelda_diff = ft(E_zelda_diff,1,FOV,FOV,DX_ratio,DY_ratio,lbd_ratio,NY,NX,True)
    P_zelda_full = ft(E_zelda_full*(R<FieldStop_LD/2),1,NX/2,NY/2,DX_ratio,DY_ratio,lbd_ratio,NY,NX,True)
    P_camera = P_zelda_full + P_zelda - P_zelda_diff 
    # intensity in pupil plane
    I = np.abs(P_camera)**2/N_lbd
    # b factor (diffraction of zelda mask)
    # b = -np.real(ft(E_zelda_noab*zelda_shape,1,FOV,FOV,D_ratio,D_ratio,lbd_ratio,N,N))/N_lbd #for later calibration 
    b = np.real(ft(E_zelda_noab*zelda_shape,1,FOV,FOV,DX_ratio,DY_ratio,lbd_ratio,NY,NX,True))/N_lbd #for later calibration    
    return (I,b)

def offset(P, D_win, lbd_0 = 1.175e-6, lbd_min = 1.15e-6, lbd_max = 1.2e-6, N_lbd = 11, theta = 53, Plot = True):
    #P: pupil array
    #D_win: size of pupil array in units of pupil diameter (it should be >= 1)
    #lbd_0: central wavelength [m]
    #lbd_min & lbd_max: wavelength extrema [m]
    #N_lbd: number of wavelengths
    #theta: zenith angle of the observation [deg]
    #Plot: plot option (True = plot)
    lbd = np.linspace(lbd_min,lbd_max,N_lbd)
    zelda_diam = 1.06 #diameter of the zelda disk
    zelda_diam_mas = LD2MAS(zelda_diam,38.542,lbd_0) #diameter of zelda mask in mas
    zelda_diam_m = zelda_diam*40*lbd #diameter of zelda mask in m 
    zelda_height = 655.25e-9
    ref_index = np.linspace(1.4481,1.4486,N_lbd) #refraction index
    D_tel = 38.542
    D = 38.542*1132/1024
    
    I_SASD = np.zeros((len(P),len(P)))  # Sans Aberration Sans Dispersion
    I_SAAD = np.zeros((len(P),len(P)))  # Sans Aberration Avec Dispersion
    # I_offset_stock = []
    
    for i in range(N_lbd):
        I_sa_sd, b_sa_sd = ZELDA(np.rot90(P),0,lbd,i,0,D,D_tel,zelda_diam_mas,zelda_height,ref_index[i])            #_sa_sd = sans aberration, sans dispersion
        I_SASD += I_sa_sd
        delta = DISPER_ADC(lbd_0*1e6,lbd[i]*1e6,theta,5,50)*1000
        I_sa_ad, b_sa_ad = ZELDA(np.rot90(P),0,lbd,i,delta,D,D_tel,zelda_diam_mas,zelda_height,ref_index[i])        #_sa_ad = sans aberration, avec dispersion
        I_SAAD += I_sa_ad
        # I_offset_stock.append(I_sa_ad - I_sa_sd) 
    
    I_offset = np.rot90(I_SAAD - I_SASD)
    
    if Plot == True :
        plt.figure(figsize=(5,5))
        plt.imshow(I_offset, cmap = 'viridis')
        # plt.axis('off')
        plt.colorbar()
        # plt.tight_layout(pad=0, h_pad=0, w_pad=0, rect=(0,0,0,0))
        plt.show()
        
    return(I_offset)

def ZELDA_RECONSTRUCTOR(I,P,b_0,lbd_0):
    op = 3-2*b_0/P-(P-I/P)/b_0
    op[op<0] = 0
    PHI_zelda = P*(-1 + np.sqrt(op))
    PHI_zelda *= lbd_0/(2*np.pi)
    PHI_zelda[np.isnan(PHI_zelda)==1] = 0
    PHI_Z = np.rot90(PHI_zelda,2)
    return PHI_Z

def BOUNDARIES1D(A):
    A = (A.round()).astype(int)
    # "Enclose" mask with sentients to catch shifts later on
    mask = np.r_[False,np.equal(A, 1),False]

    # Get the shifting indices
    idx = np.flatnonzero(mask[1:] != mask[:-1])

    # Get the start and end indices with slicing along the shifting ones
    return np.array(list(zip(idx[::2], idx[1::2]-1)))

def MAKE_CIF(A,Dot_size_micron,output_filename):
    
    # The second number following 'DS' should be equal to 100 times the size of
    # a pixel (in microns). Example: if the pixel size is 5 microns, then 'DS 1 500 1'
    
    A= 1.0*(A**2 >= 0.5)
    
    f = open(output_filename,"w+")
    
    #fprintf(fid, '%s\n','(RESOLUTION 1);');
    #fprintf(fid, '%s\n',['DS 1 ' num2str(round(100*Dot_size_micron)) ' 1;']);
    #fprintf(fid, '%s\n','L 1;');
    
    f.write("%s\n" % '(RESOLUTION 1);')
    f.write("%s\n" % ('DS 1 ' + str(format(int(np.round(100*Dot_size_micron)))) + ' 1;'))
    f.write('%s\n' % 'L 1;')

    k0=int(np.round(len(A)/2))
    l0=int(np.round(len(A)/2))
    
    for k in range(len(A)):
        B=BOUNDARIES1D(A[k,:])
        for l in range(len(B)):
            #C=B[l]
            x0=B[l,0]
            x1=B[l,1]#C[np.round(len(C)/2),2]
            f.write('%s\n' % ('P ' + str(k-k0) + ',' + str(x0-1-l0) + ' ' + str(k-k0) + ',' + str(x1-l0) + ' ' + str(k+1-k0) + ',' + str(x1-l0) + ' ' + str(k+1-k0) + ',' + str(x0-1-l0) + ';'))

    f.write('%s\n' % 'DF;')
    f.write('%s\n' % 'E')

    f.close()

def ANAMORPHOSIS(P,D,gamma):
    N = len(P)
    x = VECT1(N,D)
    x2 = VECT1(N,D*gamma[0])
    y2 = VECT1(N,D*gamma[1])
    f_R = interp2d(x,x,P.real,kind='cubic',fill_value=0)
    f_I = interp2d(x,x,P.imag,kind='cubic',fill_value=0)
    P2 = f_R(x2,y2) + 1j*f_I(x2,y2)
    return P2

def make_mp4(ims, name="", fps=20):
    print("Making mp4...")
    with imageio.get_writer("{}.mp4".format(name), mode='I', fps=fps) as writer:
        for im in ims:
            #writer.append_data(bgr2rgb(im))
            writer.append_data(im)
    print("Done")

def HALO(dsp_file_name,FOV,Min_wavelength_for_Nyquist,Im,lbd=1.6e-6,Tel_diam=38.542,angle=0.):
    '''
    dsp_file_name: filename of DSP ; a fits is expected
    FOV: field of view [arcsec]
    Min_wavelength_for_Nyquist
    lbd: wavelength [m], optional ; default is 1.6e-6
    Im: image array (intensity)
    Tel_diam: telescope diameter [m], optional ; default is ELT diameter (38.542)
    angle: rotation angle of DSP [degrees], optional
    '''
    #dsp = fits.getdata(dsp_file_name)
    dsp = dsp_file_name
    size_dsp = len(dsp)
    size_min = np.round(np.ceil(MAS2LD(FOV,Tel_diam,Min_wavelength_for_Nyquist)/2))*2*4

    if size_min > size_dsp :
        print('size of original dsp is {0}, but should be at least {1} to fit FoV at minimum wavelength'.format(size_dsp,size_min))
        print('Interpolating DSP to get correct size...')
        x_DSP,X_DSP,Y_DSP,R_DSP,T_DSP      = VECT(size_dsp,2)
        x2_DSP,X2_DSP,Y2_DSP,R2_DSP,T2_DSP = VECT(size_min,2)
        f_dsp = interp2d(x_DSP,x_DSP,dsp,kind='linear',fill_value=0)
        dsp = f_dsp(x2_DSP,x2_DSP)
        dsp[np.isnan(dsp)==True] = 0
        size_dsp = len(dsp)

    DiamFTO = 2 #;%400/382;%BigD/D;
    dim = size_dsp/2
    Diameter = 1
    fmax     = dim/Diameter*0.5
    f1D,Xf,Yf,Rf,Tf = VECT(size_dsp,2*fmax)
    df = np.abs(f1D[2]-f1D[1])
    SUMDSP = np.sum(dsp)*df**2
    FT_dsp = np.abs(ft(dsp,1,dim,dim,DiamFTO,DiamFTO,1,len(dsp),len(dsp)))
    MAXFTDSP = np.max(FT_dsp)
    RENORM_FT_DSP = FT_dsp/MAXFTDSP*SUMDSP
    
    FT2DSP_norm=(2*pi/lbd)**2
    FT_PST=RENORM_FT_DSP*FT2DSP_norm
    FT_PST=rotate(FT_PST,angle,reshape=False)
    PH_STR=2.*(FT_PST[int(size_dsp/2),int(size_dsp/2)]-FT_PST)
    FTO_atm=np.exp(-PH_STR/2.)
    SR=np.exp(-np.max(FT_PST))
    print('Strehl: {0}'.format(SR))
    
    OWA = MAS2LD(FOV,Tel_diam,lbd)/2
    FTO_tel_0=ft(Im,1,2*OWA,2*OWA,DiamFTO,DiamFTO,1,len(dsp),len(dsp))
    FTO_tot_0=FTO_tel_0*FTO_atm
    Im_post=np.abs(ft(FTO_tot_0,1,DiamFTO,DiamFTO,2*OWA,2*OWA,1,len(Im),len(Im)))
    
    return Im_post

def PROJECT_FIBER(Im,FoV,Sigma):
    N=len(Im)
    Im_filt = np.zeros((N,N),dtype=complex)
    u,U,V,R,T=VECT(N,FoV)

    #U1,V1,U2,V2 = np.meshgrid(u,u,u,u)
    
    #R = np.sqrt((U1-U2)**2+(V1-V2)**2)
    
    for k in range(N):
        for l in range(N):
            R_loc = np.sqrt((U-u[k])**2+(V-u[l])**2)
            REG = (R_loc<1)
            Im_filt[l,k]=np.sum(Im[REG]*np.exp(-R_loc[REG]**2/(2*Sigma**2)))
    return Im_filt

def PROP_HARMONI_E2E(endgame,mode,lbd_wfs, lbd, ELT_pupil, gridsize, beam_width_ratio, WFE_cube, APOD_amplitude,offset,ZD=50):
    #endgame = 'zelda' or 'science', i.e., where the light goes in the end
    #mode = 'M6', 'FPRS', 'SCAO_DIC', 'H1', 'Z', 'H2', 'ALL', 'NOTHING', 'COMMON'=M6+FPRS+SCAO_DIC+H1, 'NON-COMMON' = H2+Z+CryoWin
    #lbd_wfs in meters
    #lbd for science, in meters
    #pupil
    #pupil size
    #ratio between the beam size that is propagated and the pupil size
    #wavefront errors
    #apodizer profile (careful! it messes up the WFE estimation; Use it only to get contrast estimates)
    #offset
    #Zenith distance
    
    lens_diam = 2.58 # old 2.134
    fl_exit_pupil = 45.765 # old 37.868
    
    d_M6_focus_tel = 3.5
    d_exit_pupil_M6 = fl_exit_pupil - d_M6_focus_tel
    d_focus_tel_win1 = 1.85
    d_win1_win2 = 0.05
    d_win2_RM1 = 2.1
    
    #fl_exit_pupil = 0
    fl_RM1 = 2.0
    fl_RM2 = -1.0
    fl_RM3 = 2.0
    fl_D1 = 527e-3 # old 580e-3
    fl_D2 = 600e-3
    fl_dummyLens = 250e-3
    
        #Distances
    d_RM1_RM2 = 2.0 #
    d_RM2_RM3 = 2.0 #
    d_RM3_Fold1 = 1.74 #
    d_fold1_diSCAO = 1.61 #
    d_diSCAO_FoldHC1 = 50e-3
    d_FoldHC1_FocusInHCM = 0.60
    
    d_FocusInHCM_FoldHC2 = 0.490
    d_FoldHC2_LHC1 = 37e-3
    d_LHC1_FoldHC3 = 75e-3
    d_FoldHC3_ADC = 175e-3
    d_ADC_diZELDA = 170e-3
    
    ####calculs conjg foyer #####
    d_focus_tel_RM1 = d_focus_tel_win1+d_win1_win2+d_win2_RM1
    f_focus_RM1 = (1/fl_RM1+1/-d_focus_tel_RM1)**-1
    f_focus_RM2 = (1/fl_RM2+1/(f_focus_RM1-d_RM1_RM2))**-1
    f_focus_RM3 = (1/fl_RM3+1/(f_focus_RM2-d_RM1_RM2))**-1
    
    ## calculs conjug pupil """"
    Pup_RM1 = fl_exit_pupil  + d_focus_tel_win1 + d_win1_win2 + d_win2_RM1
    Pup_after_RM1 = (1/fl_RM1+1/-Pup_RM1)**-1
    Pup_after_RM2 = (1/fl_RM2+1/(Pup_after_RM1-d_RM1_RM2))**-1
    Pup_after_RM3 = (1/fl_RM3+1/(Pup_after_RM2-d_RM2_RM3))**-1
    d_Pup_lHC1 = Pup_after_RM3 - d_RM3_Fold1 - d_fold1_diSCAO - d_diSCAO_FoldHC1 - d_FoldHC1_FocusInHCM - d_FocusInHCM_FoldHC2 - d_FoldHC2_LHC1
    d_Pup_lHC1 = Pup_after_RM3-f_focus_RM3 - fl_D1
    Pup_after_LHC1 = (1/fl_D1+1/d_Pup_lHC1)**-1
    
    d_diZELDA_APOD = Pup_after_LHC1 - d_LHC1_FoldHC3 - d_FoldHC3_ADC - d_ADC_diZELDA
    
    # calcul conjug pupil ds voie ZELDA
    fl_L1Z = 1188e-3
    fl_L2Z = 120e-3
    d_diZELDA_L1Z = 250e-3
    d_L1Z_F1Z = 130e-3
    d_F1Z_F2Z = 700e-3
    d_F2Z_Zmask = 358e-3
    
    # ## calculs conjug pupil in ZELDA channel """"
    Pup_afterL1Z = (1/fl_L1Z+1/-(d_diZELDA_L1Z -d_diZELDA_APOD))**-1
    Pup_afterL2Z = (1/fl_L2Z+1/(Pup_afterL1Z - fl_L1Z - fl_L2Z))**-1
    
    d_Zmask_L2Z = fl_L2Z
    d_L2Z_Zcam = round(Pup_afterL2Z, 4)
    
    #calculs pour confondre la pupille sur voie directe et HCM
    # direct channel
    #d_RM3_FocusR = 4
    #d_Pup_FocusR = Pup_after_RM3 - d_RM3_FocusR
    
    #HCM channel
    d_APOD_LHC2 = 0.5923347
    #d_LHC1_LHC2 = d_LHC1_FoldHC3 + d_FoldHC3_ADC + d_ADC_diZELDA + d_diZELDA_APOD + d_APOD_LHC2
    #d_LHC1_LHC2 = 1.1257
    
    Pup_after_LHC2 = (1/fl_D2+1/-(d_APOD_LHC2))**-1
    d_LHC2_FoldHC4 = 50e-3
    d_FoldHC4_cryowin = 0.4
    d_cryowin_FocusR = 0.15
    
    d_dummyLens = fl_dummyLens
    print(Pup_after_LHC2)
    #d_PupHCM_FocusR = Pup_after_LHC2 + d_LHC2_FoldHC4 + d_FoldHC4_cryowin + d_cryowin_FocusR
        
    ### Last values, before modification on May 31st, 2023
    # lens_diam = 2.58 # old 2.134
    # fl_exit_pupil = 45.765 # old 37.868 
    
    # d_M6_focus_tel = 3.5
    # d_exit_pupil_M6 = fl_exit_pupil - d_M6_focus_tel
    # d_focus_tel_win1 = 1.85
    # d_win1_win2 = 0.05
    # d_win2_RM1 = 2.1
    
    # #fl_exit_pupil = 0
    # fl_RM1 = 2.0
    # fl_RM2 = -1.0
    # fl_RM3 = 2.0    
    # fl_D1 = 527e-3 # old 580e-3
    # fl_D2 = 600e-3
    # fl_dummyLens = 250e-3
    
    #     #Distances
    # d_RM1_RM2 = 2.0 #
    # d_RM2_RM3 = 2.0 #
    # d_RM3_Fold1 = 1.74 #
    # d_fold1_diSCAO = 1.61 #
    # d_diSCAO_FoldHC1 = 50e-3
    # d_FoldHC1_FocusInHCM = 0.60
        
    # d_FocusInHCM_FoldHC2 = 0.490
    # d_FoldHC2_LHC1 = 37e-3
    # d_LHC1_FoldHC3 = 75e-3
    # d_FoldHC3_ADC = 175e-3
    # d_ADC_diZELDA = 170e-3
    
    # ####calculs conjg foyer #####
    # d_focus_tel_RM1 = d_focus_tel_win1+d_win1_win2+d_win2_RM1
    # f_focus_RM1 = (1/fl_RM1+1/-d_focus_tel_RM1)**-1
    # f_focus_RM2 = (1/fl_RM2+1/(f_focus_RM1-d_RM1_RM2))**-1
    # f_focus_RM3 = (1/fl_RM3+1/(f_focus_RM2-d_RM1_RM2))**-1
    
    # ## calculs conjug pupil """"
    # Pup_RM1 = fl_exit_pupil  + d_focus_tel_win1 + d_win1_win2 + d_win2_RM1
    # Pup_after_RM1 = (1/fl_RM1+1/-Pup_RM1)**-1
    # Pup_after_RM2 = (1/fl_RM2+1/(Pup_after_RM1-d_RM1_RM2))**-1
    # Pup_after_RM3 = (1/fl_RM3+1/(Pup_after_RM2-d_RM2_RM3))**-1
    # d_IFSfocus_Pup = np.round(Pup_after_RM3 - f_focus_RM3,3)
    # d_Pup_lHC1 = d_IFSfocus_Pup - fl_D1
    # Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 5)
    
    # d_diZELDA_APOD = np.round(Pup_after_LHC1-d_LHC1_FoldHC3 - d_FoldHC3_ADC-d_ADC_diZELDA, 4)

    # d_APOD_LHC2 = 600e-3
    # #d_LHC1_LHC2 = d_LHC1_FoldHC3 + d_FoldHC3_ADC + d_ADC_diZELDA + d_diZELDA_APOD + d_APOD_LHC2
    # #d_LHC1_LHC2 = 1.1257
    
    # #Pup_after_LHC2 = (1/fl_D2+1/-(d_LHC1_LHC2-Pup_after_LHC1))**-1
    # d_LHC2_FoldHC4 = 50e-3
    # d_FoldHC4_cryowin = 0.4
    # d_cryowin_FocusR = 0.15
    
    # # calcul conjug pupil ds voie ZELDA
    # fl_L1Z = 1.2
    # fl_L2Z = 0.12
    # d_diZELDA_L1Z = 210e-3
    # d_L1Z_F1Z = 125e-3
    # d_F1Z_F2Z = 710e-3
    # d_F2Z_Zmask = 365e-3
    
    # # calculs conjug pupil in ZELDA channel
    # Pup_afterL1Z = (1/fl_L1Z+1/(d_LHC1_FoldHC3+d_FoldHC3_ADC+d_ADC_diZELDA-Pup_after_LHC1))**-1
    # Pup_afterL2Z = (1/fl_L2Z+1/(Pup_afterL1Z - fl_L1Z - fl_L2Z))**-1
    
    # d_Zmask_L2Z = fl_L2Z
    # d_L2Z_Zcam = round(Pup_afterL2Z, 4)
    
    # d_dummyLens = fl_dummyLens
    
    ### OLD version, modified on 23 November 2021 to meet LJoc version
    
    # # Pup_RM1 = fl_exit_pupil  + d_focus_tel_win1 + d_win1_win2 + d_win2_RM1
    # # Pup_after_RM1 = (1/fl_RM1+1/-Pup_RM1)**-1
    # # Pup_after_RM2 = (1/fl_RM2+1/(Pup_after_RM1-d_RM1_RM2))**-1
    # # Pup_after_RM3 = (1/fl_RM3+1/(Pup_after_RM2-d_RM2_RM3))**-1
    # # d_Pup_lHC1 = Pup_after_RM3 - d_RM3_Fold1 - d_fold1_diSCAO - d_diSCAO_FoldHC1 - d_FoldHC1_FocusInHCM - d_FocusInHCM_FoldHC2 - d_FoldHC2_LHC1
    # # Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 4) #Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 4)
    # # #Pup_after_LHC1 = 0.527
    
    # d_diZELDA_APOD = Pup_after_LHC1-d_LHC1_FoldHC3 - d_FoldHC3_ADC-d_ADC_diZELDA

    # # calcul conjug pupil ds voie ZELDA
    # fl_L1Z = 1310e-3
    # fl_L2Z = 120e-3
    # d_diZELDA_L1Z = 208e-3
    # d_L1Z_F1Z = 130e-3
    # d_F1Z_F2Z = 810e-3
    # d_F2Z_Zmask = 370e-3
    
    # ## calculs conjug pupil in ZELDA channel """"
    # Pup_afterL1Z = (1/fl_L1Z+1/-(d_diZELDA_L1Z -d_diZELDA_APOD))**-1
    # Pup_afterL2Z = (1/fl_L2Z+1/(Pup_afterL1Z - fl_L1Z - fl_L2Z))**-1
    
    # d_Zmask_L2Z = fl_L2Z
    # d_L2Z_Zcam = round(Pup_afterL2Z, 4)
    
    # #calculs pour confondre la pupille sur voie directe et HCM
    # # direct channel
    # #d_RM3_FocusR = 4
    # #d_Pup_FocusR = Pup_after_RM3 - d_RM3_FocusR
    
    # #HCM channel
    # d_APOD_LHC2 = 600e-3
    # #d_LHC1_LHC2 = d_LHC1_FoldHC3 + d_FoldHC3_ADC + d_ADC_diZELDA + d_diZELDA_APOD + d_APOD_LHC2
    # #d_LHC1_LHC2 = 1.1257
    
    # #Pup_after_LHC2 = (1/fl_D2+1/-(d_LHC1_LHC2-Pup_after_LHC1))**-1
    # d_LHC2_FoldHC4 = 50e-3
    # d_FoldHC4_cryowin = 0.4
    # d_cryowin_FocusR = 0.15
    
    # d_dummyLens = fl_dummyLens
    
    ###
    
    #d_PupHCM_FocusR = Pup_after_LHC2 + d_LHC2_FoldHC4 + d_FoldHC4_cryowin + d_cryowin_FocusR
    
    '''
    lens_diam = 2.134
    fl_exit_pupil = 37.868 #
    
    d_M6_focus_tel = 3.5
    d_exit_pupil_M6 = fl_exit_pupil - d_M6_focus_tel
    d_focus_tel_win1 = 1.85
    d_win1_win2 = 0.05
    d_win2_RM1 = 2.1
    
    #fl_exit_pupil = 0
    fl_RM1 = 2.0
    fl_RM2 = -1.0
    fl_RM3 = 2.0    
    fl_D1 = 580e-3
    fl_D2 = 600e-3
    fl_dummyLens = 250e-3
    #Distances
    
    d_RM1_RM2 = 2.0 #
    d_RM2_RM3 = 2.0 #
    d_RM3_Fold1 = 1.74 #
    d_fold1_diSCAO = 1.61 #
    d_diSCAO_FoldHC1 = 50e-3
    d_FoldHC1_FocusInHCM = 0.60
    d_FocusInHCM_FoldHC2 = 0.54
    d_FoldHC2_LHC1 = 40e-3
    d_LHC1_FoldHC3 = 90e-3
    
    ## calculs conjug pupil """"
    Pup_RM1 = fl_exit_pupil  + d_focus_tel_win1 + d_win1_win2 + d_win2_RM1
    Pup_after_RM1 = (1/fl_RM1+1/-Pup_RM1)**-1
    Pup_after_RM2 = (1/fl_RM2+1/(Pup_after_RM1-d_RM1_RM2))**-1
    Pup_after_RM3 = (1/fl_RM3+1/(Pup_after_RM2-d_RM2_RM3))**-1
    d_Pup_lHC1 = Pup_after_RM3 - d_RM3_Fold1 - d_fold1_diSCAO - d_diSCAO_FoldHC1 - d_FoldHC1_FocusInHCM - d_FocusInHCM_FoldHC2 - d_FoldHC2_LHC1
    Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 4)
    
    d_FoldHC3_diZELDA = 310e-3
    d_diZELDA_APOD = Pup_after_LHC1-d_LHC1_FoldHC3 - d_FoldHC3_diZELDA
    
    # calcul conjug pupil ds voie ZELDA
    fl_L1Z = 1310e-3
    fl_L2Z = 120e-3
    d_diZELDA_L1Z = 208e-3
    d_L1Z_F1Z = 130e-3
    d_F1Z_F2Z = 810e-3
    d_F2Z_Zmask = 370e-3
    
    ## calculs conjug pupil in ZELDA channel """"
    Pup_afterL1Z = (1/fl_L1Z+1/-(d_diZELDA_L1Z -d_diZELDA_APOD))**-1
    Pup_afterL2Z = (1/fl_L2Z+1/(Pup_afterL1Z - fl_L1Z - fl_L2Z))**-1
    
    d_Zmask_L2Z = fl_L2Z
    d_L2Z_Zcam = round(Pup_afterL2Z, 4)
    
    #calculs pour confondre la pupille sur voie directe et HCM
    # direct channel
    #d_RM3_FocusR = 4
    #d_Pup_FocusR = Pup_after_RM3 - d_RM3_FocusR
    
    #HCM channel
    d_APOD_LHC2 = 600e-3#590.7e-3
    #d_LHC1_LHC2 = d_LHC1_FoldHC3 + d_FoldHC3_diZELDA + d_diZELDA_APOD + d_APOD_LHC2
    
    #Pup_after_LHC2 = (1/fl_D2+1/-(d_LHC1_LHC2-Pup_after_LHC1))**-1
    d_LHC2_FoldHC4 = 50e-3
    d_FoldHC4_cryowin = 400e-3
    d_cryowin_FocusR = 0.15
    
    d_dummyLens = fl_dummyLens
    
    #d_PupHCM_FocusR = Pup_after_LHC2 + d_LHC2_FoldHC4 + d_FoldHC4_cryowin + d_cryowin_FocusR
    '''
    
    MODE_vect = np.zeros(len(WFE_cube[:,0,0]))

    if endgame == 'science':
    
        DIST_vect = [d_exit_pupil_M6,
                     d_M6_focus_tel,
                     d_focus_tel_win1,
                     d_win1_win2,
                     d_win2_RM1,
                     d_RM1_RM2,
                     d_RM2_RM3,
                     d_RM3_Fold1,
                     d_fold1_diSCAO,
                     d_diSCAO_FoldHC1,
                     d_FoldHC1_FocusInHCM,
                     d_FocusInHCM_FoldHC2,
                     d_FoldHC2_LHC1,
                     d_LHC1_FoldHC3,
                     d_FoldHC3_ADC,
                     d_ADC_diZELDA,    
                     d_diZELDA_APOD,
                     d_APOD_LHC2,
                     d_LHC2_FoldHC4,
                     d_FoldHC4_cryowin,
                     d_cryowin_FocusR]
    
        FOC_vect = [0,
                    0,
                    0,
                    0,
                    fl_RM1,
                    fl_RM2,
                    fl_RM3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    fl_D1,
                    0,
                    0,
                    0,
                    0,
                    fl_D2,
                    0,
                    0,
                    0]
        
        if mode == 'M6':
            MODE_vect[0] = 1
        elif mode == 'FPRS':
            MODE_vect[2:8] = 1
        elif mode == 'SCAO_DIC':
            MODE_vect[8] = 1
        elif mode == 'H1':
            MODE_vect[9:15] = 1
        elif mode == 'H2':
            MODE_vect[15:19] = 1
        elif mode == 'CRYO_WIN':
            MODE_vect[19] = 1
        elif mode == 'ALL':
            MODE_vect[:] = 1
        elif mode == 'COMMON':
            MODE_vect[0:15] = 1
        elif mode == 'NON_COMMON':
            MODE_vect[15:-1] = 1
        
    elif endgame == 'scao':
    
        DIST_vect = [d_exit_pupil_M6,
                     d_M6_focus_tel,
                     d_focus_tel_win1,
                     d_win1_win2,
                     d_win2_RM1,
                     d_RM1_RM2,
                     d_RM2_RM3,
                     d_RM3_Fold1,
                     d_fold1_diSCAO,
                     d_diSCAO_FoldHC1,
                     d_FoldHC1_FocusInHCM]
    
        FOC_vect = [0,
                    0,
                    0,
                    0,
                    fl_RM1,
                    fl_RM2,
                    fl_RM3,
                    0,
                    0,
                    0,
                    0]
        
        if mode == 'M6':
            MODE_vect[0] = 1
        elif mode == 'FPRS':
            MODE_vect[2:8] = 1
        elif mode == 'SCAO_DIC':
            MODE_vect[8] = 1
        elif mode == 'H1':
            MODE_vect[9:-1] = 1
        elif mode == 'ALL':
            MODE_vect[:] = 1
        elif mode == 'COMMON':
            MODE_vect[0:-1] = 1
        
    elif endgame == 'zelda':
        
        DIST_vect = [d_exit_pupil_M6,
                     d_M6_focus_tel,
                     d_focus_tel_win1,
                     d_win1_win2,
                     d_win2_RM1,
                     d_RM1_RM2,
                     d_RM2_RM3,
                     d_RM3_Fold1,
                     d_fold1_diSCAO,
                     d_diSCAO_FoldHC1,
                     d_FoldHC1_FocusInHCM,
                     d_FocusInHCM_FoldHC2,
                     d_FoldHC2_LHC1,
                     d_LHC1_FoldHC3,
                     d_FoldHC3_ADC,
                     d_ADC_diZELDA,
                     d_diZELDA_L1Z,
                     d_L1Z_F1Z,
                     d_F1Z_F2Z,
                     d_F2Z_Zmask,
                     d_Zmask_L2Z,
                     d_L2Z_Zcam]
    
        FOC_vect = [0,
                    0,
                    0,
                    0,
                    fl_RM1,
                    fl_RM2,
                    fl_RM3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    fl_D1,
                    0,
                    0,
                    0,
                    fl_L1Z,
                    0,
                    0,
                    0,
                    fl_L2Z,
                    0]
        
        if mode == 'M6':
            MODE_vect[0] = 1
        elif mode == 'FPRS':
            MODE_vect[2:8] = 1
        elif mode == 'SCAO_DIC':
            MODE_vect[8] = 1
        elif mode == 'H1':
            MODE_vect[9:15] = 1
        elif mode == 'Z':
            MODE_vect[15:21] = 1
        elif mode == 'ALL':
            MODE_vect[:] = 1
        elif mode == 'COMMON':
            MODE_vect[0:15] = 1
        elif mode == 'NON_COMMON':
            MODE_vect[15:-1] = 1
    
    wfo = proper.prop_begin(lens_diam, lbd, gridsize, beam_width_ratio)
    proper.prop_multiply(wfo, ELT_pupil)
    RHO = MAS2LD(DISPER(lbd_wfs*1e6,lbd*1e6,ZD)*1000,38.542,lbd) ### ZD should be 45 or 50
    x,X,Y,R,T = VECT(gridsize,1/beam_width_ratio)
    TILT = np.exp(2*1j*pi*X*RHO)
    proper.prop_multiply(wfo, TILT)
    
    if offset != 0:
        proper.prop_multiply( wfo, np.exp(2*1j*pi*offset/lbd))
    
    proper.prop_define_entrance(wfo)
    proper.prop_lens(wfo,fl_exit_pupil)
    
    # fig,ax = plt.subplots(5,5)
    
    for k in range(len(WFE_cube[:,0,0])):
        print(k)
        proper.prop_propagate(wfo, DIST_vect[k])
        
        # E_temp = proper.prop_get_wavefront(wfo)
        # x0 = int(k//5)
        # y0 = int(np.mod(k,5))
        
        if np.abs(FOC_vect[k]) > 0:
            proper.prop_lens(wfo,FOC_vect[k])
        if np.sum(np.abs(WFE_cube[k,:,:])) > 0:
            proper.prop_add_phase( wfo, WFE_cube[k,:,:]*MODE_vect[k])
        if k==16 and endgame == 'science':
            proper.prop_multiply(wfo,APOD_amplitude)
        # ax[x0,y0].imshow(np.abs(E_temp)**2)
#        if endgame == 'science':
#            AZ = proper.prop_get_amplitude(wfo)
#            plt.figure()
#            plt.imshow(AZ)
            
#        if endgame == 'science':
#            #smp = proper.prop_get_beamradius(wfo)
#            #print('Beam radius: {:4.2f}mm'.format(np.round(2*smp*1e3)))
#            smp = proper.prop_get_sampling(wfo)
#            print('Beam radius: {:4.2f}mm'.format(np.round(smp*1e3*gridsize*beam_width_ratio)))
#            
    Pup_finale = (1/fl_dummyLens+1/-(fl_dummyLens-Pup_after_LHC2))**-1

    if endgame == 'science' or endgame == 'scao':
        proper.prop_propagate(wfo, d_dummyLens)
        proper.prop_lens(wfo,fl_dummyLens)
        # proper.prop_propagate(wfo, d_dummyLens)
        proper.prop_propagate(wfo, Pup_finale)
    
    if endgame == 'science' or endgame == 'zelda':
        TILT = np.exp(2*1j*pi*X*RHO)
        proper.prop_multiply(wfo, TILT)
    else:
        TILT = np.exp(-2*1j*pi*X*RHO)
        proper.prop_multiply(wfo, TILT)
    
    #PUP_fin = proper.prop_get_wavefront(wfo)
    
    PHI = proper.prop_get_phase(wfo)
    PHI = PHI/(2*np.pi)*lbd
    
    E = proper.prop_get_wavefront(wfo)
        
    return PHI, E

# def PROP_HARMONI_E2E_OLD(endgame,lbd_wfs, lbd, ELT_pupil, gridsize, beam_width_ratio, WFE_cube, APOD_amplitude,offset,ZD=50):
    
    lens_diam = 2.58 # old 2.134
    fl_exit_pupil = 45.765 # old 37.868
    
    d_M6_focus_tel = 3.5
    d_exit_pupil_M6 = fl_exit_pupil - d_M6_focus_tel
    d_focus_tel_win1 = 1.85
    d_win1_win2 = 0.05
    d_win2_RM1 = 2.1
    
    #fl_exit_pupil = 0
    fl_RM1 = 2.0
    fl_RM2 = -1.0
    fl_RM3 = 2.0
    fl_D1 = 527e-3 # old 580e-3
    fl_D2 = 600e-3
    fl_dummyLens = 250e-3
    
        #Distances
    d_RM1_RM2 = 2.0 #
    d_RM2_RM3 = 2.0 #
    d_RM3_Fold1 = 1.74 #
    d_fold1_diSCAO = 1.61 #
    d_diSCAO_FoldHC1 = 50e-3
    d_FoldHC1_FocusInHCM = 0.60
    
    d_FocusInHCM_FoldHC2 = 0.490
    d_FoldHC2_LHC1 = 37e-3
    d_LHC1_FoldHC3 = 75e-3
    d_FoldHC3_ADC = 175e-3
    d_ADC_diZELDA = 170e-3
    
    ####calculs conjg foyer #####
    d_focus_tel_RM1 = d_focus_tel_win1+d_win1_win2+d_win2_RM1
    f_focus_RM1 = (1/fl_RM1+1/-d_focus_tel_RM1)**-1
    f_focus_RM2 = (1/fl_RM2+1/(f_focus_RM1-d_RM1_RM2))**-1
    f_focus_RM3 = (1/fl_RM3+1/(f_focus_RM2-d_RM1_RM2))**-1
    
    ## calculs conjug pupil """"
    Pup_RM1 = fl_exit_pupil  + d_focus_tel_win1 + d_win1_win2 + d_win2_RM1
    Pup_after_RM1 = (1/fl_RM1+1/-Pup_RM1)**-1
    Pup_after_RM2 = (1/fl_RM2+1/(Pup_after_RM1-d_RM1_RM2))**-1
    Pup_after_RM3 = round((1/fl_RM3+1/(Pup_after_RM2-d_RM2_RM3))**-1,3)
    d_Pup_lHC1 = Pup_after_RM3 - d_RM3_Fold1 - d_fold1_diSCAO - d_diSCAO_FoldHC1 - d_FoldHC1_FocusInHCM - d_FocusInHCM_FoldHC2 - d_FoldHC2_LHC1
    d_Pup_lHC1 = Pup_after_RM3-f_focus_RM3 - fl_D1
    Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 4) 
    
    d_diZELDA_APOD = Pup_after_LHC1 - d_LHC1_FoldHC3 - d_FoldHC3_ADC - d_ADC_diZELDA
    
    # calcul conjug pupil ds voie ZELDA
    fl_L1Z = 1188e-3
    fl_L2Z = 120e-3
    d_diZELDA_L1Z = 250e-3
    d_L1Z_F1Z = 130e-3
    d_F1Z_F2Z = 700e-3
    d_F2Z_Zmask = 358e-3
    
    # ## calculs conjug pupil in ZELDA channel """"
    Pup_afterL1Z = (1/fl_L1Z+1/-(d_diZELDA_L1Z -d_diZELDA_APOD))**-1
    Pup_afterL2Z = (1/fl_L2Z+1/(Pup_afterL1Z - fl_L1Z - fl_L2Z))**-1
    
    d_Zmask_L2Z = fl_L2Z
    d_L2Z_Zcam = round(Pup_afterL2Z, 4)
    
    #calculs pour confondre la pupille sur voie directe et HCM
    # direct channel
    #d_RM3_FocusR = 4
    #d_Pup_FocusR = Pup_after_RM3 - d_RM3_FocusR
    
    #HCM channel
    d_APOD_LHC2 = 0.5923347
    #d_LHC1_LHC2 = d_LHC1_FoldHC3 + d_FoldHC3_ADC + d_ADC_diZELDA + d_diZELDA_APOD + d_APOD_LHC2
    #d_LHC1_LHC2 = 1.1257
    
    Pup_after_LHC2 = (1/fl_D2+1/-(d_APOD_LHC2))**-1
    d_LHC2_FoldHC4 = 50e-3
    d_FoldHC4_cryowin = 0.4
    d_cryowin_FocusR = 0.15
    
    d_dummyLens = fl_dummyLens
    print(Pup_after_LHC2)
    #d_PupHCM_FocusR = Pup_after_LHC2 + d_LHC2_FoldHC4 + d_FoldHC4_cryowin + d_cryowin_FocusR
        
    ### Last values, before modification on May 31st, 2023
    # lens_diam = 2.58 # old 2.134
    # fl_exit_pupil = 45.765 # old 37.868 
    
    # d_M6_focus_tel = 3.5
    # d_exit_pupil_M6 = fl_exit_pupil - d_M6_focus_tel
    # d_focus_tel_win1 = 1.85
    # d_win1_win2 = 0.05
    # d_win2_RM1 = 2.1
    
    # #fl_exit_pupil = 0
    # fl_RM1 = 2.0
    # fl_RM2 = -1.0
    # fl_RM3 = 2.0    
    # fl_D1 = 527e-3 # old 580e-3
    # fl_D2 = 600e-3
    # fl_dummyLens = 250e-3
    
    #     #Distances
    # d_RM1_RM2 = 2.0 #
    # d_RM2_RM3 = 2.0 #
    # d_RM3_Fold1 = 1.74 #
    # d_fold1_diSCAO = 1.61 #
    # d_diSCAO_FoldHC1 = 50e-3
    # d_FoldHC1_FocusInHCM = 0.60
        
    # d_FocusInHCM_FoldHC2 = 0.490
    # d_FoldHC2_LHC1 = 37e-3
    # d_LHC1_FoldHC3 = 75e-3
    # d_FoldHC3_ADC = 175e-3
    # d_ADC_diZELDA = 170e-3
    
    # ####calculs conjg foyer #####
    # d_focus_tel_RM1 = d_focus_tel_win1+d_win1_win2+d_win2_RM1
    # f_focus_RM1 = (1/fl_RM1+1/-d_focus_tel_RM1)**-1
    # f_focus_RM2 = (1/fl_RM2+1/(f_focus_RM1-d_RM1_RM2))**-1
    # f_focus_RM3 = (1/fl_RM3+1/(f_focus_RM2-d_RM1_RM2))**-1
    
    # ## calculs conjug pupil """"
    # Pup_RM1 = fl_exit_pupil  + d_focus_tel_win1 + d_win1_win2 + d_win2_RM1
    # Pup_after_RM1 = (1/fl_RM1+1/-Pup_RM1)**-1
    # Pup_after_RM2 = (1/fl_RM2+1/(Pup_after_RM1-d_RM1_RM2))**-1
    # Pup_after_RM3 = (1/fl_RM3+1/(Pup_after_RM2-d_RM2_RM3))**-1
    # d_IFSfocus_Pup = np.round(Pup_after_RM3 - f_focus_RM3,3)
    # d_Pup_lHC1 = d_IFSfocus_Pup - fl_D1
    # Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 5)
    
    # d_diZELDA_APOD = np.round(Pup_after_LHC1-d_LHC1_FoldHC3 - d_FoldHC3_ADC-d_ADC_diZELDA, 4)

    # d_APOD_LHC2 = 600e-3
    # #d_LHC1_LHC2 = d_LHC1_FoldHC3 + d_FoldHC3_ADC + d_ADC_diZELDA + d_diZELDA_APOD + d_APOD_LHC2
    # #d_LHC1_LHC2 = 1.1257
    
    # #Pup_after_LHC2 = (1/fl_D2+1/-(d_LHC1_LHC2-Pup_after_LHC1))**-1
    # d_LHC2_FoldHC4 = 50e-3
    # d_FoldHC4_cryowin = 0.4
    # d_cryowin_FocusR = 0.15
    
    # # calcul conjug pupil ds voie ZELDA
    # fl_L1Z = 1.2
    # fl_L2Z = 0.12
    # d_diZELDA_L1Z = 210e-3
    # d_L1Z_F1Z = 125e-3
    # d_F1Z_F2Z = 710e-3
    # d_F2Z_Zmask = 365e-3
    
    # # calculs conjug pupil in ZELDA channel
    # Pup_afterL1Z = (1/fl_L1Z+1/(d_LHC1_FoldHC3+d_FoldHC3_ADC+d_ADC_diZELDA-Pup_after_LHC1))**-1
    # Pup_afterL2Z = (1/fl_L2Z+1/(Pup_afterL1Z - fl_L1Z - fl_L2Z))**-1
    
    # d_Zmask_L2Z = fl_L2Z
    # d_L2Z_Zcam = round(Pup_afterL2Z, 4)
    
    # d_dummyLens = fl_dummyLens
    
    ### OLD version, modified on 23 November 2021 to meet LJoc version
    
    # # Pup_RM1 = fl_exit_pupil  + d_focus_tel_win1 + d_win1_win2 + d_win2_RM1
    # # Pup_after_RM1 = (1/fl_RM1+1/-Pup_RM1)**-1
    # # Pup_after_RM2 = (1/fl_RM2+1/(Pup_after_RM1-d_RM1_RM2))**-1
    # # Pup_after_RM3 = (1/fl_RM3+1/(Pup_after_RM2-d_RM2_RM3))**-1
    # # d_Pup_lHC1 = Pup_after_RM3 - d_RM3_Fold1 - d_fold1_diSCAO - d_diSCAO_FoldHC1 - d_FoldHC1_FocusInHCM - d_FocusInHCM_FoldHC2 - d_FoldHC2_LHC1
    # # Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 4) #Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 4)
    # # #Pup_after_LHC1 = 0.527
    
    # d_diZELDA_APOD = Pup_after_LHC1-d_LHC1_FoldHC3 - d_FoldHC3_ADC-d_ADC_diZELDA

    # # calcul conjug pupil ds voie ZELDA
    # fl_L1Z = 1310e-3
    # fl_L2Z = 120e-3
    # d_diZELDA_L1Z = 208e-3
    # d_L1Z_F1Z = 130e-3
    # d_F1Z_F2Z = 810e-3
    # d_F2Z_Zmask = 370e-3
    
    # ## calculs conjug pupil in ZELDA channel """"
    # Pup_afterL1Z = (1/fl_L1Z+1/-(d_diZELDA_L1Z -d_diZELDA_APOD))**-1
    # Pup_afterL2Z = (1/fl_L2Z+1/(Pup_afterL1Z - fl_L1Z - fl_L2Z))**-1
    
    # d_Zmask_L2Z = fl_L2Z
    # d_L2Z_Zcam = round(Pup_afterL2Z, 4)
    
    # #calculs pour confondre la pupille sur voie directe et HCM
    # # direct channel
    # #d_RM3_FocusR = 4
    # #d_Pup_FocusR = Pup_after_RM3 - d_RM3_FocusR
    
    # #HCM channel
    # d_APOD_LHC2 = 600e-3
    # #d_LHC1_LHC2 = d_LHC1_FoldHC3 + d_FoldHC3_ADC + d_ADC_diZELDA + d_diZELDA_APOD + d_APOD_LHC2
    # #d_LHC1_LHC2 = 1.1257
    
    # #Pup_after_LHC2 = (1/fl_D2+1/-(d_LHC1_LHC2-Pup_after_LHC1))**-1
    # d_LHC2_FoldHC4 = 50e-3
    # d_FoldHC4_cryowin = 0.4
    # d_cryowin_FocusR = 0.15
    
    # d_dummyLens = fl_dummyLens
    
    ###
    
    #d_PupHCM_FocusR = Pup_after_LHC2 + d_LHC2_FoldHC4 + d_FoldHC4_cryowin + d_cryowin_FocusR
    
    '''
    lens_diam = 2.134
    fl_exit_pupil = 37.868 #
    
    d_M6_focus_tel = 3.5
    d_exit_pupil_M6 = fl_exit_pupil - d_M6_focus_tel
    d_focus_tel_win1 = 1.85
    d_win1_win2 = 0.05
    d_win2_RM1 = 2.1
    
    #fl_exit_pupil = 0
    fl_RM1 = 2.0
    fl_RM2 = -1.0
    fl_RM3 = 2.0    
    fl_D1 = 580e-3
    fl_D2 = 600e-3
    fl_dummyLens = 250e-3
    #Distances
    
    d_RM1_RM2 = 2.0 #
    d_RM2_RM3 = 2.0 #
    d_RM3_Fold1 = 1.74 #
    d_fold1_diSCAO = 1.61 #
    d_diSCAO_FoldHC1 = 50e-3
    d_FoldHC1_FocusInHCM = 0.60
    d_FocusInHCM_FoldHC2 = 0.54
    d_FoldHC2_LHC1 = 40e-3
    d_LHC1_FoldHC3 = 90e-3
    
    ## calculs conjug pupil """"
    Pup_RM1 = fl_exit_pupil  + d_focus_tel_win1 + d_win1_win2 + d_win2_RM1
    Pup_after_RM1 = (1/fl_RM1+1/-Pup_RM1)**-1
    Pup_after_RM2 = (1/fl_RM2+1/(Pup_after_RM1-d_RM1_RM2))**-1
    Pup_after_RM3 = (1/fl_RM3+1/(Pup_after_RM2-d_RM2_RM3))**-1
    d_Pup_lHC1 = Pup_after_RM3 - d_RM3_Fold1 - d_fold1_diSCAO - d_diSCAO_FoldHC1 - d_FoldHC1_FocusInHCM - d_FocusInHCM_FoldHC2 - d_FoldHC2_LHC1
    Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 4)
    
    d_FoldHC3_diZELDA = 310e-3
    d_diZELDA_APOD = Pup_after_LHC1-d_LHC1_FoldHC3 - d_FoldHC3_diZELDA
    
    # calcul conjug pupil ds voie ZELDA
    fl_L1Z = 1310e-3
    fl_L2Z = 120e-3
    d_diZELDA_L1Z = 208e-3
    d_L1Z_F1Z = 130e-3
    d_F1Z_F2Z = 810e-3
    d_F2Z_Zmask = 370e-3
    
    ## calculs conjug pupil in ZELDA channel """"
    Pup_afterL1Z = (1/fl_L1Z+1/-(d_diZELDA_L1Z -d_diZELDA_APOD))**-1
    Pup_afterL2Z = (1/fl_L2Z+1/(Pup_afterL1Z - fl_L1Z - fl_L2Z))**-1
    
    d_Zmask_L2Z = fl_L2Z
    d_L2Z_Zcam = round(Pup_afterL2Z, 4)
    
    #calculs pour confondre la pupille sur voie directe et HCM
    # direct channel
    #d_RM3_FocusR = 4
    #d_Pup_FocusR = Pup_after_RM3 - d_RM3_FocusR
    
    #HCM channel
    d_APOD_LHC2 = 600e-3#590.7e-3
    #d_LHC1_LHC2 = d_LHC1_FoldHC3 + d_FoldHC3_diZELDA + d_diZELDA_APOD + d_APOD_LHC2
    
    #Pup_after_LHC2 = (1/fl_D2+1/-(d_LHC1_LHC2-Pup_after_LHC1))**-1
    d_LHC2_FoldHC4 = 50e-3
    d_FoldHC4_cryowin = 400e-3
    d_cryowin_FocusR = 0.15
    
    d_dummyLens = fl_dummyLens
    
    #d_PupHCM_FocusR = Pup_after_LHC2 + d_LHC2_FoldHC4 + d_FoldHC4_cryowin + d_cryowin_FocusR
    '''
    
    if endgame == 'science':
    
        DIST_vect = [d_exit_pupil_M6,
                     d_M6_focus_tel,
                     d_focus_tel_win1,
                     d_win1_win2,
                     d_win2_RM1,
                     d_RM1_RM2,
                     d_RM2_RM3,
                     d_RM3_Fold1,
                     d_fold1_diSCAO,
                     d_diSCAO_FoldHC1,
                     d_FoldHC1_FocusInHCM,
                     d_FocusInHCM_FoldHC2,
                     d_FoldHC2_LHC1,
                     d_LHC1_FoldHC3,
                     d_FoldHC3_ADC,
                     d_ADC_diZELDA,    
                     d_diZELDA_APOD,
                     d_APOD_LHC2,
                     d_LHC2_FoldHC4,
                     d_FoldHC4_cryowin,
                     d_cryowin_FocusR]
    
        FOC_vect = [0,
                    0,
                    0,
                    0,
                    fl_RM1,
                    fl_RM2,
                    fl_RM3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    fl_D1,
                    0,
                    0,
                    0,
                    0,
                    fl_D2,
                    0,
                    0,
                    0]
        
    elif endgame == 'scao':
    
        DIST_vect = [d_exit_pupil_M6,
                     d_M6_focus_tel,
                     d_focus_tel_win1,
                     d_win1_win2,
                     d_win2_RM1,
                     d_RM1_RM2,
                     d_RM2_RM3,
                     d_RM3_Fold1,
                     d_fold1_diSCAO,
                     d_diSCAO_FoldHC1,
                     d_FoldHC1_FocusInHCM]
    
        FOC_vect = [0,
                    0,
                    0,
                    0,
                    fl_RM1,
                    fl_RM2,
                    fl_RM3,
                    0,
                    0,
                    0,
                    0]
        
    elif endgame == 'zelda':
        
        DIST_vect = [d_exit_pupil_M6,
                     d_M6_focus_tel,
                     d_focus_tel_win1,
                     d_win1_win2,
                     d_win2_RM1,
                     d_RM1_RM2,
                     d_RM2_RM3,
                     d_RM3_Fold1,
                     d_fold1_diSCAO,
                     d_diSCAO_FoldHC1,
                     d_FoldHC1_FocusInHCM,
                     d_FocusInHCM_FoldHC2,
                     d_FoldHC2_LHC1,
                     d_LHC1_FoldHC3,
                     d_FoldHC3_ADC,
                     d_ADC_diZELDA,
                     d_diZELDA_L1Z,
                     d_L1Z_F1Z,
                     d_F1Z_F2Z,
                     d_F2Z_Zmask,
                     d_Zmask_L2Z,
                     d_L2Z_Zcam]
    
        FOC_vect = [0,
                    0,
                    0,
                    0,
                    fl_RM1,
                    fl_RM2,
                    fl_RM3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    fl_D1,
                    0,
                    0,
                    0,
                    fl_L1Z,
                    0,
                    0,
                    0,
                    fl_L2Z,
                    0]
    
    wfo = proper.prop_begin(lens_diam, lbd, gridsize, beam_width_ratio)
    proper.prop_multiply(wfo, ELT_pupil)
    RHO = MAS2LD(DISPER(lbd_wfs*1e6,lbd*1e6,ZD)*1000,38.542,lbd) ### ZD should be 45 or 50
    x,X,Y,R,T = VECT(gridsize,1/beam_width_ratio)
    TILT = np.exp(2*1j*pi*X*RHO)
    proper.prop_multiply(wfo, TILT)
    
    if offset != 0:
        proper.prop_multiply( wfo, np.exp(2*1j*pi*offset/lbd))
    
    proper.prop_define_entrance(wfo)
    proper.prop_lens(wfo,fl_exit_pupil)
    
    # fig,ax = plt.subplots(5,5)
    
    for k in range(len(WFE_cube[:,0,0])):
        print(k)
        proper.prop_propagate(wfo, DIST_vect[k])
        
        # E_temp = proper.prop_get_wavefront(wfo)
        # x0 = int(k//5)
        # y0 = int(np.mod(k,5))
        
        if np.abs(FOC_vect[k]) > 0:
            proper.prop_lens(wfo,FOC_vect[k])
        if np.sum(np.abs(WFE_cube[k,:,:])) > 0:
            proper.prop_add_phase( wfo, WFE_cube[k,:,:])
        if k==16 and endgame == 'science':
            proper.prop_multiply(wfo,APOD_amplitude)
        # ax[x0,y0].imshow(np.abs(E_temp)**2)
#        if endgame == 'science':
#            AZ = proper.prop_get_amplitude(wfo)
#            plt.figure()
#            plt.imshow(AZ)
            
#        if endgame == 'science':
#            #smp = proper.prop_get_beamradius(wfo)
#            #print('Beam radius: {:4.2f}mm'.format(np.round(2*smp*1e3)))
#            smp = proper.prop_get_sampling(wfo)
#            print('Beam radius: {:4.2f}mm'.format(np.round(smp*1e3*gridsize*beam_width_ratio)))
#            
    if endgame == 'science' or endgame == 'scao':
        proper.prop_propagate(wfo, d_dummyLens)
        proper.prop_lens(wfo,fl_dummyLens)
        proper.prop_propagate(wfo, d_dummyLens)
    
    if endgame == 'science' or endgame == 'zelda':
        TILT = np.exp(2*1j*pi*X*RHO)
        proper.prop_multiply(wfo, TILT)
    else:
        TILT = np.exp(-2*1j*pi*X*RHO)
        proper.prop_multiply(wfo, TILT)
    
    #PUP_fin = proper.prop_get_wavefront(wfo)
    
    PHI = proper.prop_get_phase(wfo)
    PHI = PHI/(2*np.pi)*lbd
    
    E = proper.prop_get_wavefront(wfo)
        
    return PHI, E

### OLD VERSION - DISCARDED
# def PROP_HARMONI_E2E_SCAO(endgame,lbd_wfs, lbd, ELT_pupil, gridsize, beam_width_ratio, WFE_cube):
    
#     lens_diam = 2.134
#     fl_exit_pupil = 37.868 #
    
#     d_M6_focus_tel = 3.5
#     d_exit_pupil_M6 = fl_exit_pupil - d_M6_focus_tel
#     d_focus_tel_win1 = 1.85
#     d_win1_win2 = 0.05
#     d_win2_RM1 = 2.1
    
#     #fl_exit_pupil = 0
#     fl_RM1 = 2.0
#     fl_RM2 = -1.0
#     fl_RM3 = 2.0    
#     fl_D1 = 580e-3
#     fl_D2 = 600e-3
#     fl_dummyLens = 250e-3
    
#     #Distances
#     d_RM1_RM2 = 2.0 #
#     d_RM2_RM3 = 2.0 #
#     d_RM3_Fold1 = 1.74 #
#     d_fold1_diSCAO = 1.61 #
#     d_diSCAO_FoldHC1 = 50e-3
#     d_FoldHC1_FocusInHCM = 0.60
#     d_FocusInHCM_FoldHC2 = 0.54
#     d_FoldHC2_LHC1 = 40e-3
#     d_LHC1_FoldHC3 = 90e-3
    
#     ## calculs conjug pupil """"
#     Pup_RM1 = fl_exit_pupil  + d_focus_tel_win1 + d_win1_win2 + d_win2_RM1
#     Pup_after_RM1 = (1/fl_RM1+1/-Pup_RM1)**-1
#     Pup_after_RM2 = (1/fl_RM2+1/(Pup_after_RM1-d_RM1_RM2))**-1
#     Pup_after_RM3 = (1/fl_RM3+1/(Pup_after_RM2-d_RM2_RM3))**-1
#     d_Pup_lHC1 = Pup_after_RM3 - d_RM3_Fold1 - d_fold1_diSCAO - d_diSCAO_FoldHC1 - d_FoldHC1_FocusInHCM - d_FocusInHCM_FoldHC2 - d_FoldHC2_LHC1
#     Pup_after_LHC1 = round((1/fl_D1+1/d_Pup_lHC1)**-1, 4)
    
#     d_FoldHC3_diZELDA = 310e-3
#     d_diZELDA_APOD = Pup_after_LHC1-d_LHC1_FoldHC3 - d_FoldHC3_diZELDA
    
#     # calcul conjug pupil ds voie ZELDA
#     fl_L1Z = 1310e-3
#     fl_L2Z = 120e-3
#     d_diZELDA_L1Z = 208e-3
#     d_L1Z_F1Z = 130e-3
#     d_F1Z_F2Z = 810e-3
#     d_F2Z_Zmask = 370e-3
    
#     ## calculs conjug pupil in ZELDA channel """"
#     Pup_afterL1Z = (1/fl_L1Z+1/-(d_diZELDA_L1Z -d_diZELDA_APOD))**-1
#     Pup_afterL2Z = (1/fl_L2Z+1/(Pup_afterL1Z - fl_L1Z - fl_L2Z))**-1
    
#     d_Zmask_L2Z = fl_L2Z
#     d_L2Z_Zcam = round(Pup_afterL2Z, 4)
    
#     #calculs pour confondre la pupille sur voie directe et HCM
#     # direct channel
#     #d_RM3_FocusR = 4
#     #d_Pup_FocusR = Pup_after_RM3 - d_RM3_FocusR
    
#     #HCM channel
#     d_APOD_LHC2 = 600e-3#590.7e-3
#     #d_LHC1_LHC2 = d_LHC1_FoldHC3 + d_FoldHC3_diZELDA + d_diZELDA_APOD + d_APOD_LHC2
    
#     #Pup_after_LHC2 = (1/fl_D2+1/-(d_LHC1_LHC2-Pup_after_LHC1))**-1
#     d_LHC2_FoldHC4 = 50e-3
#     d_FoldHC4_cryowin = 400e-3
#     d_cryowin_FocusR = 0.15
    
#     d_dummyLens = fl_dummyLens
    
#     #d_PupHCM_FocusR = Pup_after_LHC2 + d_LHC2_FoldHC4 + d_FoldHC4_cryowin + d_cryowin_FocusR
    
#     if endgame == 'SCAO':
    
#         DIST_vect = [d_exit_pupil_M6,
#                      d_M6_focus_tel,
#                      d_focus_tel_win1,
#                      d_win1_win2,
#                      d_win2_RM1,
#                      d_RM1_RM2,
#                      d_RM2_RM3,
#                      d_RM3_Fold1,
#                      d_fold1_diSCAO,
#                      d_diSCAO_FoldHC1,
#                      d_FoldHC1_FocusInHCM]
    
#         FOC_vect = [0,
#                     0,
#                     0,
#                     0,
#                     fl_RM1,
#                     fl_RM2,
#                     fl_RM3,
#                     0,
#                     0,
#                     0,
#                     0]
        
#     elif endgame == 'zelda':
        
#         DIST_vect = [d_exit_pupil_M6,
#                      d_M6_focus_tel,
#                      d_focus_tel_win1,
#                      d_win1_win2,
#                      d_win2_RM1,
#                      d_RM1_RM2,
#                      d_RM2_RM3,
#                      d_RM3_Fold1,
#                      d_fold1_diSCAO,
#                      d_diSCAO_FoldHC1,
#                      d_FoldHC1_FocusInHCM,
#                      d_FocusInHCM_FoldHC2,
#                      d_FoldHC2_LHC1,
#                      d_LHC1_FoldHC3,
#                      d_FoldHC3_diZELDA,
#                      d_diZELDA_L1Z,
#                      d_L1Z_F1Z,
#                      d_F1Z_F2Z,
#                      d_F2Z_Zmask,
#                      d_Zmask_L2Z,
#                      d_L2Z_Zcam]
    
#         FOC_vect = [0,
#                     0,
#                     0,
#                     0,
#                     fl_RM1,
#                     fl_RM2,
#                     fl_RM3,
#                     0,
#                     0,
#                     0,
#                     0,
#                     0,
#                     fl_D1,
#                     0,
#                     0,
#                     fl_L1Z,
#                     0,
#                     0,
#                     0,
#                     fl_L2Z,
#                     0]
    
#     wfo = proper.prop_begin(lens_diam, lbd, gridsize, beam_width_ratio)
#     proper.prop_multiply(wfo, ELT_pupil)
#     RHO = MAS2LD(DISPER(lbd_wfs*1e6,lbd*1e6,45)*1000,38.542,lbd)
#     x,X,Y,R,T = VECT(gridsize,1/beam_width_ratio)
#     TILT = np.exp(2*1j*pi*X*RHO)
#     proper.prop_multiply(wfo, TILT)
#     proper.prop_define_entrance(wfo)
#     proper.prop_lens(wfo,fl_exit_pupil)
    
#     for k in range(len(WFE_cube[:,0,0])):
#         proper.prop_propagate(wfo, DIST_vect[k])
#         if np.abs(FOC_vect[k]) > 0:
#             proper.prop_lens(wfo,FOC_vect[k])
#         if np.sum(np.abs(WFE_cube[k,:,:])) > 0:
#             proper.prop_add_phase( wfo, WFE_cube[k,:,:])
# #        if endgame == 'science':
# #            AZ = proper.prop_get_amplitude(wfo)
# #            plt.figure()
# #            plt.imshow(AZ)
            
#     if endgame == 'SCAO':
#         proper.prop_propagate(wfo, d_dummyLens)
#         proper.prop_lens(wfo,fl_dummyLens)
#         proper.prop_propagate(wfo, d_dummyLens)

#     TILT = np.exp(2*1j*pi*X*RHO)
#     proper.prop_multiply(wfo, TILT)
    
#     #PUP_fin = proper.prop_get_wavefront(wfo)
    
#     PHI = proper.prop_get_phase(wfo)
#     PHI = PHI/(2*np.pi)*lbd
    
#     E = proper.prop_get_wavefront(wfo)
        
#     return PHI, E

def WRITE_DAT(filename,A):
    f = open(filename + '.dat', 'w')
    N = len(A)
    for k in range(N):
        for l in range(N):
            f.write("{0}".format(1.*A[k,l]))
            if l==(N-1):
                f.write("\n")
            else:
                f.write("\t")

def READ_DAT(filename):
    A = np.fromfile(filename + '.dat', dtype=float, sep='\n')
    return A

def minterp2d(x,y,A,x2,y2,mode='linear',my_fill_value=0):
    A_f = interp2d(x,y,A,kind=mode,fill_value=my_fill_value)
    B = A_f(x2,y2)
    return B

def ZERNIKE(i,r,theta):
    zernike_index = fits.getdata(WORKPATH + 'zernike_index.fits')
    n = zernike_index[i,0]
    m = zernike_index[i,1]
    if m==0:
        Z = np.sqrt(n+1)*zrf(n,0,r)
    else:
        if i//2 == i/2:
            Z = np.sqrt(2*(n+1))*zrf(n,m,r)*np.cos(m*theta)
        else:
            Z = np.sqrt(2*(n+1))*zrf(n,m,r)*np.sin(m*theta)
    return Z

def ZERNIKE_PROJECT(WF,P,D,N_order,basis='GS'):
    #Projects the WF over N_order Zernike modes, and returns the modes coefficients, and 2D maps
    #By default it adapts to the pupil P to modify the classical Zernike basis using the arbitrary_basis function that itself uses the Graham-Schmidt method
    #You can stick to the classical Zernike basis by choosing basis='TrueZernike'
    N = len(WF)
    Z_coeff = np.zeros(N_order)
    x,X,Y,R,T = VECT(N,2*D)
    if basis=='GS':
        ZER = arbitrary_basis(P,N_order,rho=R,theta=T,outside=0.0)
    elif basis =='TrueZernike':
        ZER = np.zeros((N_order,N,N))
        for i in range(N_order):
            ZER[i] = ZERNIKE(i,R,T)
    else: 
        print('This modal choice is not valid; Switching to modified Zernike')
        ZER = arbitrary_basis(P,N_order,rho=R,theta=T,outside=0.0)
    for k in range(0,N_order):
        Z = ZER[k]*WF
        Z_coeff[k]= np.sum(Z[P>0])/np.sum(P)
    return (Z_coeff,ZER)

def FOURIER_PROJECT(WF,P,D,N_order):
    #Projects the WF over N_order Fourier modes, and returns the modes coefficients, and 2D maps
    N = len(WF)
    F_coeff = np.zeros((N_order,4))
    x,X,Y,R,T = VECT(N,D)
    F_modes = np.zeros((N_order,4,N,N))
    for i in range(N_order):
        F_modes[i,0] = np.cos(2*np.pi*i*X)
        F_modes[i,1] = np.sin(2*np.pi*i*X)
        F_modes[i,2] = np.cos(2*np.pi*i*Y)
        F_modes[i,3] = np.sin(2*np.pi*i*Y)    
    for k in range(N_order):
        F_CX = F_modes[k,0]*WF
        F_SX = F_modes[k,1]*WF
        F_CY = F_modes[k,2]*WF
        F_SY = F_modes[k,3]*WF
        F_coeff[k,0]= np.sum(F_CX[P>0])/np.sum(P)
        F_coeff[k,1]= np.sum(F_SX[P>0])/np.sum(P)
        F_coeff[k,2]= np.sum(F_CY[P>0])/np.sum(P)
        F_coeff[k,3]= np.sum(F_SY[P>0])/np.sum(P)
    return (F_coeff,F_modes)

def ZERNIKE_PUP(i,r,theta):
    zernike_index = fits.getdata(WORKPATH + 'zernike_index.fits')
    n = zernike_index[i,0]
    m = zernike_index[i,1]
    if m==0:
        Z = np.sqrt(n+1)*zrf(n,0,r)
    else:
        if i//2 == i/2:
            Z = np.sqrt(2*(n+1))*zrf(n,m,r)*np.cos(m*theta)
        else:
            Z = np.sqrt(2*(n+1))*zrf(n,m,r)*np.sin(m*theta)
    return Z

def zrf(n, m, r):
    R=0
    N_max = int((n-m)//2+1)
    for s in range(0,N_max):
        num = (-1)**s * gamma(n-s+1)
        denom = gamma(s+1) * gamma((n+m)/2-s+1) * gamma((n-m)/2-s+1)
        R = R + num / denom * r**(n-2*s)
    return R

def AB2V_CONVFACT(lambda_0, BW, WORKPATH='/Users/carlotal/HARMONI/HC-PSF/common_files/'):
    """
    Converts an AB_mag to a VEGA magnitude, for a given lambda_0 [nm] and a
    BW bandwidth [nm];
    It relies on the F_nu data for VEGA from the STScI (alpha_lyr_stis_003.fits)
    The product CV is such that CV=mAB-mVEGA, or said otherwise: mVEGA=mAB-CV
    """
    DATA_V=fits.getdata(WORKPATH + 'VEGA_Fnu.fits')
    lbd=np.linspace(lambda_0-BW/2,lambda_0+BW/2,1000)
    dlbd=lbd[1]-lbd[0]
    F_nu_f = interp1d(DATA_V[:,0],DATA_V[:,1],fill_value='extrapolate')
    F_nu = F_nu_f(lbd)    
    S1=np.sum(F_nu*dlbd*3.34e-19*(lbd*10)**2);
    S2=np.sum(dlbd*np.ones(1000))
    CV = -2.5*np.log10(S1/S2)-48.6
    return CV

def PhotonCount(T,S,lambda_0,BW,mag,mag_type,throughput,WORKPATH ='/Users/carlotal/HARMONI/HC-PSF/common_files/'):
    """
    T:          exposure time [s]
    S:          telescope surface [m2]
    lambda_0:   central wavelength [nm]
    BW:         bandwidth [nm]
    mag:        apparent magnitude of the star
    mag_type:   magnitude system: either 'AB' or 'VEGA'
    throughput: telescope+instrument throughput. It does not include the coronagraph throughput.
    """
    if mag_type == 'AB':
        AB_mag=mag
    elif mag_type == 'VEGA':
        AB_mag=mag+AB2V_CONVFACT(lambda_0, BW,WORKPATH)
    else:
        AB_mag=mag
        print('Error! Choose between AB mag or VEGA. Switching to AB mag')

    PC = T*S*throughput*1.51*1e26/(1e-9*lambda_0*10)*10**(-(AB_mag+48.6)/2.5)*BW*1e-9*10*10000
    return PC

def PHOT2MAG(NPhot,T,S,lambda_0,BW,throughput,WORKPATH = '/Users/carlotal/HARMONI/HC-PSF/common_files/'):
    AB_mag = -48.6-2.5*np.log10(NPhot/(BW*1e-9*10*10000)/(T*S*throughput*1.51*1e26/(1e-9*lambda_0*10)))
    mag = AB_mag-AB2V_CONVFACT(lambda_0,BW,WORKPATH)
    return mag

def MakeBinary(A):
    N = len(A)
    for j in range(N):
        for i in range(N):
            old_pixel = A[i,j]
            new_pixel = np.round(A[i,j])
            A[i,j] = new_pixel
            quant_error = old_pixel-new_pixel
            
            if ((i+1)<N-1):
                A[i+1,j]=A[i+1,j]+7/16*quant_error

            if (i>0 and j<N-1):
                A[i-1,j+1]=A[i-1,j+1]+3/16*quant_error

            if ((j+1)<N-1):
                A[i,j+1]=A[i,j+1]+5/16*quant_error

            if ((i+1)<N-1 and (j+1)<N-1):
                A[i+1,j+1]=A[i+1,j+1]+1/16*quant_error

    return A

def AZAV(I,OWA,PhotAp):
    #photap meme dim que OWA
    #I = image
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
    #I_m = moyenne
    #I_s = écart type
    #f = rayon anneaux
    return (I_m,I_s,f)

def AZAV_SECTIONS(I,OWA,PhotAp):
    #goal is to pave each annulus with azymutal sections that have a 1 
    #photometric aperture width, compute for each of them the mean flux, and
    #compute from these N values the standard deviation 
    L = len(I)
    u,U,V,R,T = VECT(L,2*OWA)
    f = u[int(L/2):L]
    Lf = len(f)
    I_m = np.zeros(Lf)
    I_s = np.zeros(Lf)
    for k in range(Lf):
        alpha_photap = 2*np.arctan2(PhotAp/2,f[k])
        nbr_sections = int(np.round(2*np.pi/alpha_photap))
        # print(nbr_sections)
        DATA_STAT = np.zeros(nbr_sections)
        T_map = np.round(nbr_sections*(T+np.pi+np.pi/nbr_sections)/(2*np.pi)).astype(int)
        # MAP = np.zeros((len(I),len(I)))
        # MAP_init = np.zeros((len(I),len(I)))
        # REG_RAD = (R<(f[k]+PhotAp/2))*(R>(f[k]-PhotAp/2))*(I != 0)
        for k_alpha in range(nbr_sections):
            REG = (R<(f[k]+PhotAp/2))*(R>(f[k]-PhotAp/2))*(I != 0)*(T_map==(k_alpha+1))
            DATA_STAT[k_alpha] = np.sum(I[REG==True])*np.pi/4 #sum of intensity in FWHM
            # MAP[REG==1] = DATA_STAT[k_alpha]
        # MAP_INIT = I*REG_RAD
        I_m[k] = np.mean(DATA_STAT)
        I_s[k] = np.std(DATA_STAT)
        # if k == Lf//2:
        #     plt.figure()
        #     # plt.hist(DATA_STAT)
        #     plt.imshow(MAP)
        #     plt.title(I_s[k])
            
        #     plt.figure()
        #     # plt.hist(DATA_STAT)
        #     plt.imshow(MAP_INIT)
        #     plt.title(I_s[k])
    return (I_m,I_s,f)

def AZAV_STAT(I,OWA,PhotAp,fraction):
    L = len(I)
    u,U,V,R,T = VECT(L,2*OWA)
    f = u[int(L/2+1):L]
    Lf = len(f)
    I_m = np.zeros(Lf)
    I_sup = np.zeros(Lf)
    for k in range(Lf):
        REG = (R<(f[k]+PhotAp/2))*(R>(f[k]-PhotAp/2))
        I_m[k] = np.mean(I[REG==True])
        I_sup[k] = np.quantile(I[REG==True],fraction)
    return (I_m,I_sup,f)

def MAKE_ELT(N,D,n_miss,ref_err,path=WORKPATH):
    #Problem of coherence between ESO pupil and this one - Use ESO pupil if you can
    #D = D *1.021 ???
    C = fits.getdata(path + 'Coord_ELT.fits') 
    #
    N=N+N%2
    W=1.45*np.cos(np.pi/6.0)
    [x,X,Y,R,T]=VECT(N,D)
    #
    hexa=np.zeros([N,N])
    MISS_LIST=np.round(np.random.random(n_miss)*797)+1
    #
    for k in range(0,len(C)):
        Xt=X+C[k,0]
        Yt=Y+C[k,1]
        hexa=hexa+(k+1)*(Yt<0.5*W)*(Yt>=-0.5*W)*((Yt+np.tan(np.pi/3)*Xt)<W)*((Yt+np.tan(np.pi/3)*Xt)>=-W)*((Yt-np.tan(np.pi/3)*Xt)<W)*((Yt-np.tan(np.pi/3)*Xt)>=-W)
    #
    color=hexa
    hexa=np.double(hexa>0)
    MS = np.ones((N,N))
    #
    if n_miss > 0:
        for k in range(0,n_miss):
            MS *= (1-(color==MISS_LIST[k]))
    #
    t=0.5
    hexa=MS*hexa*(np.abs(X)>t/2)*((Y<(np.tan(np.pi/6)*X-t/2/np.cos(np.pi/6)))+(Y>(np.tan(np.pi/6)*X+t/2/np.cos(np.pi/6))))*((Y<(np.tan(-np.pi/6)*X-t/2/np.cos(-np.pi/6)))+(Y>(np.tan(-np.pi/6)*X+t/2/np.cos(-np.pi/6))))
    
    referr = np.zeros([N,N])
    for k in range(0,798):
        referr = referr + (color==(k+1))*(1-ref_err*np.random.random())
    
    hexa = referr * hexa
    
    #
    return (hexa,color,MS,referr)

def EELTNLS(N,a1,a2,a3):
    D=1
    d=0.3
    t=1.4/100
    Th=30*pi/180
    x,X,Y,R,T = VECT(N,D)
    a1=a1/100
    a2=a2/100
    a3=a3/100
    ELT=(R<=D/2*a1)*(R>=d/2*a2)*(((Y <= (X-t*a3/(2*np.sin(Th)))*np.tan(Th)) + (Y >= (X+t*a3/np.sin(Th)/2)*np.tan(Th)))*(X>t*a3/2)*(Y>0))
    ELT=ELT+np.fliplr(ELT)
    LS=ELT+np.flipud(ELT)
    LS=np.transpose(LS)
    return LS

def APPLY_CROSSTALK(I):
    L,M,N = np.shape(I)
    FWHM_L_pix = 2
    FWHM_X_pix = 1
    sL = FWHM_L_pix/(2*np.sqrt(2*np.log(2)))
    sX = FWHM_X_pix/(2*np.sqrt(2*np.log(2)))
    I_out = gaussian_filter(I,sigma=(sL,0,sX))
    I_mean = np.mean(I,axis=0)
    for k in range(L):
        I_out[k,:,:] = I_out[k,:,:]*(1-0.5/100)+0.5/100*I_mean
    return I_out
    
def BEAMSHIFT(N,SCREEN,Pupil_shift,D_opt_vect,Z_opt_vect,lambda_wfs,lambda_vect,alpha_zen,rot_ang_start,rot_vect,com_vect,gamma):
    #This function creates a wavefront error map and shifts it.
    N_optics = len(D_opt_vect)
    if np.shape(lambda_vect)==():
        WF = np.zeros([N,N])
        WF_NCPA = np.zeros([N,N])
        shift_vect = np.zeros(N_optics)
        alpha_disper = gamma*DISPER(1e6*lambda_wfs,1e6*lambda_vect,alpha_zen)
        alpha_disper = np.tan(alpha_disper/3600*pi/180/2)*2
        for l in range(N_optics):
            shift_vect[l] = alpha_disper*Z_opt_vect[l]/D_opt_vect[l]
        M=1358
        MNm = int(M/2-N/2)
        MNp = int(M/2+N/2)
    #1x = VECT1(len(SCREEN[:,:,1]),len(SCREEN[:,:,1])/1024)
        for l in range(N_optics):
            init_screen=SCREEN[:,:,l]
            if (rot_vect[l]==1):
                init_screen = rotate(init_screen, 90-alpha_zen-rot_ang_start, order=0, reshape=False)
            
            #interp_f = interp2d(x,x,init_screen,kind='linear',fill_value=0)
            #translated_screen = interp_f(x,x+shift[l]+Pupil_shift[l])
            pix_shift_0 = (shift_vect[l]+Pupil_shift[l])*1024
            translated_screen = shift(1.0*init_screen, [pix_shift_0,0], output=None, order=1, mode='constant', cval=0.0, prefilter=True)
            translated_screen = translated_screen[MNm:MNp,MNm:MNp]
            if (com_vect[l]==1):
                WF=WF+translated_screen
            else:
                WF_NCPA=WF_NCPA+translated_screen
    else:
        N_lambda = len(lambda_vect)
        WF = np.zeros([N,N,N_lambda])
        WF_NCPA = np.zeros([N,N,N_lambda])
        alpha_disper = np.zeros(N_lambda)
        shift_vect = np.zeros([N_lambda,N_optics])
        for k in range(N_lambda):
            alpha_disper[k] = gamma*DISPER(1e6*lambda_wfs,1e6*lambda_vect[k],alpha_zen)
            for l in range(N_optics):
                shift_vect[k,l] = np.tan(alpha_disper[k]/3600*pi/180/2)*2*Z_opt_vect[l]/D_opt_vect[l]
        M=1358
        MNm = int(M/2-N/2)
        MNp = int(M/2+N/2)
        #x = VECT1(len(SCREEN[:,:,1]),len(SCREEN[:,:,1])/1024)
        for l in range(N_optics):
            init_screen=SCREEN[:,:,l]
            if (rot_vect[l]==1):
                init_screen = rotate(init_screen, 90-alpha_zen-rot_ang_start, order=1, reshape=False)
            for k in range(N_lambda):
                #interp_f = interp2d(x,x,init_screen,kind='cubic',fill_value=0)
                #translated_screen = interp_f(x,x+shift[k,l]+Pupil_shift[l])
                pix_shift_0 = (shift_vect[k,l]+Pupil_shift[l])*1024
                translated_screen = shift(1.0*init_screen, [pix_shift_0,0], output=None, order=1, mode='constant', cval=0.0, prefilter=True)
                translated_screen = translated_screen[MNm:MNp,MNm:MNp]
                if (com_vect[l]==1):
                    WF[:,:,k]=WF[:,:,k]+translated_screen
                else:
                    WF_NCPA[:,:,k]=WF_NCPA[:,:,k]+translated_screen
    return (WF,WF_NCPA)

def M4Magic(WORKPATH,WF,P,X_off,Y_off):
    nact = 4338
    dim  = 64

    inf_func_name = WORKPATH + 'inf_func_M4.fits'
    pos_name = WORKPATH + 'positions_inf_func_M4.fits'
    MMI_name = WORKPATH + 'MMI_dd_wo_piston_4338_spider_optical_pupil.fits'
    DDWOP = fits.getdata(inf_func_name)
    DDPOSOK = fits.getdata(pos_name)
    MMI=fits.getdata(MMI_name)
    coef = np.zeros(nact)
    for k in range(nact):
        Xk = DDPOSOK[1,k]-X_off
        Yk = DDPOSOK[0,k]-Y_off
        x0 = int(Xk-dim/2-1)
        x1 = int(Xk+dim/2-1)
        y0 = int(Yk-dim/2-1)
        y1 = int(Yk+dim/2-1)
        coef[k] = np.sum(WF[x0:x1,y0:y1]*DDWOP[k,:,:]*P[x0:x1,y0:y1]) / np.sum(P)
    commands = np.dot(np.transpose(coef),np.transpose(MMI))
    WF2 = np.zeros(np.shape(P))
    for k in range(nact):
        Xk = DDPOSOK[1,k]-X_off
        Yk = DDPOSOK[0,k]-Y_off
        x0 = int(Xk-dim/2-1)
        x1 = int(Xk+dim/2-1)
        y0 = int(Yk-dim/2-1)
        y1 = int(Yk+dim/2-1)
        WF2[x0:x1,y0:y1] = WF2[x0:x1,y0:y1]+commands[k]*DDWOP[k,:,:]
    return WF2

def MPS(N,power):
    f_max = N/2
    f_vect,Xf,Yf,Rf,Tf = VECT(N,2*f_max)
    screen = np.fft.fft2(np.exp(2*1j*pi*(np.random.rand(N,N)-0.5))*Rf**(-power/2))
    KER = [[-1,1],[1,-1]]
    if N//2-N/2==0:
        KER = np.pad(KER,[int(N/2-1),int(N/2-1)],'wrap')
    else:
        KER = np.pad(KER,[int(N/2),int(N/2)],'wrap')
        KER = KER[0:N,0:N]
    screen = np.real(screen*KER)
    return screen

def MPS_DSP(N,DSP):
    # generates a random phase screen of NxN points based on a 2D array representing the PSD
    # The extent of the PSD must be 2*f_max where f_max is the maximum frequency is N/2 
    f_max = N/2
    f_vect,Xf,Yf,Rf,Tf = VECT(N,2*f_max)
    screen = np.fft.fft2(np.exp(2*1j*pi*(np.random.rand(N,N)-0.5))*np.sqrt(DSP))
    KER = [[-1,1],[1,-1]]
    if N//2-N/2==0:
        KER = np.pad(KER,[int(N/2-1),int(N/2-1)],'wrap')
    else:
        KER = np.pad(KER,[int(N/2),int(N/2)],'wrap')
        KER = KER[0:N,0:N]
    screen = np.real(screen*KER)
    return screen

def DSP(WF,gamma,P,alpha_min,alpha_max):
    #WF: wavefront
    #gamma : ratio of array physical size and pupil physical size
    #P: pupil array (potentially larger than pupil)
    # alpha_min & alpha_max : min and max spatial frequencies
    [Nx,Ny] = np.shape(WF)
    WF = WF - np.mean(WF[P>1e-16])
    #OWA = Nx / 2. #original
    OWA = Nx / 2. / gamma #changed on June 8 2022 - TEST MODE
    #M = 2*int(2*OWA/gamma)
    M = int(2*OWA)
    u,U,V,R,T = VECT(M,2*OWA)
    du = u[1]-u[0]
    E = ft_BASIC(P*WF,gamma,2*OWA,M,1)
    I = np.abs(E)**2
    if alpha_max > OWA:
        alpha_max = OWA
    REG = (R<alpha_max)*(R>alpha_min)
    Value = np.sqrt(np.nansum(I[REG])*du**2)
    return (Value,I)

def ABEVAL(WF,gamma,P,alpha_min,alpha_max):
    #Alternative version of DSP where the wavefront is Fourier filtered to 
    #only keep a specific range of aberrations, and the result is analyzed
    #by measuring the standard deviation of the wavefront errors
    #WF: wavefront
    #gamma : ratio of array physical size and pupil physical size
    #P: pupil array (potentially larger than pupil)
    # alpha_min & alpha_max : min and max spatial frequencies
    [Nx,Ny] = np.shape(WF)
    WF = WF - np.mean(WF[P>1e-6])
    #OWA = Nx / 2. #original
    OWA = Nx / 2. / gamma #changed on June 8 2022 - TEST MODE
    #M = 2*int(2*OWA/gamma)
    M = int(2*OWA)
    u,U,V,R,T = VECT(M,2*OWA)
    # du = u[1]-u[0]
    E = ft_BASIC(P*WF,gamma,2*OWA,M,1)
    # E_P = ft_BASIC(P,gamma,2*OWA,M,1)
    # I = np.abs(E)**2
    if alpha_max > OWA:
        alpha_max = OWA
    REG = (R<alpha_max)*(R>alpha_min)
    # ClassicalDSP_value = np.sqrt(np.nansum(I[REG])*du**2)
    WF_filtered = np.real(ft_BASIC(E*REG,2*OWA,gamma,Nx,-1))
    # P_filtered = np.real(ft_BASIC(E_P*REG,2*OWA,gamma,Nx,-1))
    P_nan = np.copy(P)
    # P_nan = np.copy(P_filtered)
    P_nan[P<1e-1] = np.nan
    WF_filtered -= np.nanmean(WF_filtered*P_nan)
    Value = np.nanstd(WF_filtered*P_nan)
    # return (Value,ClassicalDSP_value, WF_filtered, P_filtered)
    return Value

def DSP_vect(WF,gamma,P,alpha_min_vect,alpha_max_vect):
    #WF: wavefront
    #gamma : ratio of array physical size and pupil physical size
    #P: pupil array (potentially larger than pupil)
    # alpha_min & alpha_max : min and max spatial frequencies
    [Nx,Ny] = np.shape(WF)
    WF = WF - np.mean(WF[P>0.5])
    #OWA = Nx / 2. #original
    OWA = Nx / 2. / gamma #changed on June 8 2022 - TEST MODE
    #M = 2*int(2*OWA/gamma)
    M = int(2*OWA)
    u,U,V,R,T = VECT(M,2*OWA)
    du = u[1]-u[0]
    E = ft_BASIC(P*WF,gamma,2*OWA,M,1)
    I = np.abs(E)**2
    Value = np.zeros(len(alpha_min_vect))
    for k in range(len(alpha_min_vect)):    
        if alpha_max_vect[k] > OWA:
            alpha_max = OWA
        else: alpha_max = alpha_max_vect[k]
        REG = (R<alpha_max)*(R>alpha_min_vect[k])
        Value[k] = np.sqrt(np.sum(I[REG])*du**2)
    return (Value,I)

def DSPHR(WF,gamma,P,alpha_min,alpha_max):
    #WF: wavefront
    #gamma : ratio of array physical size and pupil physical size
    #P: pupil array (potentially larger than pupil)
    # alpha_min & alpha_max : min and max spatial frequencies
    [Nx,Ny] = np.shape(WF)
    WF = WF - np.mean(WF[P>0])
    OWA = Nx / 2.
    #M = 2*int(2*OWA/gamma)
    M = int(8*OWA)
    u,U,V,R,T = VECT(M,2*OWA)
    du = u[1]-u[0]
    E = ft_BASIC(P*WF,gamma,2*OWA,M,1)
    I = np.abs(E)**2
    if alpha_max > OWA:
        alpha_max = OWA
    REG = (R<alpha_max)*(R>alpha_min)
    Value = np.sqrt(np.sum(I[REG])*du**2)
    return (Value,I)

def MAKE_WF_CUBE(N,D,WFE):
    # N: number of points
    # D: physical diameter of array, assuming pupil has diameter 1
    # WFE : vector of wavefront errors inside the pupil
    L = len(WFE)
    x,X,Y,R,T = VECT(N,D)
    P = (R<0.5)
    WFCube = np.zeros((L,N,N))
    for k in range(L):
        SQUARE = MPS(N,2) #assuming f-2 power here
        V,I = DSP(SQUARE,D,P,0,10000)
        WFCube[k,:,:] = SQUARE/V*WFE[k]
    return WFCube

def DISPER(lbd_min,lbd_max,theta,TC=7.5,RH=15,P=712):
    #lbd_min & lbd_max must be given in microns
    #theta must be given in degrees
    #the result is given in arcsec
    if lbd_min > lbd_max:
        new_lbd_max = lbd_min
        lbd_min = lbd_max
        lbd_max = new_lbd_max
        sign = -1
    else:
        sign = 1
    #TC = 12
    T  = TC + 273.16
    #RH = 15
    #P  = 743
    lbd = np.linspace(lbd_min,lbd_max,2)
    PS = -10474.0+116.43*T-0.43284*T**2+0.00053840*T**3
    P2 = RH/100.0*PS
    P1 = P-P2
    D1 = P1/T*(1.0+P1*(57.90*1.0e-8-(9.3250*1.0e-4/T)+(0.25844/T**2)))
    D2 = P2/T*(1.0+P2*(1.0+3.7e-4*P2)*(-2.37321e-3+(2.23366/T)-(710.792/T**2)+(7.75141e4/T**3)))
    S0 = 1.0/lbd_min
    S  = 1.0/lbd
    N0_1 = 1.0e-8*((2371.34+683939.7/(130-S0**2)+4547.3/(38.9-S0**2))*D1+(6487.31+58.058*S0**2-0.71150*S0**4+0.08851*S0**6)*D2)
    N_1  = 1.0e-8*((2371.34+683939.7/(130-S**2)+4547.3/(38.9-S**2))*D1+(6487.31+58.058*S**2-0.71150*S**4+0.08851*S**6)*D2)
    D = sign*np.max(np.tan(theta*pi/180)*(N0_1-N_1)*206264.8)
    return D

def DISPER_ADC(lbd_min,lbd_max,theta,theta_min=5,theta_max=50,T_mean=7.5,RH_mean=15,P_mean=712,T_min=0,T_max=15,P_min=662,P_max=762): 
    #Computes the residual dispersion (in arcsec) after the ADC
    #Min and max values of temperature and pressure are taken into account to find the right balance
    #lbd_min & lbd_max must be given in microns
    #theta_min and other ZD must be given in degrees
    #the result is given in arcsec
    D = DISPER(lbd_min,lbd_max,theta,T_mean,RH_mean,P_mean)
    #D = np.abs(D-(DISPER(lbd_min,lbd_max,theta_min)+DISPER(lbd_min,lbd_max,theta_max))/2) 
    D = D-(DISPER(lbd_min,lbd_max,theta_min,TC=T_max,RH=15,P=P_min)+DISPER(lbd_min,lbd_max,theta_max,TC=T_min,RH=15,P=P_max))/2
    
    #if lbd_min > lbd_max:
    #    D = -D
    return D

def LST(UTC_TIME, LAT_in=-24.5892,LON_in=-70.1922):
    #UTC_TIME should be in the following format: 'YYYY-MM-DDTHH:MM:SS'
    observing_location = EarthLocation(lat=LAT_in*u.deg, lon=LON_in*u.deg)
    observing_time = Time(datetime.fromisoformat(UTC_TIME),scale='utc',location=observing_location)
    LST = (observing_time.sidereal_time('mean')).hour
    return LST

def HA(alpha,LST):
    #alpha: right ascension of the star (h)
    #LST: local sidereal time (h) ; to be obtained from the LST routine
    #HA: hour angle (h)
    HA = LST-alpha
    return HA

def ELEVATION(D,H,LAT=-24.5892):
    #Elevation function returns the elevation E of a star with a declination D when observed at an hour angle H
    #D [degrees]
    #H [hours]
    #E [degrees]
    L = LAT*pi/180
    if np.max(np.shape([D])) > 1 and np.max(np.shape([H])) > 1:
        D2,H2 = np.meshgrid(D,H)
        E = 180/pi*np.arcsin(np.sin(L)*np.sin(D2*pi/180)+np.cos(L)*np.cos(D2*pi/180)*np.cos(H2*15*pi/180))
    else:
        E = 180/pi*np.arcsin(np.sin(L)*np.sin(D*pi/180)+np.cos(L)*np.cos(D*pi/180)*np.cos(H*15*pi/180))
    return E

def AZIMUTH(D,H):
    #Elevation function returns the elevation E of a star with a declination D when observed at an hour angle H
    #D [degrees]
    #H [hours]
    #E [degrees]
    L = -24.5892*pi/180
    if np.max(np.shape([D])) > 1 and np.max(np.shape([H])) > 1:
        D2,H2 = np.meshgrid(D,H)
        A = 180/pi*np.arccos((np.cos(L)*np.sin(D2*pi/180)-np.sin(L)*np.cos(D2*pi/180)*np.cos(H2*15*pi/180))/np.cos(np.pi/180*ELEVATION(D,H)))*np.sign(H2)
    else:
        A = 180/pi*np.arccos((np.cos(L)*np.sin(D*pi/180)-np.sin(L)*np.cos(D*pi/180)*np.cos(H*15*pi/180))/np.cos(np.pi/180*ELEVATION(D,H)))*np.sign(H)
    return A

def PARANG(D,H,LAT=-24.5892):
    L  = LAT #-24.5892 for ELT
    if np.max(np.shape([D])) > 1 and np.max(np.shape([H])) > 1:
        D2,H2 = np.meshgrid(D,H)
        PA = 180/pi*np.arctan2(np.sin(H2*15*pi/180),(np.cos(D2*pi/180)*np.tan(L*pi/180)-np.sin(D2*pi/180)*np.cos(H2*15*pi/180)))
    else:
        PA = 180/pi*np.arctan2(np.sin(H*15*pi/180),(np.cos(D*pi/180)*np.tan(L*pi/180)-np.sin(D*pi/180)*np.cos(H*15*pi/180)))
    if PA<0: PA+=360
    return PA

def PSF(E):
    I = np.log10(np.abs(E)**2/np.max(np.ravel(np.abs(E)**2)))
    return I

def LD2MAS(LD, D, lbd):
    Out = LD*lbd/D*180*3600*1000/pi
    return Out

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def VECT1(N, D):
    x = np.linspace(-D/2.0, D/2.0, N, endpoint=False) + D/(2.0*N)
    return x

def VECT(N, D):
    x = np.linspace(-D/2.0, D/2.0, N, endpoint=False) + D/(2.0*N)
    X , Y = np.meshgrid(x, x, sparse = False)
    R , T = cart2pol(X,Y)
    return (x, X, Y, R, T)

def MAS2LD(MAS, D, lbd):
    Out = MAS/(lbd/D*180*3600/pi*1000)
    return Out

def ft_BASIC(Ein, W1, W2, n2, direction):
    nx, ny = np.shape(Ein)
    dx = W1/nx
    du = W2/n2
    x = (np.linspace(-nx/2,nx/2,nx,endpoint=False)+0.5)*dx
    u = (np.linspace(-n2/2,n2/2,n2,endpoint=False)+0.5)*du
    x_u = np.outer(x,u)
    if np.abs(direction) != 1:
        print('Error: direction must equal 1 (direct FT) or -1 (inverse FT)')
        print('Setting direction to 1')
    Eout = np.dot(np.transpose(np.exp(-direction*2.*pi*1j*x_u)),Ein) 
    Eout = np.dot(Eout,np.exp(-direction*2.*pi*1j*x_u))
    Eout = Eout*dx*dx
    return Eout

def IM_RGB(P,D_pup,OWA,pix_per_lbd,bandwidth,N_lbd,V_log_min=6):
    
    M = pix_per_lbd*2*OWA*2
    lbd_vect = 1+np.linspace(-bandwidth/2,bandwidth/2,N_lbd)

    I_data = np.zeros((N_lbd,M,M))
    for k in range(N_lbd):
        I_temp = PSF(ft(P,1,D_pup,D_pup,2*OWA,2*OWA,lbd_vect[k],M,M))
        I_temp += V_log_min
        I_temp /= V_log_min
        I_data[k] = I_temp

    # Computation of weights
    W = 0.5*2*N_lbd/3
    WR = np.exp(-(np.arange(N_lbd))**2/(2*(2*W/2.355)**2))
    WG = np.exp(-(np.arange(N_lbd)-N_lbd/2+0.5)**2/(2*(W/2.355)**2))
    WB = np.exp(-(np.arange(N_lbd)-N_lbd+1)**2/(2*(2*W/2.355)**2))

    # WR /= np.sum(WR)
    # WG /= np.sum(WG)
    # WB /= np.sum(WB)
    
    IR = np.average(I_data,axis=0,weights=WR)
    IG = np.average(I_data,axis=0,weights=WG)
    IB = np.average(I_data,axis=0,weights=WB)

    IM_BB = np.zeros((M,M,3))
    IM_BB[:,:,2]=IR
    IM_BB[:,:,1]=IG
    IM_BB[:,:,0]=IB
    
    return IM_BB


#def ft(Ein, z=1, a=1, b=1, u=20, v=20, lbd=1, nu=40, nv=40, inv=False):
def ft(Ein, a=1, u=20, nu=40, lbd=1, z=1, inv=False):
    # This is my mostly used version of the matrix Fourier transform.
    # Ein is the input electric field
    # z is the focal length of the virtual lens that is used to go from the pupil to the image plane or vice-versa (m)
    # a and b are the physical sizes of the input array along the x and y dimensions (m)
    # u and b are the physical sizes of the output array (m)
    # lbd is the wavelength that is considered (m)
    # nu and nv are the sampling of the output array along the x and y axes (no units)
    # Important point: to work with lbd*F/D units (i.e. measure u and v in units of lbd*F/D), you should set z=1, lbd=1, and measure a,b in units of pupil diameter (if you start from a pupil plane) 
    # This works too if you start from a focal plane, and then the roles of the a and b parameters will be switched with the u and v parameters.
    nx, ny = np.shape(Ein)
    dx = a/nx
    #dy = b/ny
    dy = dx
    du = u/nu
    dv = du
    #dv = v/nv
    x = (np.linspace(-nx/2,nx/2,nx,endpoint=False)+0.5)*dx
    y = x
    #y = (np.linspace(-ny/2,ny/2,ny,endpoint=False)+0.5)*dy
    u = (np.linspace(-nu/2,nu/2,nu,endpoint=False)+0.5)*du
    v = u
    #v = (np.linspace(-nv/2,nv/2,nv,endpoint=False)+0.5)*dv
    x_u = np.outer(x,u)
    y_v = np.outer(y,v)
    
    if inv==True:
        Eout = np.dot(np.transpose(np.exp(2.*pi*1j*x_u/lbd)),Ein) 
        Eout = np.dot(Eout,np.exp(2.*pi*1j*y_v/lbd))
        Eout = -Eout*dx*dy*np.exp(2*1j*pi*z/lbd)/(1j*lbd*z)
    else:
        Eout = np.dot(np.transpose(np.exp(-2.*pi*1j*x_u/lbd)),Ein) 
        Eout = np.dot(Eout,np.exp(-2.*pi*1j*y_v/lbd))
        Eout = Eout*dx*dy*np.exp(2*1j*pi*z/lbd)/(1j*lbd*z)
    return Eout

def ft_edge(Ein, z, a, b, u, v, lbd, nu, nv):
    #this version of ft assumes that the origin is centered on a pixel, and *NOT* between 4 pixels
    nx, ny = np.shape(Ein)
    dx = a/nx
    du = u/nu
    x = (np.linspace(-nx//2,nx//2,nx,endpoint=False)+1)*dx ## was +1 before may 26 ; proved to shift PSF by 1pix ; modified to zero to correct that
    u = (np.linspace(-nu//2,nu//2,nu,endpoint=False)+1)*du
    x_u = np.outer(x,u)
    Eout = np.dot(np.transpose(np.exp(-2.*pi*1j*x_u/lbd)),Ein) 
    Eout = np.dot(Eout,np.exp(-2.*pi*1j*x_u/lbd))
    Eout = Eout*dx*dx*np.exp(2*1j*pi*z/lbd)/(1j*lbd*z)
    return Eout

def ftinv(Ein, z, a, b, u, v, lbd, nu, nv):
    nx, ny = np.shape(Ein)
    dx = a/nx
    du = u/nu
    x = (np.linspace(-nx/2,nx/2,nx,endpoint=False)+0.5)*dx
    u = (np.linspace(-nu/2,nu/2,nu,endpoint=False)+0.5)*du
    x_u = np.outer(x,u)
    Eout = np.dot(np.transpose(np.exp(2.*pi*1j*x_u/lbd)),Ein) 
    Eout = np.dot(Eout,np.exp(2.*pi*1j*x_u/lbd))
    Eout = Eout*dx*dx*np.exp(-2*1j*pi*z/lbd)/(1j*lbd*z)
    return Eout

# ! peut etre utile
def ft_3D(Ein, z, a, b, u, v, lbd, nu, nv):
    #Loads datacube and perform ft for each spectral slice (first dimension)
    #Last 2 dimensions assumed to be X,Y
    nz, nx, ny = np.shape(Ein)
    dx = a/nx
    du = u/nu
    x = (np.linspace(-nx/2,nx/2,nx,endpoint=False)+0.5)*dx
    u = (np.linspace(-nu/2,nu/2,nu,endpoint=False)+0.5)*du
    x_u = np.outer(x,u)
    u_x = np.outer(u,x)
    t0 = time.time()
    XU_3D = np.repeat(x_u[np.newaxis,:,:],nz,axis=0)
    UX_3D = np.repeat(u_x[np.newaxis,:,:],nz,axis=0)
    LBD_XU_2D = np.repeat(lbd[:,np.newaxis],nx,axis=1)
    LBD_XU_3D = np.repeat(LBD_XU_2D[:,:,np.newaxis],nu,axis=2)
    LBD_UX_2D = np.repeat(lbd[:,np.newaxis],nu,axis=1)
    LBD_UX_3D = np.repeat(LBD_UX_2D[:,:,np.newaxis],nx,axis=2)
    LBD_UU_2D = np.repeat(lbd[:,np.newaxis],nu,axis=1)
    LBD_UU_3D = np.repeat(LBD_UU_2D[:,:,np.newaxis],nu,axis=2)
    t1 = time.time()-t0
    print(t1)
    
    print(Ein.shape)
    print(LBD_UX_3D.shape)
    print(UX_3D.shape)
    
    t0 = time.time()
    Eout = np.matmul(np.exp(-2.*pi*1j*UX_3D/LBD_UX_3D),Ein)
    #Eout = np.einsum('ijk,ikl->ijl',np.exp(-2.*pi*1j*UX_3D/LBD_UX_3D),Ein,optimize=True)
    #Eout = torch.einsum('ijk,ikl->ijl',[np.exp(-2.*pi*1j*UX_3D/LBD_UX_3D),Ein])
    t1 = time.time()-t0
    print(t1)
    
    t0 = time.time()
    Eout = np.matmul(Eout,np.exp(-2.*pi*1j*XU_3D/LBD_XU_3D))
    t1 = time.time()-t0
    print(t1)
    
    t0 = time.time()
    Eout = Eout*dx*dx*np.exp(2*1j*pi*z/LBD_UU_3D)/(1j*LBD_UU_3D*z)
    t1 = time.time()-t0
    print(t1)
    
    return Eout

def ftAMPL(Ein, a, nu, u, lbd):
    nx, ny = np.shape(Ein)
    dx = a/nx
    du = u/nu
    x = (np.linspace(-nx/2,nx/2,nx,endpoint=False)+0.5)*dx
    u = (np.linspace(-nu/2,nu/2,nu,endpoint=False)+0.5)*du
    x_u = np.outer(x,u)
    Eout = np.dot(np.transpose(np.exp(-2.*pi*1j*x_u/lbd)),Ein) 
    Eout = np.dot(Eout,np.exp(-2.*pi*1j*x_u/lbd))
    Eout = Eout/lbd*dx*dx
    return Eout
    
def Fresnel(Ein, z, a, u, lbd, nu, trigger):
    if trigger==False:
        nx = len(Ein)
        dx = a/nx
        du = u/nu
        x = (np.linspace(-nx/2,nx/2,nx,endpoint=False)+0.5)*dx
        u = (np.linspace(-nu/2,nu/2,nu,endpoint=False)+0.5)*du
        X,Y = np.meshgrid(x,x,sparse=False)
        U,V = np.meshgrid(u,u,sparse=False)
        Eout = ft(Ein*np.exp(1j*pi/(lbd*z)*(X**2+Y**2)),z,a,a,u,u,lbd,nu,nu)*np.exp(1j*pi/(lbd*z)*(U**2+V**2))
    else:    
        N = len(Ein)
        four = ft_BASIC(Ein,a,N/a,N,1)
        x_int = (np.linspace(-N/2,N/2,N,endpoint=False)+0.5)*1/a 
        X_int,Y_int = np.meshgrid(x_int,x_int,sparse=False)
        angular = 1j*lbd*z*np.exp(-1j*pi*z*lbd*(X_int**2+Y_int**2))
        Eout = ft_BASIC(angular*four,N/a,u,nu,-1)
        Eout = Eout*np.exp(2*1j*pi*z/lbd)/(1j*lbd*z)
    
    return Eout

def SQ(A):
    A = np.concatenate((np.fliplr(A),A),axis=1)
    OUT = np.concatenate((np.flipud(A),A),axis=0)
    return OUT

def VLTNLS(N,alpha):

    D = 1.
    d = 0.14
    t = 0.09/18.
    Th = 50.5*pi/180.
    x , X , Y , R , T = VECT(N,D)
    a1 = alpha
    a2 = 1+(1-alpha)*D/d
    a3 = 1+(1-alpha)*D/t
    # a1 = a1/100
    # a2 = a2/100
    # a3 = a3/100
    VLT=1*(R<=D/2.*a1)*(R>=d/2.*a2)*(((Y <= (1*(X-d/2.+t/np.sin(Th)/2.)-1*(t*a3/np.sin(Th)/2.))*np.tan(Th)) + 1*(Y >= (1*(X-d/2.+t/np.sin(Th)/2.)+1*(t*a3/np.sin(Th)/2.))*np.tan(Th)))*(X>0.)*(Y>0.))
    VLT=VLT+np.fliplr(VLT)
    LS=VLT+np.flipud(VLT)
    return LS

def WN2WL(wavenumber):
    # wavenumber in cm-1
    # wavelength in m
    wavelength = 1/(wavenumber*100)
    return wavelength

def WL2WN(wavelength):
    # wavenumber in cm-1
    # wavelength in m
    wavenumber = 1/(wavelength*100)
    return wavenumber

def SNR_estimate(flux_signal,flux_noise,RON,Dark_Current,exposure_time,pix_number):
    #flux signal, flux noise, Dark_current in e-/s
    #RON in e-
    #pix number denotes the number of pixels used for the detection (used for spectral binning) 
    SNR = flux_signal*pix_number*exposure_time/np.sqrt(pix_number*exposure_time*(flux_signal+flux_noise)+pix_number*Dark_Current*exposure_time+RON**2)
    return SNR

def oneD_Gaussian(x,a,x0,sigma,offset):
    return offset+a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_1D_gaussian(x,y,guess=0):
    n = len(x)
    if guess==0:
        mean = np.sum(x*y)/n
        sigma = np.sum(y*(x-mean)**2)/n
        offset = (y[0]+y[-1])/2
        popt,pcov = curve_fit(oneD_Gaussian,x,y,p0=[1,mean,sigma,offset])
    else:
        popt,pcov = curve_fit(oneD_Gaussian,x,y,p0=guess)
    return popt

def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple 
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def twoD_Lorentzian(xdata_tuple, amplitude, xo, yo, gamma, offset):
    (x, y) = xdata_tuple 
    xo = float(xo)
    yo = float(yo)    
    # a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    # b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    # c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    # g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    g = offset + amplitude*2/(np.pi*gamma)/(1+(2/gamma)**2*((x-xo)**2+(y-yo)**2))
    return g.ravel()

def mInterpGaussian2D(data, mask, initial_guess):
    # Data is the 2D image to look into for a Gaussian
    # initial guess should be a 7D vector with amplitude, x0, y0, sig_x, sig_y, angle, offset
    # We assume a coordinate system x,y with physical sizes that equal the dimensions
    [L2,L1] = np.shape(data)
    x = np.linspace(0, L1-1, L1)
    y = np.linspace(0, L2-1, L2)
    yy,xx = np.meshgrid(x,y)
    xx = xx[mask==1]
    yy = yy[mask==1]
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    popt, pcov = curve_fit(twoD_Gaussian, xdata, np.ravel(data[mask==1]),p0=initial_guess)

    return popt

def mInterpLorentzian2D(data, mask, initial_guess, boundaries):
    # Data is the 2D image to look into for a Lorentzian
    # initial guess should be a 5D vector with amplitude, x0, y0, gamma, offset
    # We assume a coordinate system x,y with physical sizes that equal the dimensions
    [L1,L2] = np.shape(data)
    x = np.linspace(0, L1-1, L1)
    y = np.linspace(0, L2-1, L2)
    xx,yy = np.meshgrid(x,y)
    xx = xx[mask==1]
    yy = yy[mask==1]
    xdata = np.vstack((xx.ravel(),yy.ravel()))
    popt, pcov = curve_fit(twoD_Lorentzian, xdata, np.ravel(data[mask==1]),p0=initial_guess,bounds=boundaries)

    return popt

#LUT=np.loadtxt(LUT_filename)
def Phase2BMP(LUT,A):
    #f=interp1d(LUT[:,1],LUT[:,0],'linear',fill_value='extrapolate')
    #A_BMP = f(A)
    A_BMP = np.interp(A,LUT[:,1],LUT[:,0])
    A_BMP = np.rint(A_BMP)
    A_BMP[A_BMP>255] = 255
    A_BMP[A_BMP<0] = 0
    return A_BMP

def PUP4AMPL(A,filename,option='quadrant'):
    #Creates .dat file with a file name 'filename', for AMPL from an array A
    #we assume that A is an NxN array, with N = 2*M (and M an integer)
    #Four options : 'full' saves the whole pupil, 'quadrant' saves the bottom right quadrant, 'half_X' saves the right half, and 'half_Y' saves the bottom half.
    #Default option is 'quadrant' (most pupil have an X/Y symmetry)
    A = 1.0*A
    f = open(filename+'.dat','w')
    N = len(A)   
    flag=0
    if option == 'full':
        for k in range(N):
            for l in range(N):
                f.write("{0}".format(A[k,l]))
                if l==N-1:
                        f.write("\n")        
                else:
                        f.write("\t")
    elif option == 'quadrant':
        for k in range(N//2):
            for l in range(N//2):
                f.write("{0}".format(A[k+N//2,l+N//2]))
                if l==N-1:
                    f.write("\n")
                else:
                    f.write("\t")
    elif option == 'half_X':
        for k in range(N):
                for l in range(N//2):
                    f.write("{0}".format(A[k,l+N//2]))
                    if l==N-1:
                        f.write("\n")
                    else:
                        f.write("\t")
    elif option == 'half_Y':
        for k in range(N//2):
                for l in range(N):
                    f.write("{0}".format(A[k+N//2,l]))
                    if l==N-1:
                        f.write("\n")
                    else:
                        f.write("\t")
    else:
        print('Error! Choose a valid option next time')
        flag = 1
    if flag == 1:
        return 1
    else:
        return 0 
    
def K_radvel(P,Mp,Ms,sin_i,e):
    #P: period of the planet
    #Mp: mass of the planet
    #Ms: mass of the star
    #sin_i : sine of the inclination
    #e: eccentricity
    G = 9,80665 #m/s**2
    K = (2*np.pi*G/P)**(1/3)*(Mp*sin_i)/(Mp+Ms)**(2/3)*1/(1-e**2)**(1/2)
    return K

def XY468():
    #Provides the X & Y coordinates of the DM468.
    N = 25
    x = np.linspace(-N//2,N//2,N)
    x /= N
    X,Y = np.meshgrid(x,x)
    R = np.sqrt(X**2+Y**2)
    C = (R<=0.507)    
    X_in = X[C==1]
    Y_in = Y[C==1]  
    return (C[1:,1:],X_in,Y_in)

def COM468_ZER(nZernike):
    #Returns (1) a command vector to induce a 1um Zernike polynomial on the DM468 and (2) a 2D version of the same vector to help visualize it
    Z2C_data = scipy.io.loadmat('/Users/carlotal/EXACT/WP1/ALPAO-DM468/Config/BAX439-Z2C.mat')
    #IF_data = scipy.io.loadmat('/Users/carlotal/EXACT/WP1/ALPAO-DM468/Config/BAX439-IF.mat')
    Z2C = Z2C_data.get('Z2C')
    Z = np.zeros(467)
    Z[nZernike] = 1
    COM = np.matmul(Z,Z2C)
    C,X,Y = XY468()
    COM2D = np.zeros((24,24))
    COM2D[C==1] = COM
    return (COM, COM2D)

def DopplerEffect(l0,vs):
    #l0: wavelength in m
    #vs: radial velocity in km/s
    c = 3e5 #in km/s
    l = l0*(c+vs)/c
    Dl = l-l0
    return (l,Dl)

def GratingEfficiency(k,lbd,gr,blaze_deg,incidence_angle_deg):
    #GratingEfficiency(k,lbd,gr,blaze_deg,incidence_angle_deg):
    #k: order
    #lbd: wavelength
    #gr: number of grooves per mm
    #blaze_deg: blaze angle in deg
    #k=3
    #lbda=np.linspace(1400,1900,1000)*1e-9
    #length=np.shape(lbda)[0]
    d=1e-3/gr
    blaze=blaze_deg*pi/180
    s=d*np.cos(blaze)
    alpha=incidence_angle_deg*pi/180

    #beta=36*pi/180
    r=0.82 #cf données Richardson

    beta=np.arcsin(k*lbd/d-np.sin(alpha))
    deltap=k*pi*s/d*(np.cos(blaze)-np.sin(blaze)/np.tan((alpha+beta)/2))
    E=r*(np.sin(deltap)/deltap)**2
    return E

def BlazedGrating(k,a,i,lbd):
    #a: period of the grating [m]
    #k: order number [integer]
    #i: incidence angle [deg]
    #lbd: wavelengh [m]
    #returns: r, reflexion angle [deg], and disp, the dispersion [mrad/nm]
    r = np.arcsin(k*lbd/a-np.sin(i*np.pi/180))
    #disp = a*k/np.cos(r)
    r_m = np.arcsin(k*(lbd-0.5*1e-9)/a-np.sin(i*np.pi/180))
    r_p = np.arcsin(k*(lbd+0.5*1e-9)/a-np.sin(i*np.pi/180))
    disp = (r_p-r_m)/1000
    r *= 180/np.pi
    return r,disp

def VortexTransform(E_in,D,L,charge=2):
    #Compute the pupil plane electric field at the exit of a vortex coronagraph with an OWA=L (in units of lambda/D)
    #E_in: input electric field in entrance pupil plane
    #L: OWA in units of lambda/D
    #vortex charge must be 2, 4, or 6 (higher charges could be computed too)
    #D: diamter of pupil window in units of pupil diameter
    
    #initializing x vector and X,Y,R matrices in pupil plane; we assume that the sampling and size along the x and y axs are the same
    N=len(E_in)
    x = VECT1(N,D)
    X,Y = np.meshgrid(x,x)
    R = np.sqrt(X**2+Y**2)
    PHI = np.arctan2(Y,X)
    PLR = np.pi*L*R
    J_zero = j0(2*PLR)
    J_one = j1(2*PLR)
    F=np.zeros((N,N))
    
    #Computing the F term (as in Carlotti et al. 2014)
    if charge==2:
        F=np.exp(2*1j*PHI)/(np.pi*R**2)*(-1+J_zero+PLR*J_one)
    elif charge==4:
        F=np.exp(4*1j*PHI)/(np.pi*R**2)*(2+4*J_zero+(PLR-6/PLR)*J_one)
    elif charge==6:
        F=np.exp(6*1j*PHI)/(np.pi*R**2)*(-3+(9-60/PLR**2)*J_zero+(PLR-36/PLR+60/PLR**3)*J_one)
    else:
        print('charge should be 2,4, or 6. Switching to charge 2 by default')
        F=np.exp(2*1j*PHI)/(np.pi*R**2)*(-1+J_zero+PLR*J_one)
    
    #Computing the output pupil
    E_out = convolve2d(E_in,F,mode='same')*(D/N)**2
    # E_out = convolve(E_in,F)*(D/N)**2
    return E_out
    
def hex2bin(hex_string,length_of_bin=8):
    # Convert the hexadecimal string to an integer using the base 16
    # length of bin is the minimum number of binary characters that the binary number should have ; default value is 8 
    bin_string = np.base_repr(int(hex_string, 16), base=2)  
    bin_string = bin_string.zfill(length_of_bin) 
    return bin_string

def bin2hex(bin_string,length_of_hex=2):
    # Convert the binary string to an integer using the base 2
    # length of hex is the minimum number of hex characters that the hex number should have ; default value is 2 
    hex_string = np.base_repr(int(bin_string, 2), base=16)  
    hex_string = hex_string.zfill(length_of_hex)    
    return hex_string

def bin2volt(bin_string):
    first_10 = bin_string[-10:]
    eleven = bin_string[-11]
    voltage = 1.3*float(eleven)+2.0*int(first_10,2)/1000
    return voltage

def ConvertHex_ASIC(hex_string):
    bin_string = hex2bin(hex_string).zfill(16)
    voltage = bin2volt(bin_string)
    print('Bias voltage: {}V'.format(voltage))
    if bin_string[-12] == '0':
        print('DAC power up')
    else:
        print('DAC power down')
    if bin_string[-14:-12] == '00':
        print('Lowest cap compensation for small capacitive load')
    elif bin_string[-14:-12] == '11':
        print('Highest cap compensation for large capacitive load')
    else:
        print('Error')
    if bin_string[-15] == '0':
        print('0')
    else:
        print('1')
    if bin_string[-16] == '0':
        print('Current source power up')
    else:
        print('Current source power down')
        
def UpdateHex_ASIC(hex_string,new_voltage):
    #hex_string: a 4-characters hexadecimal string
    #new_voltage: the desired voltage, in V
    ##
    #bin_string[-10:]: last ten digits of the binary string, i.e., first 10 bits, code for voltage (0.0-2.046V)
    #bin_string[-11]: 11th bit, codes for DAC range offset (1 = +1.3V, 0: +0V)
    #bin_string[-12]: 12th bit, codes for DAC Power message
    #bin_string[-14:-12]: 13th and 14th bits, code for Capacitor compensation message
    #bin_string[-15]: 15th bit, codes for Sink/Source message
    #bin_string[-16]: 16th bit, codes for Current Power message
    bin_string = hex2bin(hex_string).zfill(16)
    if new_voltage < 1.3:
        new_last_ten = np.base_repr(int(new_voltage*1000/2),base=2).zfill(10)
        bin_string = bin_string[-16:-11] + '0' + new_last_ten
    elif new_voltage > 2.046:
        new_last_ten = np.base_repr(int((new_voltage-1.3)*1000/2),base=2).zfill(10)
        bin_string = bin_string[-16:-11] + '1' + new_last_ten
    else:
        sol_1 = int(new_voltage*1000/2)*2/1000
        sol_2 = int((new_voltage-1.3)*1000/2)*2/1000+1.3
        if np.abs(sol_1-new_voltage) < np.abs(sol_2-new_voltage):
            new_last_ten = np.base_repr(int(new_voltage*1000/2),base=2).zfill(10)
            bin_string = bin_string[-16:-11] + '0' + new_last_ten 
        else:
            new_last_ten = np.base_repr(int((new_voltage-1.3)*1000/2),base=2).zfill(10)
            bin_string = bin_string[-16:-11] + '1' + new_last_ten    
    new_hex_string = np.base_repr(int(bin_string,2),base=16)
    return new_hex_string

def Phase2Sep(max_elongation,flux_ratio):
    #computes the elongation to get the desired flux_ratio (1: 100% phase, 0.5: planet half illuminated)
    cos_phi = np.arccos(2*flux_ratio-1)
    elongation = max_elongation*(np.cos(np.pi/2-cos_phi))
    return elongation

def H2RG_refcorr(Im):
    #Takes in an H2RG fits image and uses the reference pixels to correct for the spatial variation of the RON to lower it
    # First, correction of the column-to-column variations:
    Corr_column = np.zeros(32)
    Im_corr = Im.copy()
    for k in range(32):
        Corr_column[k] = (np.median(Im[0:4,k*64:(k+1)*64])+np.median(Im[2044:2048,k*64:(k+1)*64]))/2 
        Im_corr[:,k*64:(k+1)*64] -= Corr_column[k]
    # Then, correction of the vertical variations:
    vertical_offset = np.outer((np.median(Im_corr[:,0:4],axis=1)+np.median(Im_corr[:,2044:2048],axis=1))/2,np.ones(2048))
    Im_corr -= vertical_offset
    return Im_corr

def imagebin(arr, factor):
    shape = (arr.shape[0]//factor, factor,
             arr.shape[1]//factor, factor)
    return arr.reshape(shape).mean(-1).mean(1)
    

