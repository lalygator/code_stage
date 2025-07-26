param pi := 4*atan(1);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

param IWA := 3;                                                                                                                                                                                                                                     
param OWA := 8;
param N  := 100;                                                                                                                                                                                                                                                       
param dx := 1/(2*N);                                                                                                                                                                                                                                                  
param dy := dx;
#param c := 4;
param T_constraint := 0.5;

set Xs := setof {j in -N+0.5..N-0.5 by 1} j*dx;                                                                                                                                                                                                                       
set Ys := setof {j in -N+0.5..N-0.5 by 1} j*dy;   

set Pupil := setof {x in Xs, y in Ys: sqrt(x^2+y^2)<= 0.5 && sqrt(x^2+y^2)>=0.15} (x,y);                                                                                                                                                                                                                                        
param PUP_TR := sum {(x,y) in Pupil} dx*dy;

#wavelength
param lambda := 1.65e-6;

#Zernike definition

#param NA := 10;                         # 0 → 1 par pas de 0,1 = 10 nm
#set As := setof {j in 0..NA} j/NA;      # uniquement les défauts positifs
#param A_PHI := 100e-9;                  # amplitude max = 100 nm

param NA := 2;
set As := setof {j in -NA..NA} j/NA;
param A_PHI := 20e-9;
param PHI {(x,y) in Pupil} := 2*pi*A_PHI/lambda*sqrt(3)*(4*2*(x^2+y^2)-1); 

#Apodizer amplitude
var A {x in Xs, y in Ys} >= 0, <= 1, := 0.0;      

#Definition of focal plane sampling
param M := round(2*OWA);
           
param dxi := OWA/M;
param deta := OWA/M;                                                                                                                                                                                                                                   
set Xis0 := setof {j in -M..M} j*dxi;
set Xis := Xis0 union {-IWA,-IWA-0.25,-OWA+0.25,-OWA,IWA,IWA+0.25,OWA-0.25,OWA};
set Etas0 := setof {j in -M..M} j*deta;       
set Etas := Etas0 union {-IWA,-IWA-0.25,-OWA+0.25,-OWA,IWA,IWA+0.25,OWA-0.25,OWA};
                                   
# Definition of the high-contrast area
set DH := setof {xi in Xis, eta in Etas: sqrt(xi^2+eta^2)>=IWA && sqrt(xi^2+eta^2)<=OWA} (xi,eta);                                                                                                                                                                                                                                      
                                      
# Definition of the intermediate and focal plane electric fields                                                                                                                                             
var ACC {xi in Xis, y in Ys, az in As};
var ACS {xi in Xis, y in Ys, az in As};
var ASC {xi in Xis, y in Ys, az in As};
var ASS {xi in Xis, y in Ys, az in As};

var E_R {xi in Xis, eta in Etas, az in As};
var E_I {xi in Xis, eta in Etas, az in As};                             
                                                                                                                                                                                                          
#var area = sum {(x,y) in Pupil} 4*A[x,y]*dx*dy;  
#maximize transmission: area;

#Electric field complex amplitude at (0,0)
var E00;
#var CST {xi in Xis, eta in Etas} >= 0;
#minimize contrast: (sum {(xi,eta) in DarkHole} CST[xi,eta]*dxi*deta);

var c >= 0;
minimize contrast: c; 
                                                                                                                                                                                           
subject to ACC_def {xi in Xis, y in Ys, az in As}: ACC[xi,y,az] = sum {x in Xs: (x,y) in Pupil} A[x,y]*cos(az*PHI[x,y])*cos(2*pi*x*xi)*dx;                                                                                                                                                            
subject to ACS_def {xi in Xis, y in Ys, az in As}: ACS[xi,y,az] = sum {x in Xs: (x,y) in Pupil} A[x,y]*cos(az*PHI[x,y])*sin(2*pi*x*xi)*dx;  
subject to ASC_def {xi in Xis, y in Ys, az in As}: ASC[xi,y,az] = sum {x in Xs: (x,y) in Pupil} A[x,y]*sin(az*PHI[x,y])*cos(2*pi*x*xi)*dx;  
subject to ASS_def {xi in Xis, y in Ys, az in As}: ASS[xi,y,az] = sum {x in Xs: (x,y) in Pupil} A[x,y]*sin(az*PHI[x,y])*sin(2*pi*x*xi)*dx;  

subject to E_R_def {xi in Xis, eta in Etas, az in As}: E_R[xi,eta,az] = sum {y in Ys} ((ACC[xi,y,az]-ASS[xi,y,az])*cos(2*pi*y*eta)-(ACS[xi,y,az]+ASC[xi,y,az])*sin(2*pi*y*eta))*dy;
subject to E_I_def {xi in Xis, eta in Etas, az in As}: E_I[xi,eta,az] = sum {y in Ys} ((ACS[xi,y,az]+ASC[xi,y,az])*cos(2*pi*y*eta)+(ACC[xi,y,az]-ASS[xi,y,az])*sin(2*pi*y*eta))*dy;

subject to E00_def : E00 = sum {(x,y) in Pupil} A[x,y]*dx*dy; 
                                                                                                                                                                
subject to sidelobe_R_DZ_pos {(xi,eta) in DH, az in As}:  E_R[xi,eta,az] <= c;                                                                                         
subject to sidelobe_R_DZ_neg {(xi,eta) in DH, az in As}:  E_R[xi,eta,az] >= -c; 

subject to sidelobe_I_DZ_pos {(xi,eta) in DH, az in As}: E_I[xi,eta,az] <= c;
subject to sidelobe_I_DZ_neg {(xi,eta) in DH, az in As}: E_I[xi,eta,az] >= -c;

subject to TR_MIN: E00/PUP_TR >= T_constraint*0.99;
subject to TR_MAX: E00/PUP_TR <= T_constraint*1.01;                                                                                              

option solver gurobi;                                                  
option gurobi_options "tech:outlev=1 pre:solve=0 alg:method=2 bar:crossover=0 bar:homog=1 bar:iterlim=100";      
solve;        
                                                                                                                                                                                                                                 
printf {x in Xs, y in Ys}: "%10f \n", A[x,y] > "TEST_multiple_defocus.dat";     
