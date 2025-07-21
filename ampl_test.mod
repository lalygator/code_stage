param pi := 4*atan(1);

#param SPEC {{i in 1..4}};
#read {{i in 1..4}} SPEC[i] < INFO.txt;

#Inner working angle = angle interne, en lambda/D
param IWA := {rp["iwa"]};  #SPEC[1];

#Outer working angle = angle externe, en lambda/D
param OWA := 8; #SPEC[2];
#Demi nombre de points dans la pupille
param N  := 566; #283; #566;
# ce serait du coup plutot N = 50 ici 

#Echantillonage de la pupille (taille d'un pixel)
# param dx := 1132/1024/(2*N);
param dx := 39/38.542/(2*N);
param dy := dx;

#Transmission de l'apodiseur (max = 1) 
param T_constraint := {rp["t"]};

#Vecteurs décrivant le quart de la pupille 
set Xs := setof {{j in 0.5..N-0.5 by 1}} j*dx;
set Ys := setof {{j in 0.5..N-0.5 by 1}} j*dy;

# Matrice pupille
param EELT {{x in Xs,y in Ys}};
# On remplit la matrice pupille avec les données d'un fichier externe
read {{x in Xs,y in Ys}} EELT[x,y] < ELT_1132_M1=2_M4=4_ROT=5.dat;

# On définit la pupille utile pour l'apodiseur
set Pupil := setof {{x in Xs, y in Ys: EELT[x,y] >= 0.5}} (x,y);

# On définit la transmission de la pupille (=surface de la pupille originale)
param PUP_TR := sum {{(x,y) in Pupil}} 4*EELT[x,y]*dx*dy;

# On définit la variable de transmission de l'apodiseur
var A {{x in Xs, y in Ys}} >= 0, <= 1, := 0.0;


###### Paramètre à modifier pour adapter au fov des MTF ? ##################

# M est le nombre de point dans un quadrant du plan focal
param M := round(2*OWA);

# Echantillonnage dans le plan focal
param dxi := OWA/M; # = 1/2 ?
param deta := OWA/M;

# Variable de transmission dans le plan focal
var B {{xi in Xis, eta in Etas}} := 0.0;

# #############################################################################

# On définit les vecteurs xi et eta qui décrivent le plan focal
set Xis0 := setof {{j in 0..M}} j*dxi;
set Xis := Xis0 union {{IWA,IWA+0.1,IWA+0.2,IWA+0.3,OWA-0.25,OWA}};
set Etas0 := setof {{j in 0..M}} j*deta;       
set Etas := Etas0 union {{IWA,IWA+0.1,IWA+0.2,IWA+0.3,OWA-0.25,OWA}};
# j'ai enlevé les points rajouté sur 1.2*OWA comme on regardait que jusqu'a OWA


# On définit la forme du dark hole
set DH := setof {{xi in Xis, eta in Etas: sqrt(xi^2+eta^2)>=IWA && sqrt(xi^2+eta^2) <= OWA }} (xi,eta);

######################### cette région n'est plus utile si on se limite à un fov de 2*OWA et a OWA dans le quadrant
# On une deuxième région du plan focal, externe
# set GH := setof {{xi in Xis, eta in Etas: sqrt(xi^2+eta^2) > OWA && sqrt(xi^2+eta^2) <= 1.2*OWA}} (xi,eta);

# C et E sont des variables, E est le champ électrique dans le plan focal, et C est une variable intermédiaire (TF suivant un seul axe)
var C {{xi in Xis, y in Ys}};
var E {{xi in Xis, eta in Etas}};

# on définit encore une variable, qui sera l'aire de l'apodiseur
# var area = sum {{(x,y) in Pupil}} 4*A[x,y]*dx*dy;  

# Variable qui définit le champ électrique à l'origine du plan focal (E(0,0))
var E00;

# variable qui transcrit le rapport de flux maximal
var c >= 0;

# On demande au programme de minimiser la variable c (= on creuse profond)
minimize contrast: c;

# On calcule en 2 étapes le champ électrique en plan focal, avec 2 transformées de Fourier à 1 dimension à chaque fois
subject to C_def {{xi in Xis, y in Ys}}: C[xi,y] = 2*sum {{x in Xs: (x,y) in Pupil}} A[x,y]*cos(2*pi*x*xi)*dx;
subject to E_def {{xi in Xis, eta in Etas}}: E[xi,eta] = 2*sum {{y in Ys}} C[xi,y]*cos(2*pi*y*eta)*dy;

# On calcule le E(0,0)
subject to E00_def : E00 = sum {{(x,y) in Pupil}} 4*A[x,y]*dx*dy; 
# du coup c'est égal à var_area ?


# On passe le champ électrique dans le plan pupille avec une autre TF pour la multiplier à la MTF
#Echantillonage de la pupille (taille d'un pixel)
param N2 := M;
# il faut juste que ce paramètre soit adapté pour la MTF mais soit cohérent avec le M
param dx2 := dxi;
param dy2 := dx2;

#Vecteurs décrivant le quart de P2
set X2s := setof {{j in 0.5..N2-0.5 by 1}} j*dx2;
set Y2s := setof {{j in 0.5..N2-0.5 by 1}} j*dy2;

param MTF {{x2 in X2s, y2 in Y2s}};
read {{x2 in X2s, y2 in Y2s}} MTF[x2,y2] < AMPL/quad_MTF_ELT_8.dat;

# Creation des variable dans le plan P2
var C_pup {{x2 in X2s, eta in Etas}};
var E_pup {{x2 in X2s, y2 in Y2s}};
var E00_pup;

subject to C_pup_def {{x in X2s, eta in Etas}}: C_pup[x2,eta] = 2*sum {{xi in Xis: (xi,eta) in E}} B[xi,eta]*cos(2*pi*x2*xi)*dxi;
# est ce que y'a besoin de créer une variable similaire à A la variable de transmission de la pupille ?
# ici c'est la variable B, crée comme la variable de 'transmission' dans le plan focal
subject to E_pup_def {{x in X2s, y in Y2s}}: E_pup[x2,y2] = 2*sum {{eta in Etas}} C_pup[x2,eta]*cos(2*pi*y2*eta)*deta*MTF[x2,y2];
# puis le multiplier par le quadrant de MTF ici

subject to E00_pup_def : E00_pup = sum {{(xi,eta) in E_pup}} 4*B[xi,eta]*dxi*deta; 
# mais c'est utile ?

# repassage dans le plan focal
var C_foc {{xi in Xis, y2 in Y2s}};
var E_foc {{xi in Xis, eta in Etas}};

var E00_foc;

subject to C_foc_def {{xi in Xis, y2 in Y2s}}: C_foc[xi,y2] = 2*sum {{x2 in X2s: (x2,y2) in E_pup}} A[x2,y2]*cos(2*pi*x2*xi)*dx2;
subject to E_foc_def {{xi in Xis, eta in Etas}}: E_foc[xi,eta] = 2*sum {{y2 in Y2s}} C_foc[xi,y2]*cos(2*pi*y2*eta)*dy2;

subject to E00_foc_def : E00_foc = sum {{(x2, y2) in E_foc}} 4*B[x2,y2]*dx2*dy2; 

# On contraint le champ électrique au sein du dark hole pour qu'il soit plus petit qu'une certaine quantité qui dépend de la position dans le dark hole
subject to sidelobe_DZ_pos {{(xi,eta) in DH}}:  E_foc[xi,eta]^2 <= c;

# Modifier le E00 ici ?
# On contraint la transmission de l'apodiseur
subject to TR_MIN: E00_foc/PUP_TR >= T_constraint*0.99;
subject to TR_MAX: E00_foc/PUP_TR <= T_constraint*1.01;

# On appelle l'algo de résolution pour résoudre itérativement le problème
option solver gurobi;
option gurobi_options "outlev=1 presolve=0 lpmethod=2 crossover=0 iterlim=100";
solve;

# On sauvergarde le résultat
printf {{x in Xs, y in Ys}}: "%10f \{"n"}", A[x,y] > ("/Users/lalyboyer/Desktop/code_stage/AMPL/APOD_MTF.dat" );

"""