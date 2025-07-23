param pi := 4*atan(1);

#param SPEC {{i in 1..4}};
#read {{i in 1..4}} SPEC[i] < INFO.txt;

#Inner working angle = angle interne, en lambda/D
param IWA := 3;  #SPEC[1];

#Outer working angle = angle externe, en lambda/D
param OWA := 8; #SPEC[2];
#Demi nombre de points dans la pupille
param N := 50; #283; #566;
param P := 100 #nombre d'écran de phase

#Echantillonage de la pupille (taille d'un pixel)
param dx := 39/38.542/(2*N);
param dy := dx;

#Transmission de l'apodiseur (max = 1) 
param T_constraint := 0.5;

#Vecteurs décrivant le quart de la pupille 
set Xs := setof {j in 0.5..N-0.5 by 1} j*dx;
set Ys := setof {j in 0.5..N-0.5 by 1} j*dy;
set Is := 0..P by 1; #vecteur allant de 0 à P par pas de 1

# Matrice pupille
param EELT {x in Xs,y in Ys};
# On remplit la matrice pupille avec les données d'un fichier externe
read {x in Xs,y in Ys} EELT[x,y] < ELT_100_Q.dat;

# Initialisation des écran de phase 
param Phase {x in Xs, y in Ys, i in Is};
read {x in Xs,y in Ys, i in Is} Phase[x,y,i] < file_phase.dat;
## Il faut créer un fichier file_phase.dat en conséquence comme un cube avec le meme nombre de pixel par contre

# On définit la pupille utile pour l'apodiseur
set Pupil := setof {x in Xs, y in Ys: EELT[x,y] >= 0.5} (x,y);

# On définit la transmission de la pupille (=surface de la pupille originale)
param PUP_TR := sum {(x,y) in Pupil} 4*EELT[x,y]*dx*dy;

# On définit la variable de transmission de l'apodiseur
var A {x in Xs, y in Ys} >= 0, <= 1, := 0.0;

# M est le nombre de point dans un quadrant du plan focal
param M := round(2*OWA);

# Echantillonnage dans le plan focal
param dxi := OWA/M; # = 1/2 ?
param deta := OWA/M;

# On définit les vecteurs xi et eta qui décrivent le plan focal
set Xis0 := setof {j in 0..M} j*dxi;
set Xis := Xis0 union {IWA,IWA+0.1,IWA+0.2,IWA+0.3,OWA-0.25,OWA};
set Etas0 := setof {j in 0..M} j*deta;       
set Etas := Etas0 union {IWA,IWA+0.1,IWA+0.2,IWA+0.3,OWA-0.25,OWA};

# On définit la forme du dark hole
set DH := setof {xi in Xis, eta in Etas: sqrt(xi^2+eta^2)>=IWA && sqrt(xi^2+eta^2) <= OWA } (xi,eta);

# C et E sont des variables, E est le champ électrique dans le plan focal, et C est une variable intermédiaire (TF suivant un seul axe)
var C {xi in Xis, y in Ys, i in Is};
var E {xi in Xis, eta in Etas, i in Is};

# Variable qui définit le champ électrique à l'origine du plan focal (E(0,0))
var E00;

# variable qui transcrit le rapport de flux maximal
var c >= 0;

# On demande au programme de minimiser la variable c (= on creuse profond)
minimize contrast: c;

# On calcule en 2 étapes le champ électrique en plan focal, avec 2 transformées de Fourier à 1 dimension à chaque fois
subject to C_def {xi in Xis, y in Ys, i in Is}: C[xi,y,i] = 2*sum {x in Xs: (x,y) in Pupil} A[x,y]*cos(Phase[x,y,i])*cos(2*pi*x*xi)*dx;
subject to E_def {xi in Xis, eta in Etas, i in Is}: E[xi,eta,i] = 2*sum {y in Ys} C[xi,y,i]*cos(Phase[0,y,i])*cos(2*pi*y*eta)*dy;

# pas très sûre là
# On calcule le E(0,0)
subject to E00_def : E00 = sum {(x,y) in Pupil, i in Is} 4*A[x,y]*cos(Phase[x,y,i])*dx*dy;

# On contraint le champ électrique au sein du dark hole pour qu'il soit plus petit qu'une certaine quantité qui dépend de la position dans le dark hole
subject to sidelobe_DZ_pos {(xi,eta) in DH, i in Is}: E[xi,eta,i] <= c;

# On contraint la transmission de l'apodiseur
subject to TR_MIN: E00/PUP_TR >= T_constraint*0.99;
subject to TR_MAX: E00/PUP_TR <= T_constraint*1.01;

# On appelle l'algo de résolution pour résoudre itérativement le problème
option solver gurobi;                                                  
option gurobi_options "tech:outlev=1 pre:solve=0 alg:method=2 bar:crossover=0 bar:homog=1 bar:iterlim=100";      
solve;

# On sauvergarde le résultat
printf {x in Xs, y in Ys}: "%10f \{"n"}", A[x,y] > "APOD_MTF.dat";