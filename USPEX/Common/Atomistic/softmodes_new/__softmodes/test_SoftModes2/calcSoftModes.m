function [freq, eigvector] = calcSoftModes(bond_in, type, R_val, ...
    N_val, val, lat, coords, kVector0)

%Calculate vibrational modes base on the dynamic matrix (D)
%constructed from bond hardness model

%INPUT:
%bond_in: all bonds information from BondHardness.m
%lattice
%coordinates, type, covalent radii(R_val)
%valence, valence electrons
%kVector

%OUTPUT:
%freq/eigenvector of all modes

% K-vector should be in A^-1, very important!
% reciprocal_lat_x = 2pi*(lat_y X lat_z)/V;
% k_abs = k*reciprocal_lattice
%OUT=[];
if nargin == 7
    kVector0 = [0 0 0];
end
rec_lat = zeros(3,3);
rec_lat(1,:) = 2*pi*cross(lat(2,:), lat(3,:))/det(lat);
rec_lat(2,:) = 2*pi*cross(lat(3,:), lat(1,:))/det(lat);
rec_lat(3,:) = 2*pi*cross(lat(1,:), lat(2,:))/det(lat);
kVector = kVector0*rec_lat;

%obtain the bond information
bond_group = unique(round(bond_in(:,4)));         %list of bond group
N_group    = length(bond_group);                  %number of bond group
N_atom = length(type);                            %number of atoms

%Initiallization of Dynamic matrix (3N*3N)
D = zeros(3*N_atom, 3*N_atom);

%Calculate bond valence using classical Brown's bond valence model.
%nu_factor should be normalized to satisfy sum rule
nu_factor = zeros(1,N_atom);
for k = 1 : N_atom
    nu_full = 0;
    for m = 1 : size(bond_in,1)
        if (bond_in(m,1) == k)
            nu_full = nu_full + exp(-1*bond_in(m,3)/0.37);
        end
        if (bond_in(m,2) == k)
            nu_full = nu_full + exp(-1*bond_in(m,3)/0.37);
        end
    end
    nu_factor(k) = val(type(k))/nu_full;
end
for bondtype = 1 : N_group
    
    ID = find(bond_in(:,4) == bond_group(bondtype));
    
    for i = 1:length(ID)
        it = ID(i);
        ID1 = bond_in(it,1);
        ID2 = bond_in(it,2);
        a = type(ID1);
        b = type(ID2);
        delta = bond_in(it,3);
        R_a = R_val(a) + delta/2;
        R_b = R_val(b) + delta/2;
        nu = exp(-delta/0.37);
        EN_a = 0.481*N_val(a)/R_a;
        EN_b = 0.481*N_val(b)/R_b;
        CN_a = val(a)/(nu*nu_factor(ID1));
        CN_b = val(b)/(nu*nu_factor(ID2));
        f_ab = 0.25*abs(EN_a - EN_b)/sqrt(EN_a*EN_b);
        X_ab = sqrt((EN_a*EN_b)/(CN_a*CN_b));
        
        vect = coords(ID1,:) - coords(ID2,:) - bond_in(it, 5:7);
        dR_ab = vect*lat;
        C = round(dR_ab/norm(dR_ab)*1000000)/1000000;
        phase_k1 = exp(1i*dot(kVector, dR_ab));
        phase_k2 = exp(1i*dot(kVector, -1*dR_ab));
        H = X_ab*exp(-2.7*f_ab);
        
        %OUT = [OUT; C, H, bond_in(it, 1:4)];
        if ID1 == ID2
            if ~isequal(round(bond_in(it,5:7)),[0 0 0])
                D = AddDynMatSelf(D, ID1, H, C, phase_k1);
            end
        else
            D = AddDynMat(D, ID1, ID2, H, C, phase_k1, phase_k2);
        end
    end
end
[eigvector, freq] = eig(D);
eigvector = real(eigvector);
freq = sort((diag(freq))');
%OUT
%D


%----------------------------------------------------
function D = AddDynMat(D, a, b, H, Cos, phase1, phase2)
C = H * [Cos(1)*Cos(1), Cos(1)*Cos(2), Cos(1)*Cos(3);
    Cos(2)*Cos(1), Cos(2)*Cos(2), Cos(2)*Cos(3);
    Cos(3)*Cos(1), Cos(3)*Cos(2), Cos(3)*Cos(3)];

Ida = [(a-1)*3+1 : 1 : a*3];
Idb = [(b-1)*3+1 : 1 : b*3];
D(Ida,Ida) = D(Ida,Ida) + C;
D(Ida,Idb) = D(Ida,Idb) - C*phase1;
D(Idb,Ida) = D(Idb,Ida) - C*phase2;
D(Idb,Idb) = D(Idb,Idb) + C;

%----------------------------------------------------
function D = AddDynMatSelf(D, a, H, Cos, phase)
C = H * [Cos(1)*Cos(1), Cos(1)*Cos(2), Cos(1)*Cos(3);
    Cos(2)*Cos(1), Cos(2)*Cos(2), Cos(2)*Cos(3);
    Cos(3)*Cos(1), Cos(3)*Cos(2), Cos(3)*Cos(3)];

Ida = [(a-1)*3+1 : 1 : a*3];
D(Ida,Ida) = D(Ida,Ida) + C*(1-phase);
