function H = calcHardness(bond_in, type, R_val, N_val, val, vol)

%Calculate hardness for a given structure from bond hardness model
%see http://han.ess.sunysb.edu/hardness/)

%INPUT:
%bond_in: all bonds information from BondHardness.m
%volume
%type, covalent radii(R_val)
%valence, valence electrons

%OUTPUT:
%Hardness (GPa)


bond_group = unique(round(bond_in(:,4)));         %list of bond group
N_group    = length(bond_group);                  %number of bond group
N_atom = length(type);                            %number of atoms

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

%Apply the bond hardness model here
%two for loops here
%outer loop goes through all different bond groups,
%inner loop goes though all individual bonds in a given group
%inner loop can take arithmetic/geometric average
%out loop must use geometric average

H = 1;
for bondtype = 1 : N_group
    ID = find(bond_in(:,4) == bond_group(bondtype));
    h_tmp1 = [];
    for i = 1:length(ID)
        it = ID(i);
        a = type(bond_in(it,1));  % atom1 in the bond
        b = type(bond_in(it,2));  % atom2 in the bond
        delta = bond_in(it,3);
        R_a = R_val(a) + delta/2;
        R_b = R_val(b) + delta/2;
        nu = exp(-delta/0.37);
        EN_a = 0.481*N_val(a)/R_a;      % electronegativity
        EN_b = 0.481*N_val(b)/R_b;
        % Effective CN that describes the atomic valence
        CN_a = val(a)/(nu*nu_factor(bond_in(it,1)));
        CN_b = val(b)/(nu*nu_factor(bond_in(it,2)));
        
        f_ab = 0.25*abs(EN_a - EN_b)/sqrt(EN_a*EN_b);  % ionicity indicator
        X_ab = sqrt((EN_a*EN_b)/(CN_a*CN_b));     % electron-holding energy
        %h_tmp = h_tmp*X_ab*exp(-2.7*f_ab);
        h_tmp1 = [h_tmp1; X_ab*exp(-2.7*f_ab)];
    end
    %h_tmp = i*h_tmp^(1.0/i)  % geometric average
    %h_tmp = i*mean(h_tmp1)   % arithmetic  average
    h_tmp = i*geomean(h_tmp1); % geometric average
    
    H = H*h_tmp;
end

%final equation looks like follows
H = 423.8*N_group*(H^(1.0/N_group))/vol - 3.4;
