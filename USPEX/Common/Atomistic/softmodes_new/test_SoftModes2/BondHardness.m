function [bond_in, type, R_val] = BondHardness(lat, coor, numIons, ...
    atomType, goodBond, dimension)

% calculates Bonds which make contribution to hardness
% INPUT:

% from POSCAR
% lattice
% coordinates
% composition
% atomType

% from empirical database (should be defined by user, otherwise use default)
% R_val - covalent radii,
% N_val - number of valence electrons,
% val   - valence

% OUTPUT
% B: HardBond Matrix (N*7 array) as follows
% atom i, atom j, delta, distance, bond_type, [i, j, k] periodic vector

% Algorithm
% 1, calculate all atomic pairs within a distance range [max_bond]
% 2, group all bonds within a distance range [same_bond]
% 3, consider all short bonds within a range [small_bond],
% 4, if the atoms are not fully connected in 3D, add long bonds
% 5, repeat 4 until 3D connectivity is satisfied

%The following parameters defines how to evaluate the bond by distance
same_bond  = 0.05;  % same bond within this distance
max_bond  = 5;     % maximum distance deviation for bonds search,
small_bond = -0.37*log(goodBond); % maximum deviation to consider a bond

%Generate R_val and type for each atom
N_atom = sum(numIons);
N_type = length(numIons);
R_val   = zeros(N_type,1);
count  = 0;
for i = 1:N_type
    type(count+1:count+numIons(i))=i;
    count = count + numIons(i);
    R_val(i) = str2num(covalentRadius(atomType(i)));
end

%1, Calculate Bonds within upper bound to max_bond
%2, Group bond by using same_bond criterion
bond_total = MaxBonds(lat, coor, numIons, atomType, ...
    max_bond, same_bond, dimension);
bond_group = unique(round(bond_total(:,4)));
bond_group_ref = bond_group;

%3, add bonds by group,
bond_in = [];

%4, first to consider all the short bonds
bond_loop = 1;
while bond_loop < length(bond_group_ref)
    ID = find( round(bond_total(:,4)) == bond_group_ref(bond_loop) );
    
    if ~isempty(ID)
        a = type(bond_total(ID(1),1));
        b = type(bond_total(ID(1),2));
        if bond_total(ID(1),3) < small_bond(a,b)
            %disp(['ADD-----' num2str(bond_loop)])
            %bond_total(ID,:)
            bond_in = [bond_in; bond_total(ID,:)];    %Add
            bond_total(ID,:) = [];                    %Remove
            bond_group(1)    = [];                    %Remove
        end
    end
    bond_loop = bond_loop + 1;
end

%5, check 3D connectivity, if not satisfied, add more bonds
%   but we only include those bonds which could increase connectivity
%   otherwise, we won't add them
List = ConnectList(bond_in(:,1:2));
while length(List) < N_atom
    if isempty(bond_total)
        %disp('Running out of all bonds, has to exit');
        break
    else
        %disp('The stuture is not fully connected, adding more bonds');
        ID = find(abs(bond_total(:,4)-bond_group(1)) < 0.1);
        bond_tmp = [bond_in; bond_total(ID,:)];
        List_new = ConnectList(bond_tmp(:,1:2));
        bond_total(ID,:) = [];
        bond_group(1) = [];
        if length(List_new) > length(List) %increase connectivity accept
            %disp('The connectivity is increased, accept adding more bonds');
            bond_in = bond_tmp;
            List = List_new;
        else
            %disp('The connectivity is not increased, reject adding more bonds');
        end
    end
end

%6, Remove double count of bond like [i,i] pair;
ID = find(bond_in(:,1)==bond_in(:,2));
list = ID(1:2:length(ID));
bond_in(list,:) = [];

%---------------------------------------------------------------------
