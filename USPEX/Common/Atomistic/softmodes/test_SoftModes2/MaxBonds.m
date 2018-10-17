function bonds = MaxBonds(lat, coor, numIons, atomType,...
    max_bond, same_bond, dimension)

% This program is to
% 1, obtain all bonds for a given structures
% 2, sort the bonds by distance
% 3, assign bond types

%----------------INPUT ----------------
% lattice, coordinates of a given crystal structure;
% numIons: 4 4
% atomType: [12 8] (Mg O)

%---------------OUTPUT ----------------
% bonds_sorted
% [atom_i, atom_j, distance, type , 1*3 vectcor];

coor = coor - floor(coor);   %scale it the [0 1]
N_atom = sum(numIons);
Rmax = max_bond;
N_type = length(numIons);
Rval  = zeros(N_type,1);
count = 0;
for i = 1:N_type
    type(count+1:count+numIons(i))=i;
    count = count + numIons(i);
    Rval(i) = str2num(covalentRadius(atomType(i)));
end
%------    Search for all atomic pairs  --------------%
%Here we count the coordination of each atom one by one
%Firstly we construct the possible direction matrix
%such as [1 0 0; 0 1 0; 0 0 1; .......
%Ideally, the best way to is to do lattice optimization first
%But we don't consider it for now

%---This is the direction matrix
%---maximally consider the direction of [1 2 0],
Matrix = SuperMatrix(-1,1,-1,1,-1,1);
ToDelete = all(Matrix==0, 2);
Matrix(ToDelete,:) =[]; %[0 0 0]
Target = Matrix;

%-For each atom, we plot a sphere with Rmax as the radius and calculate
%-the possible super cell size
%-1. Here we search for the optimum supercell for each atom
%-   This way we don't need to make too big supercells
%-   It can drastically reduce the cost when N is more then 100
%-2. vectorize the part of distance matrix
bonds = [];
type1 = type;
coor1 = coor;
N_atom1 = N_atom;
for i = 1:N_atom
    
    %--Build the super cell matrix
    for j = 1:size(Matrix, 1)
        Target(j,:) = Rmax/norm(Matrix(j,:)*lat)*Matrix(j,:) + coor(i,:);
    end
    lenX1 = floor(min(Target(:,1)));
    lenY1 = floor(min(Target(:,2)));
    lenZ1 = floor(min(Target(:,3)));
    lenX2 = floor(max(Target(:,1)));
    lenY2 = floor(max(Target(:,2)));
    lenZ2 = floor(max(Target(:,3)));
    if dimension == 0
        [lenX1, lenX2, lenY1, lenY2, lenZ1, lenZ2]=[0 0 0 0 0 0];
    elseif dimension == 2
        [lenZ1, lenZ2] = [0 0];
    end
    Matrix_tmp = SuperMatrix(lenX1, lenX2, lenY1, lenY2, lenZ1, lenZ2);
    
    %--Obtain the distances by vectorization, pdist2 is used here
    N_Matrix  = size(Matrix_tmp, 1);
    tmp_type  = repmat(type1', N_Matrix, 1);   %using type1
    tmp_Rval  = Rval(tmp_type)+Rval(type(i));
    tmp_ID    = repmat([i:N_atom]', N_Matrix, 1); %Index be CAREFUL
    S_coor    = repmat(coor1, N_Matrix, 1);    %using coor1
    S_Matrix  = reshape(repmat(Matrix_tmp, 1, N_atom1)', 3, N_atom1*N_Matrix)';
    
    tmp_dist  = (pdist2(coor(i,:)*lat, (S_coor + S_Matrix)*lat))';
    To_Delete = find(abs(tmp_dist-tmp_Rval)>Rmax | tmp_dist<0.5);
    tmp_dist  = tmp_dist - tmp_Rval;
    %This can be done much easier in python !!!!
    N_total   = N_Matrix*N_atom1;  %total size of tmp_dist: N_Matrix*
    tmp2 = [i*ones(N_total,1), tmp_ID, tmp_dist, zeros(N_total,1), S_Matrix];
    tmp2(To_Delete,:) = [];
    bonds = [bonds; tmp2];
    % discard the reference atom from now, to avoid double count of [i,j] pair
    coor1(1,:) = [];
    type1(1) = [];
    N_atom1 = N_atom1-1;
end

%bonds: 1, atom-i; 2, atom-j; 3, dist; 4, bond_type
%Now we assign the bond type
[tmp, rank] = sort(bonds(:,3));   %sort the bonds by dist from low to high
bonds = bonds(rank,:);
N_bonds = size(bonds,1);
bond_type = 0;
for i =  1: N_bonds
    if bonds(i,4) == 0
        bond_type = bond_type + 1;
        %obtain all bonds with the same type by distance
        ID = find(bonds(:,3)<bonds(i,3)+same_bond);
        for j = 1:length(ID)
            if bonds(ID(j),4)==0
                if isequal( type(bonds(i,1:2)), type(bonds(ID(j),1:2)) )
                    bonds(ID(j),4) = bond_type;
                end
            end
        end
    end
end
