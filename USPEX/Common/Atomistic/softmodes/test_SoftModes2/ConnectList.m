function List = ConnectList(bonds)

%find the connectivity list from the 1st atom in a structure
%The algorithm is as follows
%1, construcut a neighbor list for each atom in the structure
%2, from the 1st atom (layer 1), find its connected atoms (layer 2)
%3, search for the atoms in the next layer,
%4, if new atom is found, Add to the list
%5, Stop the search until all atoms are explored

%INPUT:
%all bond pairs in a structure

%OUTPUT
%List
%1       ------->  layer 1
%3, 4    ------->  layer 2
%.......
%2, 9, 8 ------->  layer n


N_atom = length(unique(bonds));
bonds = round(bonds);

%Build the neighbor list:  Pair(N_atom, 20)
%The last column is the coordination number

Pair = zeros(N_atom,20);
for i = 1:N_atom
    ToAdd = [];
    
    ID = find(bonds(:,1)==i);
    if ~isempty(ID)
        ToAdd = [ToAdd; bonds(ID,2)];
    end
    
    ID = find(bonds(:,2)==i);
    if ~isempty(ID)
        ToAdd = [ToAdd; bonds(ID,1)];
    end
    
    if ~isempty(ToAdd)
        ToAdd = unique(ToAdd);
        for j = 1:length(ToAdd)
            Pair(i,j) = ToAdd(j);
        end
        Pair(i,end) = j;
    end
end

%Start to build the connectivity map
%  1
% 3 4
%2 9 8
%.....

%Initailize the 1st layer
List = [];
List(1)     = 1;
N_atom_Prev = 0;                           %N_atom in previous layer

for i = 1: N_atom
    N_atom_Current = length(List);         %N_atom in current layer
    for j = N_atom_Prev+1:N_atom_Current
        ID = List(j);
        for k = 1:Pair(ID,end)             %check next layer
            if isempty( find(List==Pair(ID,k)) )
                List = [List; Pair(ID,k) ]; %add new atom to the list
            end
        end
    end
    
    if length(List) == N_atom_Current
        %disp('Search is complete');
        break;
    else
        N_atom_Prev = N_atom_Current;
    end
end
%debug
%List
%Pair

