function [coor, lat, numIons, atomType] = Read_POSCAR(filename)
%This rountine is to read crystal structure
%File: POSCAR (4.6/5.2)
%Selective dyanmics mode for surface
%Last updated by Qiang Zhu (2013/10/03)

[fid,message] = fopen(filename);
tmp = fgetl(fid); % system description
tmp = fgetl(fid); % scaling_factor
scale_factor = str2num(tmp);

%%%%%%%%%%LATTICE%%%%%%%%%%%%%
lat = zeros(3);
for i = 1 : 3
    tmp = fgetl(fid);
    lat(i,:) = str2num(tmp);
end
lat = lat*scale_factor;

%%%%%%%%%%atomType%%%%%%%%%%%%%
tmp = fgetl(fid); % atom type
c1 = findstr(tmp, ' ');
c  = sort(str2num(['0 ' num2str(c1)]));
c(end+1) = length(tmp) + 1;
ind1 = 1;
for i = 2 : length(c)
    if c(i-1)+1 > c(i)-1
        continue
    end
    tmp1 = tmp(c(i-1)+1 : c(i)-1);
    tmp1 = strtrim(tmp1);
    for j = 1 : 105
        if strcmp(lower(tmp1), lower(megaDoof(j)))
            atomType(ind1) = j;
            break;
        end
    end
    ind1 = ind1 + 1;
end

%%%%%%%%%%numIons%%%%%%%%%%%%%
tmp = fgetl(fid); % numIons
numIons = str2num(tmp);

%%%%%%%%%%coordinates%%%%%%%%%%%%%
natom = sum(numIons);
tmp_mode = fgetl(fid);
if (tmp_mode(1) == 's') | (tmp_mode(1) == 'S') % selective mode
    tmp_mode = fgetl(fid);
    sss = fscanf(fid,'%g %g %g %s %s %s',[6,natom]);
else
    sss = fscanf(fid,'%g %g %g',[3,natom]);
end
ss=sss';
coor = ss(:,1:3);

fclose(fid);

if (tmp_mode(1) == 'c') | (tmp_mode(1) == 'C') | (tmp_mode(1) == 'k') | (tmp_mode(1) == 'K') % Cartesian
    coor = coor*scale_factor;
    coor = coor/lat;
end
%% the local optimizer may not constrain the coordinates to [0,1], so we do this now.
coor = coor - floor(coor);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
