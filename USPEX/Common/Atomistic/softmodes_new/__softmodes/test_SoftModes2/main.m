for i = 1:5 %1:4 %0
    filename = ['POSCAR_' num2str(i)];
    disp('');
    disp(['------------------------ ' filename ' -------------------------']);
    disp('-----------------------------------------------------------');
    tic
    [COORDINATES, LATTICE, numIons, atomType] = Read_POSCAR(filename);
    goodBonds = calcDefaultGoodBonds(atomType);
    val   = zeros(length(atomType),1);
    N_val = zeros(length(atomType),1);
    for j = 1:length(atomType)
        val(j) = str2num(valence(atomType(j)));
        N_val(j) = str2num(valenceElectronsNumber(atomType(j)));
    end
    
    for j = 1:2  % makesupercell
        [COORDINATES, LATTICE, Size] = Make_SuperCell(COORDINATES, LATTICE, j, 1);
        numIons = Size*numIons;
        
        [B, type, R_val] = BondHardness(LATTICE, COORDINATES, numIons, ...
            atomType, goodBonds, 3);
        H = calcHardness(B, type, R_val, N_val, val, det(LATTICE));
        
        disp('-----------------------------------------------------------');
        disp('-----------------------------------------------------------');
        disp(['Total N of Atoms : '  num2str(sum(numIons)) ]);
        disp(['Hardness         : '  num2str(H) ' GPa']);
        
        tic
        [freq1, eig1] = calcSoftModes(B, type, R_val, N_val, val, ...
            LATTICE, COORDINATES);
        disp(['Time elapsed for new code  : '  num2str(toc) ' seconds' ]);
        
        tic
        [freq2, eig2] = calcSoftModes_old(N_val, val, numIons, ...
            LATTICE, COORDINATES, atomType, goodBonds);
        disp(['Time elapsed for old code1 : '  num2str(toc) ' seconds' ]);
        
        tic
        [freq3, eig3] = calcSoftModes_K(N_val, val, numIons, ...
            LATTICE, COORDINATES, atomType, goodBonds, [0 0 0]);
        disp(['Time elapsed for old code2 : '  num2str(toc) ' seconds' ]);
        
        
        diff1 = norm(freq1 - freq3);
        
        disp(['freq1: ' num2str(norm(freq1)) ' <===> freq2: ' num2str(norm(freq2)) '  | diff: ' num2str(diff1)]);
        
        if diff1 > 0.01
            disp('Warning, results are inconsistent, STOP')
            % quit
        end
        
        disp('-----------------------------------------------------------');
        disp('-----------------------------------------------------------');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
