This is to rewrite and test our implementation 
on bond hardness model related calculation:
At the moment, it has been used in calculating
1, hardness
2, softmode

The idea is simple 
1, count all important bonds (BondHardness.m)
2, calculate properties (hardness/Dynamic Matrix)
3, with Dyanmic matrix, we can calculate its
eigenvector/eigenvalue to obtain all vibrational 
modes at any given K-vector

Important files:
BondHardness.m:  Collect all bonds
calcHardness.m: 
calcSoftModes.m: 

How to run:

test1, Hardness/Softmodes for gamma point
matlab <main.m > output

test22, Softmodes for gamma point
matlab <main2.m > output

updates: 
test1 should be fine now
test2 return inconsistent results for POSCAR_3/4, 
needs to check
