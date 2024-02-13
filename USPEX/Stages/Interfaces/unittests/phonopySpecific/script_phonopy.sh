#!/bin/bash
source /ccc/cont003/home/icmpe/baturinv/.local/envs/dftenv/bin/activate

#dim=$(./get_supercell_for_phonopy.py)
phonopy -d --dim="%DIM" POSCAR

for k in `/bin/ls POSCAR-*`;do    
 dispname=`echo $k|sed s/POSCAR/disp/`;    
 mkdir $dispname;    
 mv $k $dispname/POSCAR;    
 cp ./myrun $dispname;
 cp ./INCAR $dispname;
 cp ./POTCAR $dispname;
done;

for k in `/bin/ls -d disp-*`;do    
 cd $k;   
 ccc_msub myrun;    
 cd ..;  
done;

./correct_atomnames_dim_in_conf.sh

#!/bin/bash
source /ccc/cont003/home/icmpe/baturinv/.local/envs/dftenv/bin/activate

phonopy -f disp-*/vasprun.xml
phonopy -p -s band.conf
phonopy -p -s mesh.conf # Q-mesh
phonopy -p -s -t mesh.conf
