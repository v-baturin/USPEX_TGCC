#!/bin/bash
#MSUB -A gen6175 # code projet
#MSUB -q skylake    # partition
#MSUB -T 600    # time in seconds (1h:3600 ; 2h:7200 ; 4h:14400 ; 6h:21600 ; 8h:28800)
#MSUB -n 8     # nombre de coeur (128 coeur = 1 noeud)
#MSUB -c 1       # nombre de CPU par coeur
#MSUB -m scratch,work,store
#MSUB -r  rec_test
#MSUB  -o  output
#MSUB  -e  error

ml purge
source $CCCWORKDIR/USPEX/uspex_venv/bin/activate
ml geos/3.9.1
python3 -m Common.USPEX.main -r
if [ ! -f USPEX_IS_DONE ]; then ./uspex-submit.sh; fi

