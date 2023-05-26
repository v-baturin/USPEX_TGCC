echo "" >> subm.out
echo "$(date '+%Y-%m-%d %H:%M:%S')" >> subm.out
ccc_msub uspexjob.sh >> subm.out 2>&1 &

