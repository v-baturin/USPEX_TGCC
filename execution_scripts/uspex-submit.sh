#!/bin/bash
echo "" >> subm.out
echo "$(date '+%Y-%m-%d %H:%M:%S')" >> subm.out

# Set the command to run
command1="ccc_msub uspexjob.sh >> subm.out 2>&1 &"

# Set the log file
log_file="log"


# Check if the log file exists and contains the target string
if [[ -f "$log_file" ]] && grep -q "All systems in population failed relaxation" "$log_file"; then
    echo "Last line of the log file contains 'All systems in population failed relaxation'. Exiting."
    exit 0
else
    echo "Submitting uspex" >> subm.out
    # Run command1
    $command1
fi


