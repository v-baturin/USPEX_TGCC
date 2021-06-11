#!/bin/sh

JobLog=./USPEX_JOB_LOG   #output the script info(PATH and ID)
DoneTag=./USPEX_IS_DONE  #Stop the script when USPEX is finished
ErrorTag=./still_running #Stop the script when USPEX is Error
let Interval=30             #Time interval to run the script

now="$(date +'%H:%M %d/%m/%Y')"
SCRIPT=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPT")
printf "%-30s %12s %6d(PID) at %16s  per %3d seconds\n" "$SCRIPTDIR"  "$0"   "$BASHPID"  "$now" "$Interval" >> $JobLog

while [ ! -f $DoneTag ]; do
#   if [ -f $ErrorTag ];
#   then
#       printf "%-30s %12s at %16s  ERROR\n" "$SCRIPTDIR"  "$0"  "$now"  >> $JobLog 
#       break;
#   else
       date >> log
       USPEX -r  >> log
       sleep $Interval
#   fi
   if [ -f $DoneTag ];
   then
       printf "%-30s %12s at %16s  DONE\n" "$SCRIPTDIR"  "$0"  "$now"  >> $JobLog
   fi
done
