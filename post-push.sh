#!/bin/bash

# Remote servers

#SERVER0="crivellc@irene-amd-fr.ccc.cea.fr:/ccc/work/cont003/gen6175/crivellc/USPEX/USPEX_TGCC-TGCC_re"
#SERVER1="baturinv@irene-fr.ccc.cea.fr:/ccc/work/cont003/gen6175/baturinv/USPEX/Common_DB"
SERVER2="baturinv@irene-fr.ccc.cea.fr:/ccc/work/cont003/gen6175/baturinv/USPEX/Common_v2"

# Hardcoded passwords
PASSWORD_VB="Roz@Luxemburg669"
#PASSWORD2="qdkgov38!Gfd"

# Function to synchronize the current directory with remote servers using rsync
synchronize_with_servers() {
  echo "Synchronizing current directory with remote servers..."

  # Synchronize with remote servers using rsync and sshpass
  sshpass -p "$PASSWORD_VB" rsync --progress --recursive --compress --exclude="unittests" --exclude="testFolder" --exclude=".git" --exclude="uspex.egg-info/" --exclude="venv/" --exclude=".idea/" --exclude="post-push.sh" ./ "$SERVER2"
#  sshpass -p "$PASSWORD2" rsync --progress --recursive --compress --exclude=".git" --exclude="uspex.egg-info/" --exclude="venv/" --exclude=".idea/" --exclude="post-push.sh" ./ "$SERVER2"

  echo "Current directory synchronized with remote servers successfully."
}

# Execute the synchronization function
synchronize_with_servers
