#!/bin/bash
SERVER2="irene_vb_hydra:/ccc/work/cont003/gen6175/baturinv/USPEX/Common"

# Extract the server hostname for the prompt
SERVER_NAME=$(echo "$SERVER2" | cut -d'@' -f2 | cut -d':' -f1)

# Prompt the user for the password with the server name included
read -sp "Enter password for $SERVER_NAME: " PASSWORD_VB
echo

# Function to synchronize the current directory with remote servers using rsync
synchronize_with_servers() {
  echo "Synchronizing current directory with remote servers..."
  sshpass -p "$PASSWORD_VB" rsync --progress --recursive --compress --exclude="unittests" --exclude="testFolder" --exclude=".git" --exclude="uspex.egg-info/" --exclude="venv/" --exclude=".idea/" --exclude="post-push.sh" ./ "$SERVER2"
#  echo "Current directory synchronized with remote servers successfully."
}
synchronize_with_servers
