#!/bin/bash
# vim:sw=4:ts=4:et

# set -e

# alias start_docker_shell='docker exec -it main-bash-1 sh'

cat << EOF
Running Docker Compose with 2 containers: 
   - FastAPI Server "main-web-1"
   - bash shell "main-bash-1"
EOF

# ./collect_transcripts.py
# creates a static index page to reference recordings and transcripts 
# in directories ./audio and ./text
set -x
python3 ./collect_transcripts.py
docker compose up -d --build
docker ps
set +x

echo "*** Docker Startup Complete"
echo -e "\n-------------------------------------\n"

cat << EOF
*** The Web Server can be reached at http://localhost:8000

*** To attach to the FastAPI console, run the command:
    $ docker attach main-web-1

*** To view the FastAPI logs, run the command:
    $ docker compose logs web

*** To attach to the Bash Shell run the command:
    $ docker attach main-bash-1

*** To start another shell session, run the command:
    $ docker exec -it main-bash-1 sh

*** To close all docker containers, run cmd:
    $ docker compose down

EOF

