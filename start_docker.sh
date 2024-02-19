#!/bin/bash
# vim:sw=4:ts=4:et

# set -e

alias start_docker_shell='docker exec -it main-bash-1 sh'

cat << EOF
Running Docker Compose with 2 containers: 
   - FastAPI Server "main-web-1"
   - bash shell "main-bash-1"
EOF

set -x
docker compose up -d --build
docker ps
set +x

echo "*** Docker Startup Complete"
echo -e "\n-------------------------------------\n"

cat << EOF
*** The Web Server can be reach at http://localhost:8000
*** The Bash Shell can be reached by running the command:
    $ docker exec -it main-bash-1 sh
          or use the alias
    $ start_docker_shell

*** To view the FastAPI logs, run the command:
    $ docker compose logs web

*** To close docker, run cmd:
    $ docker compose down
EOF

