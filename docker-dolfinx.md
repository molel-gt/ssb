# Docker for FEniCSx
docker
 - https://docs.docker.com/engine/install/


dolfinx docker container
 - `docker run -ti dolfinx/dolfinx:stable`

check the name of the containers running with
 - `docker ps`

dolfinx docker container with local directory `/home/molel/dev/ssb` mounted to docker container directory `/root/work` (need to create the directory work first)
 - `docker run -it --volume /home/molel/dev/ssb/:/root/work dolfinx/dolfinx:stable`

run dolfinx program
 - `docker exec -it pedantic_poincare sh -c "python3 /root/work/transport.py"`<br>

where *pedantic_poincare* is docker container name, *sh* specifies we are working with bash shell, and the command passed is how we'd run the program were we in the docker container.