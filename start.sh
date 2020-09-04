#!/bin/bash
#title           :start.sh
#description     :This script will fetch the docker image of the project from dockerhub if not available locally and
#                 start the docker container. The web app can then be accessed at http://127.0.0.1:8050
#author		       : Tapan Sharma
#date            :06/09/2020
#version         :0.1
#usage		       :bash start.sh
#notes           :Install Vim/Emacs to use this script.


# Exit on error
set -e

#### ASSUMING DOCKER ENGINE IS INSTALLED ON THE LOCAL SYSTEM WHERE THIS SCRIPT WILL BE EXECUTED.

# 1. Check if project's docker image available locally
echo "Checking local availability of project docker image."
image=`docker images | grep group07-co2 | awk '{print $1;}'`
if [ -n "$image" ]; then
	echo "Image locally available."
else
	echo "Image not available. Pulling latest image"
	docker pull t6nand/group07-co2
	echo "Docker Image for project pulled successfully............"
fi

# 2. Run docker image for the project
echo "Running docker container.............."
docker run -d -p 8050:8050 t6nand/group07-co2
echo "Waiting for web-app to start.........."

# 3. Wait for server to start and log.
until $(curl --output /dev/null --silent --head --fail http://127.0.0.1:8050); do
    printf '.'
    sleep 2
done
echo "All Done! web-app running at http://127.0.0.1:8050"