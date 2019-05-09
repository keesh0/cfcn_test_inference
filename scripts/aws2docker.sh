#!/usr/bin/env bash

#sets up and copies to Docker image

SCRIPT_NAME="$(basename ${0})"
read -r -d '' MANIFEST <<MANIFEST
manifest
*******************************************
${SCRIPT_NAME}
call: ${0} ${*}
path: ${PWD}
real_call: `readlink -m ${0}` ${*}
real_path: $(pwd -P)
user: `whoami`
date: `date`
hostname: $(hostname)
*******************************************
MANIFEST
echo "Starting ${SCRIPT_NAME}"


##################################################
#Usage
##################################################

read -r -d '' DOCS <<DOCS
\n
aws2dokcer.sh usage: $0 options

OPTIONS:
    -h            [optional]  help, Show this message
\n
DOCS


##################################################
#Bash handling
##################################################

#set -o errexit
set -o pipefail
set -o nounset

while getopts ":h" OPTION
do
    case $OPTION in
        h) logInfo "${DOCS}"; exit 0 ;;
        \?) logInfo "Invalid option: ${OPTARG}"; exit 1 ;;
        :) logInfo "Missing argument for ${OPTARG}"; exit 1 ;;
        *) logInfo "Unexpected option ${OPTARG}"; exit 1 ;;
    esac
done

#Actual meat
docker stop cascaded-fcn
docker rm $(docker ps -aq)

#may wish to change the following to GPU=1 and then update Python script to use GPU
#888 the return may not be the container id
docker_cont_id=`GPU=0 docker run -v /Users/ubuntu/data_cfcn:/data -p 8888:8888 --name=cascaded-fcn --workdir=/Cascaded-FCN -ti --privileged -d patrickchrist/cascadedfcn bash`

docker exec -it cascaded-fcn mkdir -p ${docker_cont_id}:Cascaded-FCN/scripts
docker cp ./cascaded_unet_inference.py ${docker_cont_id}:Cascaded-FCN/scripts/

docker cp ./step1_deploy.prototxt.orig ${docker_cont_id}:Cascaded-FCN/models/cascadedfcn/step1/
docker cp ./step1_deploy.prototxt ${docker_cont_id}:Cascaded-FCN/models/cascadedfcn/step1/
docker cp ./step2_deploy.prototxt.orig ${docker_cont_id}:Cascaded-FCN/models/cascadedfcn/step2/
docker cp ./step2_deploy.prototxt ${docker_cont_id}:Cascaded-FCN/models/cascadedfcn/step2/

#Also copy weights to save download time
docker cp ./step1_weights.caffemodel ${docker_cont_id}:Cascaded-FCN/models/cascadedfcn/step1/
docker cp ./step2_weights.caffemodel ${docker_cont_id}:Cascaded-FCN/models/cascadedfcn/step2/

docker exec -it cascaded-fcn bash
#now in container