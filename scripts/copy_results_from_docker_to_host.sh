#!/usr/bin/env bash

#copies from Docker image to AWS instance to local host

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
run from AWS instance
OPTIONS:
    -a <path>     [required]  AWS EC2 instance prefix  i.e. ec2-34-207-189-19
    -d            [required]  docker container id
    -h            [optional]  help, Show this message
\n
DOCS


##################################################
#Bash handling
##################################################

#set -o errexit
set -o pipefail
set -o nounset

AWS_EC2_PREFIX="not set"
DOCKER_CONT_ID="not set"
while getopts ":ha:d:" OPTION
do
    case $OPTION in
        a) AWS_EC2_PREFIX="${OPTARG}" ;;
        d) DOCKER_CONT_ID="${OPTARG}" ;;
        h) echo "${DOCS}"; exit 0 ;;
        \?) echo "Invalid option: ${OPTARG}"; exit 1 ;;
        :) echo "Missing argument for ${OPTARG}"; exit 1 ;;
        *) echo "Unexpected option ${OPTARG}"; exit 1 ;;
    esac
done

#Actual meat
#copy from docker to AWS
docker cp ${docker_cont_id}:Cascaded-FCN/models/cascadedfcn/scripts/results ./results

#copy from AWS to local host
#WIN: "/cygdrive/C/Users/eric/.ssh/eric-win-key.pem"
#MAC: "~/.ssh/eric-key.pem"
scp -i "/cygdrive/C/Users/eric/.ssh/eric-win-key.pem" -r ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/results ./results/
