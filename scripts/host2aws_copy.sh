#!/usr/bin/env bash

#copy files to an AWS EC2 instance
#should be run from cfcn script dirs

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
host2aws_copy.sh usage: $0 options

OPTIONS:
    -h            [optional]  help, Show this message
    -a <path>     [required]  AWS EC2 instance prefix  i.e. ec2-34-207-189-19
\n
DOCS


##################################################
#Bash handling
##################################################

set -o errexit
set -o pipefail
set -o nounset

AWS_EC2_PREFIX="not set"
while getopts ":ha:" OPTION
do
    case $OPTION in
        h) echo "${DOCS}"; exit 0 ;;
        a) AWS_EC2_PREFIX="${OPTARG}" ;;
        \?) echo "Invalid option: ${OPTARG}"; exit 1 ;;
        :) echo "Missing argument for ${OPTARG}"; exit 1 ;;
        *) echo "Unexpected option ${OPTARG}"; exit 1 ;;
    esac
done

#Mac to AWS
#copy over fresh python script
scp -i "~/.ssh/eric-key.pem" cascaded_unet_inference.py ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/cascaded_unet_inference.py  #from mac to AWS
scp -i "~/.ssh/eric-key.pem" test_cascaded_unet_inference.py ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/test_cascaded_unet_inference.py  #from mac to AWS
scp -i "~/.ssh/eric-key.pem" aws2docker.sh ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/aws2docker.sh  #from mac to AWS
scp -i "~/.ssh/eric-key.pem" copy_results_from_docker_to_host.sh ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/copy_results_from_docker_to_host.sh  #from mac to AWS

#copy over original and CPU-forced models
scp -i "~/.ssh/eric-key.pem" ../models/cascadedfcn/step1/step1_deploy.prototxt.orig ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/step1_deploy.prototxt.orig
scp -i "~/.ssh/eric-key.pem" ../models/cascadedfcn/step1/step1_deploy.prototxt ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/step1_deploy.prototxt
scp -i "~/.ssh/eric-key.pem" ../models/cascadedfcn/step2/step2_deploy.prototxt.orig ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/step2_deploy.prototxt.orig
scp -i "~/.ssh/eric-key.pem" ../models/cascadedfcn/step2/step2_deploy.prototxt ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/step2_deploy.prototxt

#Also copy weights to save download time
scp -i "~/.ssh/eric-key.pem" ../models/cascadedfcn/step1/step1_weights.caffemodel ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/step1_weights.caffemodel
scp -i "~/.ssh/eric-key.pem" ../models/cascadedfcn/step2/step2_weights.caffemodel ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com:~/step2_weights.caffemodel
