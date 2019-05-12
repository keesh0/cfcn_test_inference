#!/usr/bin/bash

set -x

#sets ups an AWS EC2 instance

##################################################
#Usage
##################################################

read -r -d '' DOCS <<DOCS
host2aws_ssh.sh usage: $0 options

OPTIONS:
    -h            [optional]  help, Show this message
    -a <path>     [required]  AWS EC2 instance prefix  i.e. ec2-34-207-189-19
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

#the meat
#Mac
#ssh -i ~/.ssh/eric-key.pem ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com
#Win
ssh -i  /cygdrive/C/Users/eric/.ssh/eric-win-key.pem ubuntu@${AWS_EC2_PREFIX}.compute-1.amazonaws.com
