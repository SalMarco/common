#!/bin/sh 

export GENDER1=$(aws ec2 describe-instances --filter Name=tag:Name,Values=gender-dev-NOCF | grep "PublicDnsName" | head -1 | awk -F'"' '{print $4}')
export GENDER2=$(aws ec2 describe-instances --filter Name=tag:Name,Values=Gender_DLUbuntuAWS | grep "PublicDnsName" | head -1 | awk -F'"' '{print $4}')
export FORSEO=$(aws ec2 describe-instances --filter Name=tag:Name,Values=FORSEo-prod-allin1 | grep "PublicDnsName" | head -1 | awk -F'"' '{print $4}')
export EPRICE=$(aws ec2 describe-instances --filter Name=tag:Name,Values=ePrice4Krux_TMP | grep "PublicDnsName" | head -1 | awk -F'"' '{print $4}')
export GENDER3=$(aws ec2 describe-instances --filter Name=tag:Name,Values=Gender_DLUbuntuAWS_2 | grep "PublicDnsName" | head -1 | awk -F'"' '{print $4}')
export ANTIPLAGIO=$(aws ec2 describe-instances --filter Name=tag:Name,Values=antiplagio-dev | grep "PublicDnsName" | head -1 | awk -F'"' '{print $4}')
