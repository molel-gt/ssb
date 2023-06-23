#!/bin/bash

sudo apt update

sudo apt install -y  slurmctld slurmdb slurmdbb
sudo apt install -y munge
sudo apt install -y mariadb-server

sudo systemctl restart slurmctld
sudo systemctl restart slurmd
sudo systemctl restart slurmdbd
sudo systemctl restart mariadb
sudo systemctl restart munge

sudo slurmdbd -Dvv
sudo sacctmgr create account gts-tf74 cluster=localcluster
sudo sacctmgr create user slurm account=gts-tf74

sudo sacctmgr add qos inferno MaxTRESMins="cpu=15897600" Flags=DenyOnLimit,NoDecay,UsageFactorSafe MaxTRESPU="cpu=6000" MaxTRESPerNode=250000
sudo sacctmgr add user slurm Account=gts-tf74 Partitions=High