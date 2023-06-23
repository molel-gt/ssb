#!/bin/bash

sudo systemctl restart slurmctld && sudo systemctl restart slurmd && sudo systemctl restart slurmdbd && sudo systemctl restart munge && sudo systemctl restart mariadb && sudo slurmdbd -Dvv

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
sudo sacctmgr modify cluster localcluster set QOS=inferno

# banking
sudo apt-get install libswitch-perl

cd ~/dev
git clone https://github.com/jcftang/slurm-bank.git
cd slurm-bank
make && sudo make install

sudo sbank project create -c localcluster -a gts-tf74
sudo sbank deposit -c localcluster -a gts-tf74 -t 1000
sbank balance statement -A
sbank cluster cpuhrs

sudo sacctmgr modify user slurm set GrpTRES=cpu=1000 GrpTRESRunMin=cpu=2000000
sudo sacctmgr update qos inferno set priority=10 MaxTRESPerUser=cpu=400
sudo sacctmgr -i modify user where name=slurm set DefaultQOS=inferno

sudo sacctmgr show stats
sudo sacctmgr show problem
sudo sacctmgr show RunawayJobs

sreport cluster UserUtilizationByAccount
sreport cluster AccountUtilizationByUser