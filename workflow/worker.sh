#!/bin/bash


# get environment variables
SLR_NUM=$1
PR_NUM=$2


# final runs path
FR_PATH="../NEW_inputs/Final_Runs"


# change directory to the corresponding SLR/PR run, execute, then go back to the /Final_Run/
cd ${FR_PATH}/RunSLR${SLR_NUM}P${PR_NUM} && ./swtv4 EastDoverSWI.SEAWAT.nam && cd $FR_PATH
