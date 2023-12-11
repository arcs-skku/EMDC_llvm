#!/bin/bash

APP=$1
OPT1=$2 # Option can be ZC(Zero-Copy), GP(GPU-Pinned), DP(Demand-paging)
PER1=$3 # Percentage can be 0%, 10%, 20%, ..., 100%
OPT2=$4
PER2=$5

if [[ $APP == "2mm" ]]; then
    nsys nvprof ./2mm $OPT1 $PER1 $OPT2 $PER2
elif [[ $APP == "2dconv" ]]; then
    nsys nvprof ./2dconv $OPT1 $PER1 $OPT2 $PER2
elif [[ $APP == "3mm" ]]; then
    nsys nvprof ./3mm $OPT1 $PER1 $OPT2 $PER2
fi
