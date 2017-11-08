#!/bin/bash

[ $# -ne 1 ] && echo "format error: $0 <nj>" && exit 1
nj=$1

for x in $(seq $nj); do
    echo "CHiME4_simulate_data_pactched($x);" > run_simulate_$x.m
done

./run.pl JOB=1:$nj simulate_log/JOB.log \
    matlab -nojvm -nodesktop -nosplash -r run_simulate_JOB
