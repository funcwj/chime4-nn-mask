#!/bin/bash
# wujian@17.11.9


[ $# -ne 1 ] && echo "format error: $0: <stage>" && exit 1

nj=15
chime_dir=./CHiME4
masks_dst=./masks
enhan_dst=../enhan/nngev_6mics
estimator=./mdl/estimator_0.3496

stage=$1

json_list="tr05_simu tr05_real dt05_simu dt05_real et05_simu et05_real"

# step 1: split .json files
if [ $stage -eq 1 ]; then 
    echo "spliting json data..."
    for x in $json_list; do
        ./local/split_json.py $chime_dir/data/annotations/$x.json $nj \
            --output-dir $chime_dir/data/annotations || exit 1
    done
    exit 0
fi

# step 2: get clean/noise data from simulate data in CHiME4
if [ $stage -eq 2 ]; then 
    echo "seperate clean/noise wave from CHiME4 simulated data..."
    cp ./local/CHiME4_simulate_data_pactched.m ./local/run.pl $chime_dir/simulation
    for x in $(seq $nj); do
        echo "CHiME4_simulate_data_pactched($x);" > $chime_dir/simulation/run_simulate_$x.m
    done
    cd $chime_dir/simulation
    ./run.pl JOB=1:$nj log/simulate.JOB.log \
       matlab -nojvm -nodesktop -nosplash -r run_simulate_JOB
    exit 0
fi
# step 3: prepare mask for NN training 
# modified chime_data.py from https://github.com/fgnt/nn-gev 

if [ $stage -eq 3 ]; then 
    echo "prepare target masks, noise/clean spectrums for NN training..."
    ./local/run.pl JOB=1:$nj log/calculate_mask.JOB.log \
        python -c "import chime_data; chime_data.prepare_training_data('$chime_dir/data', '$masks_dst', JOB)"
    exit 0
fi

# step 4: training estimator
if [ $stage -eq 4 ]; then 
    echo "training mask estimator..."
    ./train_estimator.py $masks_dst --nj $nj > train.log 
    exit 0;
fi 

# step 5: enhance data
# I do not modify the implemetation of beamforming in fgnt
if [ $stage -eq 5 ]; then 
    for x in et05_simu et05_real dt05_simu dt05_real; do 
        echo "enhance $x.json..."
        ./local/run.pl JOB=1:$nj log/gev_et05_simu.JOB.log \
            ./apply_nngev_beamformer.py --enhan-dir $enhan_dst $estimator ${x}_JOB.json 
    done
    exit 0
fi
