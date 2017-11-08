#!/bin/bash

nj=15

[ $# -ne 2 ] && echo "format error: $0: <chime-data-dir> <dest-mask-dir>" && exit 1

chime_data=$1
masks_dest=$2

# modified chime_data.py from https://github.com/fgnt/nn-gev 
./run.pl JOB=1:$nj log/calculate_mask.JOB.log \
    python -c "import chime_data; chime_data.prepare_training_data('$chime_data', '$masks_dest', JOB)"
