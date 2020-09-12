#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

order=3
prune="0 1 1"
mem_rate=40%
output_dir=
arpa_name=
fallback="0.5 1 1.5"

input=$1

cat $input | lmplz \
    -o $order \
    -S $mem_rate \
    --prune $prune \
    --discount_fallback $fallback \
    --arpa $2

echo "local/train_kenlm.sh succeeded"
exit 0;
