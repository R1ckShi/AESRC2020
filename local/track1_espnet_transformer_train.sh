#!/bin/bash

# Copyright 2020 Speechlab @ SJTU (Author: Yizhou Lu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=1
stop_stage=2
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
log=100

preprocess_config=conf/specaug.yaml
train_config=conf/track1_accent_transformer.yaml

# others
accum_grad=2
n_iter_processes=2
lsm_weight=0.0
epochs=40
elayers=12
batch_size=20
pretrained_model=

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_json=kdata/train/ar.json
valid_json=kdata/cv_all/ar.json

expdir=exp/track1_accent_classification_transformer_elayers${elayers}
mkdir -p ${expdir}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --report-interval-iters ${log} \
        --accum-grad ${accum_grad} \
        --n-iter-processes ${n_iter_processes} \
        --elayers ${elayers} \
        --lsm-weight ${lsm_weight} \
        --epochs ${epochs} \
        --batch-size ${batch_size} \
        ${pretrained_model:+--pretrained-model $pretrained_model} \
        --train-json ${train_json} \
        --valid-json ${valid_json}
fi

decode_dir=decode_track1
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Decoding"
    nj=10
    
    # split data
    dev_root=kdata/cv_all
    splitjson.py --parts ${nj} ${dev_root}/ar.json
    #### use CPU for decoding
    ngpu=0

    slurm.pl JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --batchsize 0 \
        --recog-json ${dev_root}/split${nj}utt/ar.JOB.json \
        --result-label ${expdir}/${decode_dir}/ar.JOB.json \
        --model ${expdir}/results/${recog_model}

    concatjson.py ${expdir}/${decode_dir}/ar.*.json >  ${expdir}/${decode_dir}/ar.json
    echo "Decoding finished"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Analyze decoding results"
    python ./local/tools/parse_track1_jsons.py  ${expdir}/${decode_dir}/ar.json ${expdir}/${decode_dir}/result.txt

fi
