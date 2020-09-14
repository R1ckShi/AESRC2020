#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=1        # start from 0 if you need to start from data preparation
stop_stage=1
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/espnet_train.conf
lm_config=conf/espnet_lm.yaml
decode_config=conf/espnet_decode.yaml
preprocess_config=conf/espnet_specaug.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=0             # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5

# exp tag
tag="base" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data=kdata
exp=exp-espnet

train_set=train
train_dev=cv_all
#recog_set="dev test"
recog_set="cv/UK cv/US cv/CHN cv/JPN cv/KR cv/RU cv/IND cv/PT"


lexi=$data/lang/lexicon.txt
dict=$data/lang/units.txt
echo "dictionary: ${dict}"

# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=$exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: LM Preparation"
    lmdatadir=$data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 $data/train/text | cut -f 2- -d" " \
        > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 $data/${train_dev}/text | cut -f 2- -d" " \
        > ${lmdatadir}/valid.txt

    ${cuda_cmd} --gpu 1 ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu $ngpu \
        --backend ${backend} \
		--batchsize 1000 \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=$exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Network Training"
    ${cuda_cmd} --gpu $ngpu ${expdir}/train.log \
        asr_train.py \
	        --config ${train_config} \
	        --ngpu $ngpu \
	        --backend ${backend} \
	        --outdir ${expdir}/results \
	        --tensorboard-dir tensorboard/${expname} \
	        --debugmode ${debugmode} \
	        --dict ${dict} \
	        --debugdir ${expdir} \
        	--minibatches ${N} \
	        --verbose ${verbose} \
	        --resume ${resume} \
	        --train-json $data/$train_set/asr.json \
	        --valid-json $data/$train_dev/asr.json \
			--n-iter-processes $ngpu
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding"
    nj=30
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
	recog_model=model.last${n_average}.avg.best
	average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=$data/$rtask
		echo $feat_recog_dir 
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/asr.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/asr.JOB.json \
            --result-label ${expdir}/${decode_dir}/asr.JOB.json \
            --model ${expdir}/results/${recog_model}

        score_sclite.sh ${expdir}/${decode_dir} ${dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
