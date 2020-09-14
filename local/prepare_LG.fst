#!/bin/bash 

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "prepare_all.sh <corpus-data-dir> <lm-training-text>"
  echo " e.g prepare_all.sh data data/train/trans the data/ contains the dir of data and mfcc."
  exit 1;
fi

data_set=$1
train_text=$2

# L
if [ $stage -le 1 ]; then
  ./utils/prepare_lang.sh --position-dependent-phones false \
    $data_set/local/dict "<UNK>" $data_set/local/lang $data_set/lang || exit 1;
fi

# arpa LM
if [ $stage -le 2 ]; then
  local/train_kenlm.sh $train_text \
      $data_set/local/lm.arpa || exit 1;
fi

# G compilation, check LG composition
if [ $stage -le 3 ]; then
  ./local/format_lm.sh $data_set/lang $data_set/local/lm.arpa \
    $data_set/local/dict/lexicon.txt $data_set/lang_test || exit 1;
fi

echo "local/prepare_lang.sh succeeded"
exit 0;

