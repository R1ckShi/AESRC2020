#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

set -e

nj=50
stage=2

. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh;
. ./utils/parse_options.sh

data=kdata
exp=kaldi-exp

test_sets="CHN IND KR PT RU UK US JPN"

# mono
if [ $stage -le 2 ]; then
  # training
  steps/train_mono.sh --cmd "$decode_cmd" --nj $nj \
    $data/train $data/lang $exp/mono || exit 1;

  # alignment
  steps/align_si.sh --cmd "$decode_cmd" --nj $nj \
    $data/train $data/lang $exp/mono $exp/mono_ali || exit 1;
fi

# tri1
if [ $stage -le 3 ]; then
  # training
  steps/train_deltas.sh --cmd "$decode_cmd" \
    4000 32000 $data/train $data/lang $exp/mono_ali $exp/tri1 || exit 1;

  # alignment
  steps/align_si.sh --cmd "$decode_cmd" --nj $nj \
    $data/train $data/lang $exp/tri1 $exp/tri1_ali || exit 1;
fi

# tri2
if [ $stage -le 4 ]; then
  # training
  steps/train_deltas.sh --cmd "$decode_cmd" \
    7000 56000 $data/train $data/lang $exp/tri1_ali $exp/tri2 || exit 1;

  # alignment
  steps/align_si.sh --cmd "$decode_cmd" --nj $nj \
    $data/train $data/lang $exp/tri2 $exp/tri2_ali || exit 1;
fi

# tri3
if [ $stage -le 5 ]; then
  # training [LDA+MLLT]
  steps/train_lda_mllt.sh --cmd "$decode_cmd" \
    10000 80000 $data/train $data/lang $exp/tri2_ali $exp/tri3 || exit 1;

  # decoding
  utils/mkgraph.sh $data/lang_test $exp/tri3 $exp/tri3/graph || exit 1;

  for test_set in $test_sets;do
    steps/decode.sh --cmd "$decode_cmd" --nj 30 --config conf/decode.conf \
      $exp/tri3/graph $data/cv/$test_set $exp/tri3/decode_test_$test_set
  done

  # alignment
  steps/align_si.sh --cmd "$decode_cmd" --nj $nj \
    $data/train $data/lang $exp/tri3 $exp/tri3_ali || exit 1;
fi

echo "local/track2_kaldi_gmm_train.sh succeeded"
exit 0;

