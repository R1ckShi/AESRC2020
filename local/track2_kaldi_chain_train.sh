#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

set -e

exp=exp-kaldi
data=kdata
# configs for 'chain'
affix=all
stage=1
train_stage=-6
get_egs_stage=0
dir=$exp/chain/tdnn  # Note: _sp will get added to this
decode_iter=

# training options
num_epochs=5
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=2
nj=50
minibatch_size=128
dropout_schedule='0,0@0.20,0.3@0.50,0'
frames_per_eg=150,110,90
remove_egs=false
common_egs_dir=
common_egs_dir=
xent_regularize=0.1
graph=$exp/chain/graph 

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

dir=${dir}${affix:+_$affix}_sp
train_set=train 
test_sets="CHN IND KR PT RU UK US JPN"
ali_dir=$exp/tri3_ali
treedir=$exp/chain/tri4_cd_tree_sp
lang=$data/lang_chain

if [ $stage -le 1 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj 60 --cmd "$train_cmd" $data/$train_set \
    $data/lang $exp/tri3 $exp/tri4_sp_lats
  rm $exp/tri4_sp_lats/fsts.*.gz # save space
fi

if [ $stage -le 2 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r $data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 3 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 11500 $data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 4 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  output_opts="l2-regularize=0.015"
  tdnn_opts="l2-regularize=0.03 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.66"
  prefinal_opts="l2-regularize=0.03"
  ivector_affine_opts="l2-regularize=0.005"
  cnn_opts="l2-regularize=0.005"
  linear_opts="orthonormal-constraint=1.0"
  echo "$feat_dim"
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=71 name=input
  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=71 height-out=71 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=32 
  linear-component name=cnn2 dim=284 $linear_opts
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1280
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf18 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  tdnnf-layer name=tdnnf19 $tdnnf_opts dim=1280 bottleneck-dim=256 time-stride=3
  linear-component name=prefinal-l dim=512 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=512 big-dim=1280
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=512 big-dim=1280
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 5 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "run.pl" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $data/${train_set} \
    --tree-dir $treedir \
    --lat-dir $exp/tri4_sp_lats \
    --dir $dir  || exit 1;
fi


if [ $stage -le 6 ]; then
	./local/mkgraph.sh $lang $dir/final.mdl $graph 
fi

if [ $stage -le 7 ]; then
  for test_set in $test_sets; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
		--nj 20 --cmd "$decode_cmd" \
		$graph $data/cv/${test_set} $dir/decode_${test_set} || exit 1;
  done
fi

echo "local/track2_kaldi_chain_train.sh succeeded"
exit 0;
