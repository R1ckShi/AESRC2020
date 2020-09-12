#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

raw_data=rdata     # raw data with metadata, txt and wav
data=kdata         # data transformed into kaldi format
zipped_data=$raw_data/AESRC2020.zip 

stage=7
feature_cmd="run.pl"
nj=50


# unzip and rename each accent
if [ $stage -le 1 ];then
    # unzip $zipped_data
    mv $raw_data/American\ English\ Speech\ Data $raw_data/US
    mv $raw_data/British\ English\ Speech\ Data $raw_data/UK
    mv $raw_data/Chinese\ Speaking\ English\ Speech\ Data $raw_data/CHN 
    mv $raw_data/Indian\ English\ Speech\ Data $raw_data/IND 
    mv $raw_data/Portuguese\ Speaking\ English\ Speech\ Data $raw_data/PT 
    mv $raw_data/Russian\ Speaking\ English\ Speech\ Data $raw_data/RU 
    mv $raw_data/Japanese\ Speaking\ English\ Speech\ Data $raw_data/JPN 
    mv $raw_data/Korean\ Speaking\ English\ Speech\ Data $raw_data/KR
fi


# generate kaldi format data for all
if [ $stage -le 2 ];then 
    echo "Generating kaldi format data."
    mkdir -p $data/data_all
    find `pwd`/ -name '*.wav' > $data/data_all/wavpath
    awk -F'/' '{print $(NF-2)"-"$(NF-1)"-"$NF}' $data/data_all/wavpath | sed 's:\.wav::g' > $data/data_all/uttlist
    paste $data/data_all/uttlist $data/data_all/wavpath > $data/data_all/wav.scp
    python local/preprocess.py $data/data_all/wav.scp $data/data_all/trans $data/data_all/utt2spk # faster than for in shell
    ./utils/utt2spk_to_spk2utt.pl $data/data_all/utt2spk > $data/data_all/spk2utt
fi


# clean transcription
if [ $stage -le 3 ];then
    echo "Cleaning transcription."
    tr '[a-z]' '[A-Z]' < $data/data_all/trans > $data/data_all/trans_upper
    # turn "." in specific abbreviations into "<m>" tag
    sed -i -e 's: MR\.: MR<m>:g' -e 's: MRS\.: MRS<m>:g' -e 's: MS\.: MS<m>:g' \
        -e 's:^MR\.:MR<m>:g' -e 's:^MRS\.:MRS<m>:g' -e 's:^MS\.:MS<m>:g' $data/data_all/trans_upper 
    sed -i 's:ST\.:STREET:g' $data/data_all/trans_upper 
    # punctuation marks
    sed -i "s%,\|\.\|?\|!\|;\|-\|:\|,'\|\.'\|?'\|!'\| '% %g" $data/data_all/trans_upper
    sed -i 's:<m>:.:g' $data/data_all/trans_upper
    # blank
    sed -i 's:[ ][ ]*: :g' $data/data_all/trans_upper
    paste $data/data_all/uttlist $data/data_all/trans_upper > $data/data_all/text
fi


# extracting filter-bank features and cmvn
if [ $stage -le 4 ];then 
    ./utils/fix_data_dir.sh $data/data_all
    ./steps/make_fbank.sh --cmd $feature_cmd --nj $nj --fbank-config $raw_data/fbank.conf $data/data_all $data/feats/log $data/feats/ark
    ./steps/compute_cmvn_stats.sh $data/data_all $data/feats/log $data/feats/ark # for kaldi 
fi


# divide development set for cross validation
if [ $stage -le 5 ];then 
    for i in US UK IND CHN JPN PT RU KR;do 
        ./utils/subset_data_dir.sh --spk-list local/files/cvlist/${i}_cv_spk $data/data_all $data/cv/$i 
        cat $data/cv/$i/feats.scp >> $data/cv.scp 
    done
    ./utils/filter_scp.pl --exclude $data/cv.scp $data/data_all/feats.scp > $data/train.scp 
    ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/train
	./utils/subset_data_dir.sh --utt-list $data/cv.scp $data/data_all $data/cv_all
	compute-cmvn-stats scp:$data/train/feats.scp `pwd`/$data/train/dump_cmvn.ark # for espnet
    rm $data/cv.scp $data/train.scp 
fi


# generate label file and dump features for track2:E2E
if [ $stage -le 6 ];then 
    for i in US UK IND CHN JPN PT RU KR;do 
        local/dump.sh --cmd $feature_cmd --nj 3 --do_delta false \
            $data/cv/$i/feats.scp $data/train/dump_cmvn.ark $data/cv/$i/dump/log $data/cv/$i/dump # for track2 e2e testing
    done 
    local/dump.sh --cmd $feature_cmd --nj $nj  --do_delta false \
        $data/train/feats.scp $data/train/dump_cmvn.ark $data/train/dump/log $data/train/dump
    local/dump.sh --cmd $feature_cmd --nj $nj  --do_delta false \
        $data/cv_all/feats.scp $data/train/dump_cmvn.ark $data/cv_all/dump/log $data/cv_all/dump # for track1 testing
fi


# generate label file for track1
if [ $stage -le 7 ];then 
    for i in train cv_all;do 
        cut -f 1 $data/$i/text > $data/$i/uttlist 
        cut -d '-' -f 1 $data/$i/text | sed -e "s:^:<:g" -e "s:$:>:g" > $data/$i/accentlist
        paste $data/$i/uttlist $data/$i/accentlist > $data/$i/utt2accent 
        rm $data/$i/uttlist
		local/data2json.sh --nj 20 --feat $data/$i/dump/feats.scp --text $data/$i/utt2accent --oov 8 $data/$i local/files/ar.dict > $data/$i/ar.json
	done
fi    

echo "local/prepare_data.sh succeeded"
exit 0;
