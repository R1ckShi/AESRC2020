# AESRC2020


#### 介绍

Interspeech 2020 口音英语识别挑战赛数据准备相关脚本、训练流程代码与基线实验结果。

Data preparation scripts and training pipeline for the Interspeech 2020 Accented English Speech Recognition Challenge (AESRC).

#### 依赖环境

1.  安装Kaldi (数据准备有关功能脚本、Track2传统模型训练) 
    [Github链接](https://github.com/kaldi-asr/kaldi)
2.  安装ESPnet（Track1 E2E AR Model训练、Track2 E2E ASR Transformer训练）
    [Github链接](https://github.com/espnet/espnet)
3.  （可选）安装Google SentencePiece （Track2 E2E ASR 词表缩减、建模单元构建）
    [Github链接](https://github.com/google/sentencepiece)
4.  （可选）安装KenLM （N-gram语言模型训练）
    [Github链接](http://https://github.com/kpu/kenlm)

#### 使用说明

 **数据准备 Data Preparation** 

1.  下载评测数据
2.  准备数据，划分开发集，特征准备以及训练BPE模型 `./local/prepare_data.sh`

 **口音识别赛道 AR Track** 

训练Track1 ESPnet AR模型 `./local/track1_espnet_transformer_train.sh`

 **语音识别赛道 ASR Track** 

1.  训练Track2 Kaldi GMM对齐模型  `./local/track2_kaldi_gmm_train.sh`
2.  生成Lattice，决策树，训练Track2 Kaldi Chain Model  `./local/track2_kaldi_chain_train.sh`
3.  训练Track2 ESPnet Transformer模型（Track2 ESPnet RNN语言模型） `./local/track2_espnet_transformer.sh`

**注意**
1.  官方不提供Kaldi模型所需的英文的发音词典
2.  训练脚本中不包括数据扩充、添加Librispeech数据等，参赛者可按需添加
3.  正确安装并激活Kaldi与ESPnet的环境之后才能运行相关脚本
4.  ASR Track中Baseline提供了多种数据的组合、Librispeech全量数据预训练等试验结果
5.  参赛者应严格按照评测中关于数据使用的相关规则训练模型，以确保结果的公平可比性

#### 基线实验结果

**Track1基线实验结果** 

| Model    | RU   | KR   | US   | PT   | JPN  | UK   | CHN  | IND  | AVE  |
| -------- | -- |---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Transformer-6L  | 34.0 | 43.7 | 30.6 | 65.7 | 44.0 | 74.5 | 50.9 | 75.2 | 52.2 |
| Transformer-12L | 49.6 | 26.0 | 21.2 | 51.8 | 42.7 | 85.0 | 38.2 | 66.1 | 47.8 |
| + ASR-init      | 75.7 | 55.6 | 60.2 | 85.5 | 73.2 | 93.9 | 67.0 | 97.0 | 76.1 |


**Track2基线实验结果** 

Kaldi Hybrid Chain Model: CNN + 18 TDNN
*基于内部的非开源英文发音词典

ESPnet Transformer Model: 12 Encoder + 6 Decoder (simple self-attention, CTC joint training used, 1k sub-word BPE)

详细超参数见`./files/conf/`目录中模型配置与相关脚本中的设置
![输入图片说明](https://images.gitee.com/uploads/images/2020/0911/221502_b412b116_8055062.png "track2.png")
