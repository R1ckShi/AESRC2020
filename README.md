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
3.  训练Track2 ESPnet Transformer模型（Track2 ESPnet RNN语言模型） `./local/track2_espnet_transformer_train.sh`

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

Transformer-6L、Transformer-12L均使用`./local/track1_espnet_transformer_train.sh`训练（elayers分别为6、12），ASR-init实验使用Track2中Joint CTC/Attention模型进行初始化

**Track2基线实验结果** 

Kaldi Hybrid Chain Model: CNN + 18 TDNN
*基于内部的非开源英文发音词典
*随后会公布基于CMU词典的结果

ESPnet Transformer Model: 12 Encoder + 6 Decoder (simple self-attention, CTC joint training used, 1k sub-word BPE)

详细超参数见`./local/files/conf/`目录中模型配置与相关脚本中的设置
<table>
<thead>
  <tr>
    <th rowspan="2"></th>
    <th rowspan="2">Data</th>
    <th rowspan="2">Decode Related</th>
    <th colspan="9">CER on cv set</th>
  </tr>
  <tr>
    <td>RU</td>
    <td>KR</td>
    <td>US</td>
    <td>PT</td>
    <td>JPN</td>
    <td>UK</td>
    <td>CHN</td>
    <td>IND</td>
    <td>AVE</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3"><br><br><br>Kaldi</td>
    <td>Accent 160</td>
    <td rowspan="3">-</td>
    <td>6.67</td>
    <td>11.46</td>
    <td>15.95</td>
    <td>10.27</td>
    <td>9.78</td>
    <td>16.88</td>
    <td>20.97</td>
    <td>17.48</td>
    <td>13.68</td>
  </tr>
  <tr>
    <td>Libri 960 + Accent 160 Retrain</td>
    <td>6.61</td>
    <td>10.95</td>
    <td>15.33</td>
    <td>9.79</td>
    <td>9.75</td>
    <td>16.03</td>
    <td>19.68</td>
    <td>16.93</td>
    <td>13.13</td>
  </tr>
  <tr>
    <td>Accent 160 + Libri160</td>
    <td>6.95</td>
    <td>11.76</td>
    <td>13.05</td>
    <td>9.96</td>
    <td>10.15</td>
    <td>14.21</td>
    <td>20.76</td>
    <td>18.26</td>
    <td>13.14</td>
  </tr>
  <tr>
    <td rowspan="5"><br><br><br><br><br>ESPnet</td>
    <td>Accent 160</td>
    <td>+0.3 RNNLM</td>
    <td>5.26</td>
    <td>7.69</td>
    <td>9.96</td>
    <td>7.45</td>
    <td>6.79</td>
    <td>10.06</td>
    <td>11.77</td>
    <td>10.05</td>
    <td>8.63</td>
  </tr>
  <tr>
    <td>Libri 960 + Accent 160 Retrain</td>
    <td>+0.3 RNNLM</td>
    <td>4.6</td>
    <td>6.4</td>
    <td>7.42</td>
    <td>5.9</td>
    <td>5.71</td>
    <td>7.64</td>
    <td>9.87</td>
    <td>7.85</td>
    <td>6.92</td>
  </tr>
  <tr>
    <td rowspan="3"><br><br><br>Accent 160 + Libri160</td>
    <td>+0.3 RNNLM + 0.3 CTC</td>
    <td>4.76</td>
    <td>7.81</td>
    <td>7.71</td>
    <td>6.36</td>
    <td>6.4</td>
    <td>7.23</td>
    <td>10.77</td>
    <td>8.01</td>
    <td>7.38</td>
  </tr>
  <tr>
    <td>+0.3 RNNLM</td>
    <td>4.68</td>
    <td>7.59</td>
    <td>7.7</td>
    <td>6.42</td>
    <td>6.37</td>
    <td>7.76</td>
    <td>10.88</td>
    <td>8.41</td>
    <td>7.48</td>
  </tr>
  <tr>
    <td></td>
    <td>5.35</td>
    <td>9.07</td>
    <td>8.52</td>
    <td>7.13</td>
    <td>7.29</td>
    <td>8.6</td>
    <td>12.03</td>
    <td>9.05</td>
    <td>8.38</td>
  </tr>
</tbody>
</table>
