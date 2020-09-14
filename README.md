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
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-ycr8{background-color:#ffffff;text-align:left;vertical-align:top}
.tg .tg-tdvk{background-color:#ffffc7;text-align:left;vertical-align:middle}
.tg .tg-zumz{background-color:#ffffc7;border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-w747{background-color:#dae8fc;text-align:left;vertical-align:top}
.tg .tg-zv36{background-color:#ffffff;border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-6v43{background-color:#ffffff;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-c6of{background-color:#ffffff;border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-7g6k{background-color:#ffffff;border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-py60{background-color:#ffffc7;border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-ncd7{background-color:#ffffc7;border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-m9r4{background-color:#ffffc7;text-align:left;vertical-align:top}
.tg .tg-zci2{background-color:#dae8fc;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-bv7r{background-color:#dae8fc;text-align:left;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c6of" rowspan="2"></th>
    <th class="tg-zv36" rowspan="2">Data</th>
    <th class="tg-zv36" rowspan="2">Decode Related</th>
    <th class="tg-7g6k" colspan="9">CER on cv set</th>
  </tr>
  <tr>
    <td class="tg-zv36"><span style="font-weight:bold;font-style:normal">RU</span></td>
    <td class="tg-zv36"><span style="font-weight:bold;font-style:normal">KR</span></td>
    <td class="tg-6v43"><span style="font-weight:bold;font-style:normal">US</span></td>
    <td class="tg-6v43"><span style="font-weight:bold;font-style:normal">PT</span></td>
    <td class="tg-6v43"><span style="font-weight:bold;font-style:normal">JPN</span></td>
    <td class="tg-6v43"><span style="font-weight:bold;font-style:normal">UK</span></td>
    <td class="tg-6v43"><span style="font-weight:bold;font-style:normal">CHN</span></td>
    <td class="tg-6v43"><span style="font-weight:bold;font-style:normal">IND</span></td>
    <td class="tg-ycr8"><span style="font-weight:bold">AVE</span></td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-py60" rowspan="3"><br><br><br>Kaldi</td>
    <td class="tg-ncd7">Accent 160</td>
    <td class="tg-ncd7" rowspan="3">-</td>
    <td class="tg-zumz"><span style="font-weight:normal;font-style:normal">6.67</span></td>
    <td class="tg-zumz"><span style="font-weight:normal;font-style:normal">11.46</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">15.95</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">10.27</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">9.78</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">16.88</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">20.97</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">17.48</span></td>
    <td class="tg-m9r4">13.68</td>
  </tr>
  <tr>
    <td class="tg-ncd7">Libri 960 + Accent 160 Retrain</td>
    <td class="tg-zumz"><span style="font-weight:normal;font-style:normal">6.61</span></td>
    <td class="tg-zumz"><span style="font-weight:normal;font-style:normal">10.95</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">15.33</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">9.79</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">9.75</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">16.03</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">19.68</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">16.93</span></td>
    <td class="tg-m9r4">13.13</td>
  </tr>
  <tr>
    <td class="tg-m9r4">Accent 160 + Libri160</td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">6.95</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">11.76</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">13.05</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">9.96</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">10.15</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">14.21</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">20.76</span></td>
    <td class="tg-tdvk"><span style="font-weight:normal;font-style:normal">18.26</span></td>
    <td class="tg-m9r4">13.14</td>
  </tr>
  <tr>
    <td class="tg-zci2" rowspan="5"><br><br><br><br><br>ESPnet</td>
    <td class="tg-w747">Accent 160</td>
    <td class="tg-w747">+0.3 RNNLM</td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">5.26</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.69</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">9.96</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.45</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">6.79</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">10.06</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">11.77</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">10.05</span></td>
    <td class="tg-w747">8.63</td>
  </tr>
  <tr>
    <td class="tg-w747">Libri 960 + Accent 160 Retrain</td>
    <td class="tg-w747">+0.3 RNNLM</td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">4.6</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">6.4</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.42</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">5.9</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">5.71</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.64</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">9.87</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.85</span></td>
    <td class="tg-w747">6.92</td>
  </tr>
  <tr>
    <td class="tg-w747" rowspan="3"><br><br><br>Accent 160 + Libri160</td>
    <td class="tg-w747">+0.3 RNNLM + 0.3 CTC</td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">4.76</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.81</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.71</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">6.36</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">6.4</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.23</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">10.77</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">8.01</span></td>
    <td class="tg-w747">7.38</td>
  </tr>
  <tr>
    <td class="tg-w747">+0.3 RNNLM</td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">4.68</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.59</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.7</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">6.42</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">6.37</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.76</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">10.88</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">8.41</span></td>
    <td class="tg-w747">7.48</td>
  </tr>
  <tr>
    <td class="tg-w747"></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">5.35</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">9.07</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">8.52</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.13</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">7.29</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">8.6</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">12.03</span></td>
    <td class="tg-bv7r"><span style="font-weight:normal;font-style:normal">9.05</span></td>
    <td class="tg-w747">8.38</td>
  </tr>
</tbody>
</table>
![输入图片说明](https://images.gitee.com/uploads/images/2020/0911/221502_b412b116_8055062.png "track2.png")
