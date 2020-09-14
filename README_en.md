# AESRC2020


#### Introduction

Data preparation scripts and training pipeline for the Interspeech 2020 Accented English Speech Recognition Challenge (AESRC).

#### Dependent Environment

1.  Install Kaldi (Data preparation scripts, Track2 traditional ASR model training) 
    [Github Link](https://github.com/kaldi-asr/kaldi)
2.  Install ESPnet(Track1 E2E AR Model training, Track2 E2E ASR Transformer training)
    [Github Link](https://github.com/espnet/espnet)
3.  (Optional) Install Google SentencePiece (Track2 E2E ASR modeling units building)
    [Github Link](https://github.com/google/sentencepiece)
4.  (Optional) Install KenLM (N-gram language model training)
    [Github Link](http://https://github.com/kpu/kenlm)

#### Usage

 **Data Preparation** 

1.  Download challenge data
2.  Data preparation, divide cv set, feature extraction and bpe model training `./local/prepare_data.sh`

 **AR Track** 

Train Track1 ESPnet AR model `./local/track1_espnet_transformer_train.sh`

 **ASR Track** 

1.  Train Track2 Kaldi GMM alignment model  `./local/track2_kaldi_gmm_train.sh`
2.  Generate Lattice, decision tree, Train Track2 Kaldi Chain Model  `./local/track2_kaldi_chain_train.sh`
3.  Train Track2 ESPnet Transformer Model (Track2 ESPnet RNN Language Model) `./local/track2_espnet_transformer_train.sh`

**Notice**
1.  There's no lexicon provided, please prepare it by yourself.
2.  Data augment methods are not included in scirpts.
3.  Install Kaldi and ESPnet and activate their envrionment then you can run the scripts.
4.  Baseline experiments in Track2 include several data using methods.
5.  Participants should obey the rules about data strictly.

#### Baseline Experiments Results

**Track1** 

| Model    | RU   | KR   | US   | PT   | JPN  | UK   | CHN  | IND  | AVE  |
| -------- | -- |---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Transformer-3L  | 30.0 | 45.0 | 45.7 | 57.2 | 48.5 | 70.0 | 56.2 | 83.5 | 54.1 |
| Transformer-6L  | 34.0 | 43.7 | 30.6 | 65.7 | 44.0 | 74.5 | 50.9 | 75.2 | 52.2 |
| Transformer-12L | 49.6 | 26.0 | 21.2 | 51.8 | 42.7 | 85.0 | 38.2 | 66.1 | 47.8 |
| + ASR-init      | 75.7 | 55.6 | 60.2 | 85.5 | 73.2 | 93.9 | 67.0 | 97.0 | 76.1 |

Transformer-3L, Transformer-6L, Transformer-12L all use`./local/track1_espnet_transformer_train.sh` (elayers: 3, 6, 12)

ASR-init uses encoder in Track2 to initialize self-attention parameters

*In cv sets, we found that the acc of some accent is strongly related with speaker. As there are few speakers in cv sets, the absolute value above is not statistically significant, and the test set will contain more speakers

**Track2** 

Kaldi Hybrid Chain Model: CNN + 18 TDNN
*Based on internal non open source dictionary
*Results on CMU dict comes up soon

ESPnet Transformer Model: 12 Encoder + 6 Decoder (simple self-attention, CTC joint training used, 1k sub-word BPE)

You can find detailed hyperparameters settings in `./local/files/conf/` and training scripts
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
    <td rowspan="3">Kaldi<br></td>
    <td>Accent160</td>
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
    <td>Libri960 ~ Accent160</td>
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
    <td>Accent160 + Libri160</td>
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
    <td rowspan="5">ESPnet<br></td>
    <td>Accent160</td>
    <td>+0.3RNNLM</td>
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
    <td>Libri960 ~ Accent160</td>
    <td>+0.3RNNLM</td>
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
    <td rowspan="3">Accent160 +Libri160<br></td>
    <td>-</td>
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
  <tr>
    <td>+0.3RNNLM</td>
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
    <td>+0.3RNNLM+0.3CTC</td>
    <td>4.76</td>
    <td>7.81</td>
    <td>7.71</td>
    <td>6.36</td>
    <td>6.4</td>
    <td>7.23</td>
    <td>10.77</td>
    <td>8.01</td>
    <td>7.38   </td>
  </tr>
</tbody>
</table>
* Data A ~ Data B means fine-tune Data A model with Data B
