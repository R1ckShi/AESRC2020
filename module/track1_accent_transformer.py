#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 SpeechLab @ SJTU (Author: Yizhou Lu)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

""" Transformer-based accent recognition model (pytorch), 
    Codes mainly borrowed from espnet (https://github.com/espnet/espnet)
"""

from argparse import Namespace
from distutils.util import strtobool

import logging
import math

import torch
import chainer
from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport

class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, acc, loss):
        """Report at every step."""
        reporter.report({'acc': acc}, self)
        reporter.report({'loss': loss}, self)

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        group.add_argument('--pretrained-model', default="", type=str,
                           help='pretrained ASR model for initialization')
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        odim = odim - 1 # ignore additional dim added by data2json
        self.odim = odim 
        self.ignore_id = ignore_id
        self.subsample = [1]
        self.reporter = Reporter()
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        self.output = torch.nn.Linear(2 * args.adim, self.odim) # mean + std pooling
        # reset parameters
        self.reset_parameters(args)
        logging.warning(self)

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        if args.pretrained_model:
            path = args.pretrained_model
            logging.warning("load pretrained asr model from {}".format(path))
            if 'snapshot' in path:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
            else:
                model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(model_state_dict, strict=False)
            del model_state_dict
        else:
            initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: label smoothing loss value
        :rtype: torch.Tensor
        """
        # forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        mean = torch.mean(hs_pad, dim=1).unsqueeze(1)
        std = torch.std(hs_pad, dim=1).unsqueeze(1)
        hs_pad = torch.cat((mean, std), dim=-1) # (B, 1, D)
        # output layer
        pred_pad = self.output(hs_pad)

        # compute loss
        self.loss = self.criterion(pred_pad, ys_pad)
        self.acc = th_accuracy(pred_pad.view(-1, self.odim), ys_pad,
                               ignore_label=self.ignore_id)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(self.acc, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0) # (B, T, D) with #B=1
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0) # returns tensor(T, D)

    # todo: batch decoding
    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        """
        enc_output = self.encode(x).unsqueeze(0) # (1, T, D)
        mean = torch.mean(enc_output, dim=1).unsqueeze(1) # (1, 1, D)
        std = torch.std(enc_output, dim=1).unsqueeze(1)
        enc_output = torch.cat((mean, std), dim=-1)
        lpz = self.output(enc_output)
        lpz = lpz.squeeze(0) # shape of (T, D)
        idx = lpz.argmax(-1).cpu().numpy().tolist()
        hyp = {}
        # [-1] is added here to be compatible with ASR decoding, see espnet/asr/asr_utils/parse_hypothesis
        hyp['yseq'] = [-1] + idx
        hyp['score'] = -1
        logging.info(hyp['yseq'])
        return [hyp]

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.cpu().numpy()
        return ret

    # fix calculate_all_ctc_probs method not implemented bug
    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad):
        return None

