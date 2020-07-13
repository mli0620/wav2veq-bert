import torch.nn as nn
import sys
from .bert import BERT
from argparse import Namespace
from .ctc import CTC
import torch

class BERT_CTC(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("asr model setting")
        group.add_argument('--adim', default=512, type=int,
                help='Number of attention transformation dimensions')
        group.add_argument('--dropout-rate', default=0.0, type=float,
                help='Dropout rate for the encoder')


    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.ctc = CTC(vocab_size, 512, 0.0, ctc_type='builtin', reduce=True)

    def forward(self, x, y,ilens):#, segment_label):
        #ilens = torch.tensor([200])
        x = self.bert(x)
        loss_ctc = self.ctc(x,ilens,y) 
        return loss_ctc

