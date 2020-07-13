#!/bin/bash
python train_asr.py -c /mnt/lustre/xushuang2/mli/BERT-pytorch/bert_pytorch/data/train.src -v data/vocab.train -o output/bert.model -hs 512 -l 12 -a 8 -b 8 -s 2000 -e 30 --trans_t data/train_5000.text --asr_vocab data/asr.train  --ckpt /mnt/lustre/xushuang2/mli/BERT-pytorch/bert_pytorch/output/bert.model.ep1


