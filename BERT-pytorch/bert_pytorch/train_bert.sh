#!/bin/bash
python __main__.py -c data/valid.src -v data/vocab.train -o output/bert.model -hs 512 -l 12 -a 8 -b 18 -s 1000 -e 30 #--ckpt /mnt/lustre/xushuang2/mli/BERT-pytorch/bert_pytorch/output/bert.model.ep13


