#!/usr/bin/env python3
import re
import codecs
import json
import re
import sys
from bpemb import BPEmb
import os 
from os.path import join, basename, exists, getctime,splitext
#bpemb_en = BPEmb(lang="en", dim=50)


def main(BPE,dataset):
    lines_raw = codecs.open('data/%s.text'%dataset, encoding="utf-8").readlines()
    with codecs.open('data/%s_%s.text'%(dataset,vs), 'w', encoding="utf-8") as f:
        for line_raw in lines_raw:
            bpe_list = bpemb_en.encode(line_raw[:-1])
            new_line = ' '.join(bpe_list) + '\n'
            f.write(new_line)



if __name__ == "__main__":
    #bpemb_en = BPEmb(lang="en", dim=50)
    dataset = sys.argv[1]
    vs = int(sys.argv[2])
    bpemb_en = BPEmb(lang="en", dim=50,vs = vs)
    main(bpemb_en,dataset)
