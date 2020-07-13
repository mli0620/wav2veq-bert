from torch.utils.data import Dataset
import tqdm
import torch
import random
import sys
import copy

def pad_list(x, pad_value,max_len):
    pad = x.new(max_len).fill_(pad_value)
    pad[:x.size(0)] = x
    return pad
    


class ASRDataset(Dataset):
    def __init__(self, corpus_path, vocab,asr_path, vocab_asr, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.vocab_asr = vocab_asr
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.asr_path = asr_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            with open(asr_path, "r", encoding=encoding) as f_asr:
                if self.corpus_lines is None and not on_memory:
                    for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        self.corpus_lines += 1

                if on_memory:
                    self.lines = [line[:-1].split("\t")
                                  for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                    self.lines_asr = [line[:-1].split("\t")
                                  for line in tqdm.tqdm(f_asr, desc="Loading Dataset", total=corpus_lines)]

                    self.corpus_lines = len(self.lines)
                    self.max_len = max(len(line[0].split()) for line in self.lines_asr)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1,t1_label = self.random_sent(item)
        t1 = t1[0]
        t1_label = t1_label[0]
        ilen = max(min(self.seq_len,len(t1.split())),self.max_len*2)
        t1 = self.vocab.to_seq(t1, self.seq_len)
        t1_label = self.vocab_asr.to_seq(t1_label, len(t1_label.split())) 

        bert_input = t1[:self.seq_len]
        asr_label = t1_label


        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        
        #print(111111,len(bert_input))
        #print(2222,len(asr_label))
        #sys.exit()
        #output = {"bert_input": bert_input}#,
        #          "asr_label": asr_label}#,
                  #"segment_label": segment_label,
                  #"is_next": is_next_label}
        #return {key: torch.tensor(value) for key, value in output.items()}
        asr_label = torch.tensor(asr_label)
        #print(32423,asr_label)
        asr_label_pad = pad_list(asr_label, self.vocab_asr.pad_index,self.max_len) 
        #print(13213,asr_label_pad) 
        #sys.exit()
        return torch.tensor(bert_input),asr_label_pad,torch.tensor(ilen)

    def random_sent(self, index):
        t1, t1_label = self.get_corpus_line(index)
        return t1, t1_label

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item],self.lines_asr[item]
            #return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2


