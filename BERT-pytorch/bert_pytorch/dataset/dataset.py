from torch.utils.data import Dataset
import tqdm
import torch
import random
import sys
import copy

class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1]
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                

                self.corpus_lines = len(self.lines)
        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1 = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = t1_random

        #t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t1_label = t1_label

        bert_input = t1[:self.seq_len]
        bert_label = t1_label[:self.seq_len]


        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding)
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label}#,
                  #"segment_label": segment_label,
                  #"is_next": is_next_label}
        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = self.vocab.to_seq(sentence, self.seq_len)
        ori_tokens = copy.deepcopy(tokens)
        output_label = []
        for i, token in enumerate(ori_tokens):
            prob = random.random()
            if prob < 0.05 :#and token != self.vocab.pad_index:
                span = min(10,len(tokens)-i)
                tokens[i:i+span] = [self.vocab.mask_index for _ in range(span)]

                output_label.append(token)

            else:
                if tokens[i] == self.vocab.mask_index:
                    output_label.append(token)
                else:
                    output_label.append(0)
        return tokens, output_label

    def random_sent(self, index):
        t1 = self.get_corpus_line(index)
        return t1

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            #return self.lines[random.randrange(len(self.lines))][1]
            return self.lines[random.randrange(len(self.lines))]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]

