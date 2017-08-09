import json
import torch
import random


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


class Dictionary(object):
    pad = 0
    sos = 1
    eos = 2
    oov = 3
    offset = 4

    def __init__(self, dict_file, vocab_size):
        words = []
        with open(dict_file) as f:
            for line in f:
                if line == '':
                    continue
                words.append(line.split(',')[0])
        self.word2idx = {}
        self.word2idx['<pad>'] = self.pad
        self.word2idx['<sos>'] = self.sos
        self.word2idx['<eos>'] = self.eos
        self.word2idx['<oov>'] = self.oov
        for idx, word in enumerate(words[:vocab_size]):
            self.word2idx[word] = idx + self.offset
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, key):
        if type(key) is str:
            if key in self.word2idx:
                return self.word2idx[key]
            else:
                # print(key, [ord(c) for c in key])
                # exit()
                return self.oov
        if type(key) is int:
            if key in self.idx2word:
                return self.idx2word[key]
        raise KeyError(key)


class Corpus(object):
    def __init__(self, data_file, dict_file, vocab_size=11000):
        self.dictionary = Dictionary(dict_file, vocab_size)
        with open(data_file) as f:
            raw = json.load(f)
        self.raw = raw

    def get_data(self, split=None):
        data = [[self.dictionary.sos] +
                [self.dictionary[w] for w in line] +
                [self.dictionary.eos]
                for chunk in self.raw for line in chunk]
        if not split:
            return data
        else:
            train_size = int(len(data) * max(split, 1 - split))
            return data[:train_size], data[train_size:]

    def get_chunks(self, size):
        chunks = []
        for chunk in self.raw:
            if len(chunk) > size:
                # source has no end symbol
                chunk = [[self.dictionary.sos] +
                         [self.dictionary[w] for w in line]
                         for line in chunk]
                for i in range(len(chunk) - size + 1):
                    chunks.append(chunk[i: i + size])
        return chunks


def batchify(data, bsz, shuffle=False):
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        batch = data[i * bsz:(i + 1) * bsz]

        # source has no end symbol
        source = [x[:-1] for x in batch]
        # target has no start symbol
        target = [x[1:] for x in batch]

        # find length to pad to, subtract 1 from lengths b/c includes BOTH starts & end symbols
        lengths = [len(x) - 1 for x in batch]
        maxlen = max(lengths)
        for x, y in zip(source, target):
            padding = (maxlen - len(x)) * [Dictionary.pad]
            x += padding
            y += padding

        source = torch.LongTensor(source)
        target = torch.LongTensor(target)
        # substract 1 for length embedding indexing
        lengths = torch.LongTensor([x - 1 for x in lengths])

        batches.append((source, target, lengths))

    return batches


class BatchGen(object):
    """Generate endlessly the pair of current line and next line"""
    def __init__(self, chunks, batch_size):
        super(BatchGen, self).__init__()
        self.chunks = chunks
        self.batch_size = batch_size

    def pad_data(self, data):
        maxlen = max(map(len, data))
        return [x + (maxlen - len(x)) * [Dictionary.pad] for x in data]

    def __iter__(self):
        return self

    def __next__(self):
        batch = random.sample(self.chunks, self.batch_size)
        source, target = list(zip(*batch))
        source = self.pad_data(source)
        target = self.pad_data(target)
        source = torch.LongTensor(source)
        target = torch.LongTensor(target)
        return source, target
