import json
import torch
import random
from glob import glob


class BatchGen(object):
    """Generate endlessly the pair of current line and next line"""
    def __init__(self, pairs, batch_size):
        super(BatchGen, self).__init__()
        self.pairs = pairs
        self.batch_size = batch_size

    def pad_data(self, data):
        maxlen = max(map(len, data))
        for x in data:
            zeros = (maxlen - len(x)) * [0]
            x += zeros

    def __iter__(self):
        return self

    def __next__(self):
        batch = random.sample(self.pairs, self.batch_size)
        source, target = list(zip(*batch))
        self.pad_data(source)
        self.pad_data(target)
        source = torch.LongTensor(source)
        target = torch.LongTensor(target)
        return source, target


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


class Dictionary(object):
    def __init__(self, dict_file, vocab_size):
        words = []
        with open(dict_file) as f:
            for line in f:
                if line == '':
                    continue
                words.append(line.strip().split(',')[0])
        self.word2idx = {}
        self.pad = '<pad>'
        self.sos = '<sos>'
        self.eos = '<eos>'
        self.oov = '<oov>'
        self.offset = 4
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        for idx, word in enumerate(words[:vocab_size]):
            self.word2idx[word] = idx + 4
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, key):
        if type(key) is str:
            if key in self.word2idx:
                return self.word2idx[key]
            else:
                return self.word2idx[self.oov]
        if type(key) is int:
            if key in self.idx2word:
                return self.idx2word[key]
        raise KeyError(key)


class Corpus(object):
    def __init__(self, data_path, dict_file, vocab_size=1e6, kfold = 10, subset=None):
        self.dictionary = Dictionary(dict_file, vocab_size)
        self.vocab_size = vocab_size

        self.data, self.pairs = self.load_data(data_path, subset)
        self.train, self.test = self.split_data(kfold)

    def split_data(self, kfold):
        random.shuffle(self.data)
        train_size = int((kfold * 1. - 1) / kfold * len(self.data))
        return self.data[0 : train_size], self.data[train_size : ]

    def load_data(self, data_dir, subset):
        files = glob('{}/*/*.lrc'.format(data_dir))
        if subset:
            files = random.sample(files, subset)
        source = []
        target = []
        data = []
        for file in files:
            with open(file) as f:
                song = []
                for line in f:
                    words = []
                    if line == '':
                        continue
                    for x in line.strip().split(','):
                        x = int(x)
                        if x >= self.vocab_size:
                            x = self.dictionary[self.dictionary.oov]
                        else:
                            x += self.dictionary.offset
                        words.append(x)
                    words = [self.dictionary[self.dictionary.sos]] + words
                    words += [self.dictionary[self.dictionary.eos]]
                    song.append(words)
                data.extend(song)
            source.extend([line[:-1] for line in song[:-1]])
            target.extend([line[:-1] for line in song[1:]])
        pairs = list(zip(source, target))
        return data, pairs

    def load_data_new(self, file, subset):
        with open(file) as f:
            raw = json.load(f)
        self.raw = raw

    def gen_data(self):
        for chunk in self.raw:
            for line in chunk:
                yield [self.dictionary.sos] + \
                      [self.dictionary[w] for w in line] + \
                      [self.dictionary.eos]

    def gen_chunks(self, size):
        for chunk in self.raw:
            if len(chunk) > size:
                chunk = [[self.dictionary.sos] +
                         [self.dictionary[w] for w in line] +
                         [self.dictionary.eos]
                         for line in chunk]
                for i in range(len(chunk) - size + 1):
                    yield chunk[i: i + size]


def batchify(data, bsz, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        batch = data[i*bsz:(i+1)*bsz]

        # source has no end symbol
        source = [x[:-1] for x in batch]
        # target has no start symbol
        target = [x[1:] for x in batch]

        # find length to pad to, subtract 1 from lengths b/c includes BOTH starts & end symbols
        maxlen = max([len(x)-1 for x in batch])
        for x, y in zip(source, target):
            zeros = (maxlen-len(x))*[0]
            x += zeros
            y += zeros

        source = torch.LongTensor(source)
        target = torch.LongTensor(target).view(-1)

        batches.append((source, target))

    return batches


# def load_kenlm():
#     global kenlm
#     import kenlm


# def train_ngram_lm(kenlm_path, data_path, output_path, N):
#     """
#     Trains a modified Kneser-Ney n-gram KenLM from a text file.
#     Creates a .arpa file to store n-grams.
#     """
#     # create .arpa file of n-grams
#     curdir = os.path.abspath(os.path.curdir)
#     #
#     command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
#               " >"+os.path.join(curdir, output_path)
#     os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)

#     load_kenlm()
#     # create language model
#     model = kenlm.Model(output_path)

#     return model


# def get_ppl(lm, sentences):
#     """
#     Assume sentences is a list of strings (space delimited sentences)
#     """
#     total_nll = 0
#     total_wc = 0
#     for sent in sentences:
#         words = sent.strip().split()
#         score = lm.score(sent, bos=True, eos=False)
#         word_count = len(words)
#         total_wc += word_count
#         total_nll += score
#     ppl = 10**-(total_nll/total_wc)
#     return ppl
