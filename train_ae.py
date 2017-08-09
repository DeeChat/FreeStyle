import argparse
import json
import os
import sys
import random
import logging
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models import Seq2Seq
from utils import Corpus, to_gpu, batchify, Dictionary

'''
TODO clear pushover
Example Usage:
(train)
python train_ae.py --data_file chunks.json --dict_file vocab.txt --epochs 30 --max_len 20 --outf ae --batch_size 64 --split 0.1 --log_interval 100 --cuda
(debug)
python train_ae.py --data_file sample.json --dict_file vocab.txt --max_len 20 --outf ae --batch_size 2 --split 0.1 --log_interval 10
'''


parser = argparse.ArgumentParser('Train an autoencoder.')
# Path Arguments
parser.add_argument('--data_file', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--dict_file', type=str, required=True,
                    help='location of the dictionary file')
parser.add_argument('--outf', type=str, default='example',
                    help='output directory name')
parser.add_argument('--max_len', type=int, required=True,
                    help='max number of tokens in lines')
# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=12000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
# Training Arguments
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=15,
                    help='maximum number of epochs')
parser.add_argument('--split', type=float, default=0.1,
                    help='the ratio of validation data.'
                         'set it to 0 to switch off validating')
# Model Arguments
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--noise_radius', type=float, default=0.2,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.995,
                    help='anneal noise_radius exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')

# other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=100)

args = parser.parse_args()
print(vars(args))

# create output directory
out_dir = './output/{}'.format(args.outf)
os.makedirs(out_dir, exist_ok=True)

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(out_dir, 'logs.txt'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def main():
    # prepare corpus
    corpus = Corpus(args.data_file,
                    args.dict_file,
                    vocab_size=args.vocab_size)

    # dumping vocabulary
    with open(os.path.join(out_dir, 'vocab.json'), 'w') as f:
        json.dump(corpus.dictionary.word2idx, f)

    # save arguments
    ntokens = len(corpus.dictionary.word2idx)
    args.ntokens = ntokens
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    log.info('[Data Loaded.]')

    autoencoder = AutoEncoder()

    if args.split:
        train, valid = corpus.get_data(split=args.split)
        valid = batchify(valid, args.batch_size, shuffle=False)
    else:
        train = corpus.get_data()

    for epoch in range(1, args.epochs + 1):
        # shuffle train data in each epoch
        batches = batchify(train, args.batch_size, shuffle=True)

        global_iters = 0
        start_time = datetime.now()

        for i, batch in enumerate(batches):
            loss = autoencoder.update(batch)
            if i % args.log_interval == 0 and i > 0:
                log.info(('[Epoch {} {}/{} Loss {:.5f} ETA {}]').format(
                    epoch, i, len(batches), loss,
                    str((datetime.now() - start_time) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))

            global_iters += 1
            if global_iters % 100 == 0:
                autoencoder.anneal()

        if args.split:
            word_acc, sent_acc = autoencoder.evaluate(valid)
            msg = 'Epoch {} word acc: {} | sent acc: {}'.format(epoch, word_acc, sent_acc)
            log.warn(msg)
        autoencoder.save(out_dir, 'autoencoder_model_{}.pt'.format(epoch))


class AutoEncoder:
    def __init__(self):
        self.autoencoder = Seq2Seq(emsize=args.emsize,
                                   nhidden=args.nhidden,
                                   ntokens=args.ntokens,
                                   nlayers=args.nlayers,
                                   max_len=args.max_len,
                                   noise_radius=args.noise_radius,
                                   hidden_init=args.hidden_init,
                                   dropout=args.dropout,
                                   gpu=args.cuda)
        # self.optimizer = optim.SGD(self.autoencoder.parameters(), lr=args.lr)
        self.optimizer = optim.Adam(self.autoencoder.parameters())
        self.criterion = nn.CrossEntropyLoss()
        if args.cuda:
            self.autoencoder = self.autoencoder.cuda()
            self.criterion = self.criterion.cuda()

    def update(self, batch):
        self.autoencoder.train()
        self.autoencoder.zero_grad()

        source, target, length = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))
        length = to_gpu(args.cuda, Variable(length))

        # output: batch x seq_len x ntokens
        output = self.autoencoder(source, length, noise=True)

        # output_size: batch_size, maxlen, self.ntokens
        flattened_output = output.view(-1, args.ntokens)

        loss = self.criterion(flattened_output / args.temp, target.view(-1))
        loss.backward()

        # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm(self.autoencoder.parameters(), args.clip)
        self.optimizer.step()

        return loss.data.cpu().numpy()[0]

    def anneal(self):
        '''exponentially decaying noise on autoencoder'''
        self.autoencoder.noise_radius = self.autoencoder.noise_radius * args.noise_anneal

    def evaluate(self, valid):
        self.autoencoder.eval()
        word_accuracies = []
        sent_accuracies = []

        for i, batch in enumerate(valid):
            source, target, length = batch
            source = to_gpu(args.cuda, Variable(source, volatile=True))
            target = to_gpu(args.cuda, Variable(target, volatile=True))
            length = to_gpu(args.cuda, Variable(length, volatile=True))

            # output: batch x seq_len x ntokens
            code = self.autoencoder.encode(source)
            max_indices = self.autoencoder.generate(code, length).contiguous()

            # ============word accuracy============
            word_accuracies.extend(     # strip the last <eos>
                max_indices.view(-1).eq(target[:, :-1].contiguous().view(-1)).data.cpu().tolist())

            # ==============generate examples==================
            max_indices = max_indices.data.cpu().numpy()
            target = target.data.cpu().numpy()

            for t, idx in zip(target, max_indices):
                # real sentence
                real = [x for x in t if x >= Dictionary.offset]

                # autoencoder output sentence
                gen = [x for x in idx if x >= Dictionary.offset]
                sent_accuracies.append(real == gen)

        return np.mean(word_accuracies), np.mean(sent_accuracies)

    def save(self, dirname, filename):
        with open(os.path.join(dirname, filename), 'wb') as f:
            torch.save(self.autoencoder.state_dict(), f)


if __name__ == '__main__':
    main()
