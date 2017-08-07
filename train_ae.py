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
from utils import Corpus, to_gpu, batchify

'''
Example Usage:
(train)
python train_ae.py --data_file data/chunks_default.json --dict_file data/vocab.txt --outf train-ae-64 --batch_size 64 --split 0.1 \
--log_interval 100 --wv_file data/new_word_embeddings.txt --wv_dict_file data/wv_dict.json --cuda
(debug)
python train_ae.py --data_file data/sample.json --dict_file data/vocab.txt --outf sample-ae --batch_size 2 --split 0.1 \
--log_interval 10 --wv_file data/new_word_embeddings.txt --wv_dict_file data/wv_dict.json
'''


parser = argparse.ArgumentParser('Train an autoencoder.')
# Path Arguments
parser.add_argument('--data_file', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--dict_file', type=str, required=True,
                    help='location of the dictionary file')
parser.add_argument('--outf', type=str, default='example',
                    help='output directory name')
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
                    help='the ratio of test data.')
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
parser.add_argument('--wv_file', type=str, default="word_embedding.txt")
parser.add_argument('--wv_dict_file', type=str, default="wv_dict.json")

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
formatter = logging.Formatter(fmt='[%(asctime)s] - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def get_embeddings(idx2word, wv_path, wv_dict_path):
    with open(wv_dict_path, 'r') as f:
        wv_dict = json.load(f)
    pretrained_wv = {}
    with open(wv_path, 'r') as f:
        line = f.readline()
        n_words, vector_size = map(int, line.split())
        for line in f.readlines():
            s = line.split()
            idx = s[0]
            v = list(map(float, s[1:]))
            pretrained_wv[idx] = v
    assert(vector_size == args.emsize)
    embeddings = []
    ntokens = len(idx2word)
    for i in range(ntokens):
        w = idx2word[i]
        if w in wv_dict and wv_dict[w] in pretrained_wv:
            embeddings.append(pretrained_wv[wv_dict[w]])
        else:
            print('{} is not in pretrained word embeddings'.format(w))
            embeddings.append(np.random.uniform(-0.1, 0.1, vector_size).tolist())

    return torch.Tensor(embeddings)

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

    embeddings = None # get_embeddings(corpus.dictionary.idx2word, args.wv_file, args.wv_dict_file)
    log.info('[Data Loaded.]')

    autoencoder = AutoEncoder(embeddings)

    train, valid = corpus.get_data(split=args.split)
    valid = batchify(valid, args.batch_size, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        # shuffle train data in each epoch
        batches = batchify(train, args.batch_size, shuffle=True)

        global_iters = 0
        start_time = datetime.now()

        for i, batch in enumerate(batches):
            loss, average_precision, global_acc = autoencoder.update(batch)
            if i % args.log_interval == 0 and i > 0:
                log.info(('[Epoch {} {}/{} Loss: {:.5f}, AP: {:.5f}, global acc: {:0.5f}, ETA: {}]').format(
                    epoch, i, len(batches), loss, average_precision, global_acc,
                    str((datetime.now() - start_time) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))

            global_iters += 1
            if global_iters % 100 == 0:
                autoencoder.anneal()

        valid_loss, average_precision, global_acc = autoencoder.evaluate(valid)
        log.warn('Epoch {} valid loss: {:.5f} | AP: {:.5f} | global acc: {:.5f}'.format(epoch, valid_loss, average_precision, global_acc))

        autoencoder.save(out_dir, 'autoencoder_model_{}.pt'.format(epoch))

class AutoEncoder:
    def __init__(self, pretrained_weights = None):
        self.autoencoder = Seq2Seq(emsize=args.emsize,
                                   nhidden=args.nhidden,
                                   ntokens=args.ntokens,
                                   nlayers=args.nlayers,
                                   noise_radius=args.noise_radius,
                                   hidden_init=args.hidden_init,
                                   dropout=args.dropout,
                                   gpu=args.cuda,
                                   pretrained_weights=pretrained_weights
                                   )
        # self.optimizer = optim.SGD(self.autoencoder.parameters(), lr=args.lr)
        self.optimizer = optim.Adam(self.autoencoder.parameters())
        self.criterion = nn.CrossEntropyLoss()
        if args.cuda:
            self.autoencoder = self.autoencoder.cuda()
            self.criterion = self.criterion.cuda()

    def update(self, batch):
        self.autoencoder.train()
        self.autoencoder.zero_grad()

        source, target = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))

        # Create sentence length mask over padding
        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), args.ntokens)

        # output: batch x seq_len x ntokens
        output = self.autoencoder(source, noise=True)

        # output_size: batch_size, maxlen, self.ntokens
        flattened_output = output.view(-1, args.ntokens)

        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, args.ntokens)
        loss = self.criterion(masked_output / args.temp, masked_target)
        loss.backward()

        # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm(self.autoencoder.parameters(), args.clip)
        self.optimizer.step()

        # new accuracy
        all_accuracies = []
        max_vals, max_indices = torch.max(flattened_output, 1)
        max_len = max_indices.size(0) // args.batch_size
        for i in range(args.batch_size):
            sample_mask = mask[i * max_len: (i + 1) * max_len]
            sample_out = max_indices[i * max_len: (i + 1) * max_len].masked_select(sample_mask)
            sample_target = target[i * max_len: (i + 1) * max_len].masked_select(sample_mask)
            acc = torch.mean(sample_out.eq(sample_target).float()).data[0]
            all_accuracies.append(acc)
            # print('acc = ', acc, ',  mean = ', np.mean(all_accuracies))
        global_acc = len(list(filter(lambda x: x >= 1.0, all_accuracies)))*1.0 / len(all_accuracies)
        # print(all_accuracies, ', global_acc = ', global_acc)

        return loss.data.cpu().numpy()[0], np.mean(all_accuracies), global_acc

    def anneal(self):
        '''exponentially decaying noise on autoencoder'''
        self.autoencoder.noise_radius = self.autoencoder.noise_radius * args.noise_anneal

    def evaluate(self, valid):
        self.autoencoder.eval()
        total_loss = []
        all_accuracies = []

        for i, batch in enumerate(valid):
            source, target = batch
            source = to_gpu(args.cuda, Variable(source, volatile=True))
            target = to_gpu(args.cuda, Variable(target, volatile=True))

            mask = target.gt(0)
            masked_target = target.masked_select(mask)
            # examples x ntokens
            output_mask = mask.unsqueeze(1).expand(mask.size(0), args.ntokens)

            # output: batch x seq_len x ntokens
            output = self.autoencoder(source, noise=True)
            flattened_output = output.view(-1, args.ntokens)

            masked_output = \
                flattened_output.masked_select(output_mask).view(-1, args.ntokens)
            total_loss.append(self.criterion(masked_output / args.temp, masked_target).data.cpu().numpy()[0])

            # new accuracy
            max_vals, max_indices = torch.max(flattened_output, 1)
            max_len = max_indices.size(0) // args.batch_size
            for i in range(args.batch_size):
                sample_mask = mask[i * max_len: (i + 1) * max_len]
                sample_out = max_indices[i * max_len: (i + 1) * max_len].masked_select(sample_mask)
                sample_target = target[i * max_len: (i + 1) * max_len].masked_select(sample_mask)
                acc = torch.mean(sample_out.eq(sample_target).float()).data[0]
                all_accuracies.append(acc)
            global_acc = len(list(filter(lambda x: x >= 1.0, all_accuracies)))*1.0 / len(all_accuracies)
            # print(all_accuracies, ', global_acc = ', global_acc)

            # accuracy
            # max_vals, max_indices = torch.max(masked_output, 1)
            # all_accuracies.append(
            #    torch.mean(max_indices.eq(masked_target).float()).data[0])
            # global_acc = len(list(filter(lambda x: x >= 1.0, all_accuracies)))*1.0 / len(all_accuracies)
            # print('global_acc = ', global_acc)
        return np.mean(total_loss), np.mean(all_accuracies), global_acc

    def save(self, dirname, filename):
        with open(os.path.join(dirname, filename), 'wb') as f:
            torch.save(self.autoencoder.state_dict(), f)

if __name__ == '__main__':
    main()
