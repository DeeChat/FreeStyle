import argparse
import time
import json
import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from models import Seq2Seq
from utils import Corpus, to_gpu, batchify

def process(args):
    def train_ae(batch, total_loss_ae, start_time, i):
        """
        train autoencoder in batch
        """
        autoencoder.train()
        autoencoder.zero_grad()

        source, target = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))

        # Create sentence length mask over padding
        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        # output: batch x seq_len x ntokens
        output = autoencoder(source, noise=True)

        # output_size: batch_size, maxlen, self.ntokens
        flattened_output = output.view(-1, ntokens)

        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        loss = criterion_ce(masked_output/args.temp, masked_target)
        loss.backward()

        # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
        optimizer_ae.step()

        total_loss_ae += loss.data

        if i % args.log_interval == 0 and i > 0:
            # accuracy
            probs = F.softmax(masked_output)
            max_vals, max_indices = torch.max(probs, 1)
            accuracy = torch.mean(max_indices.eq(masked_target).float()).data[0]

            cur_loss = total_loss_ae[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
                  .format(epoch, i, len(train_data),
                          elapsed * 1000 / args.log_interval,
                          cur_loss, math.exp(cur_loss), accuracy))

            with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}\n'.
                        format(epoch, i, len(train_data),
                               elapsed * 1000 / args.log_interval,
                               cur_loss, math.exp(cur_loss), accuracy))

            total_loss_ae = 0
            start_time = time.time()

        return total_loss_ae, start_time

    def save_model(model, dir, filename):
        with open(os.path.join(dir, filename), 'wb') as f:
            torch.save(model.state_dict(), f)

    def evaluate_autoencoder(data_source, epoch):
        # Turn on evaluation mode which disables dropout.
        autoencoder.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary.word2idx)
        all_accuracies = 0
        bcnt = 0
        for i, batch in enumerate(data_source):
            source, target = batch
            source = to_gpu(args.cuda, Variable(source, volatile=True))
            target = to_gpu(args.cuda, Variable(target, volatile=True))

            mask = target.gt(0)
            masked_target = target.masked_select(mask)
            # examples x ntokens
            output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

            # output: batch x seq_len x ntokens
            output = autoencoder(source, noise=True)
            flattened_output = output.view(-1, ntokens)

            masked_output = \
                flattened_output.masked_select(output_mask).view(-1, ntokens)
            total_loss += criterion_ce(masked_output/args.temp, masked_target).data

            # accuracy
            max_vals, max_indices = torch.max(masked_output, 1)
            all_accuracies += \
                torch.mean(max_indices.eq(masked_target).float()).data[0]
            bcnt += 1

            aeoutf = "./output/%s/%d_autoencoder.txt" % (args.outf, epoch)
            with open(aeoutf, "a") as f:
                max_values, max_indices = torch.max(output, 2)
                max_indices = \
                    max_indices.view(output.size(0), -1).data.cpu().numpy()
                target = target.view(output.size(0), -1).data.cpu().numpy()
                for t, idx in zip(target, max_indices):
                    # real sentence
                    chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                    f.write(chars)
                    f.write("\n")
                    # autoencoder output sentence
                    chars = " ".join([corpus.dictionary.idx2word[x] for x in idx])
                    f.write(chars)
                    f.write("\n\n")

        return total_loss[0] / len(data_source), all_accuracies/bcnt

    ###############################
    # 1. prepare corpus
    corpus = Corpus(args.data_path,
                    args.dict_file,
                    vocab_size=args.vocab_size,
                    subset=args.subset)
    print('data_size = {}, train_size = {}, test_size = {}'.
          format(len(corpus.data), len(corpus.train), len(corpus.test)))

    # dumping vocabulary
    with open('./output/{}/vocab.json'.format(args.outf), 'w') as f:
        json.dump(corpus.dictionary.word2idx, f)

    # save arguments
    ntokens = len(corpus.dictionary.word2idx)
    print("Vocabulary Size: {}".format(ntokens))
    args.ntokens = ntokens

    print("Loaded data!")

    ###############################
    # 2. create autoencoder
    autoencoder = Seq2Seq(emsize=args.emsize,
                          nhidden=args.nhidden,
                          ntokens=ntokens,
                          nlayers=args.nlayers,
                          noise_radius=args.noise_radius,
                          hidden_init=args.hidden_init,
                          dropout=args.dropout,
                          gpu=args.cuda)

    optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
    criterion_ce = nn.CrossEntropyLoss()
    if args.cuda:
        autoencoder = autoencoder.cuda()
        criterion_ce = criterion_ce.cuda()

    ###############################
    eval_batch_size = 10
    test_data = batchify(corpus.test, eval_batch_size, shuffle=False)

    # 3. start training autoencoder
    for epoch in range(1, args.epochs+1):
        # shuffle train data in each epoch
        train_data = batchify(corpus.train, args.batch_size, shuffle=True)

        total_loss_ae = 0
        epoch_start_time = time.time()
        start_time = time.time()

        # loop through all batches in training data
        for niter in range(len(train_data)):
            total_loss_ae, start_time = \
                train_ae(train_data[niter], total_loss_ae, start_time, niter)

            # if niter_global % 100 == 0:
            #    # exponentially decaying noise on autoencoder
            #    autoencoder.noise_radius = \
            #        autoencoder.noise_radius*args.noise_anneal

        ###############################
        # 4. start testing
        # end of epoch ----------------------------
        # evaluation
        test_loss, accuracy = evaluate_autoencoder(test_data, epoch)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
           'test ppl {:5.2f} | acc {:3.3f}'.
           format(epoch, (time.time() - epoch_start_time),
                  test_loss, math.exp(test_loss), accuracy))
        print('-' * 89)

        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write('-' * 89)
            f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                 ' test ppl {:5.2f} | acc {:3.3f}\n'.
                 format(epoch, (time.time() - epoch_start_time),
                        test_loss, math.exp(test_loss), accuracy))
            f.write('-' * 89)
            f.write('\n')

        save_model(autoencoder, "./output/{}".format(args.outf), 'autoencoder_model.pt')


def main():
    print('start training autoencoder.....')
    parser = argparse.ArgumentParser()
    # Path Arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--dict_file', type=str, required=True,
                        help='location of the dictionary file')
    parser.add_argument('--outf', type=str, default='example',
                        help='output directory name')
    # Data Processing Arguments
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='cut vocabulary down to this size '
                             '(most frequently seen words in train)')
    # Training Arguments
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=15,
                        help='maximum number of epochs')
    parser.add_argument('--min_epochs', type=int, default=6,
                        help="minimum number of epochs to train for")
    parser.add_argument('--kfold', type=int, default=10,
                        help='the ratio of train data is (kfold - 1)/kfold')
    parser.add_argument('--subset', type=int, default=0,
                        help='use a small amount of data for quick debugging.')
    # Model Arguments
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhidden', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr_ae', type=float, default=1,
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

    ## other
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # make output directory if it doesn't already exist
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    if not os.path.isdir('./output/{}'.format(args.outf)):
        os.makedirs('./output/{}'.format(args.outf))

    with open('./output/{}/args.json'.format(args.outf), 'w') as f:
        json.dump(vars(args), f)
    with open("./output/{}/logs.txt".format(args.outf), 'w') as f:
        f.write(str(vars(args)))
        f.write("\n\n")

    process(args)

if __name__ == '__main__':
    """
    python ./train_ae.py --data_path data/lyrics --dict_file data/dictionary --outf train-ae \
    --batch_size 32 --kfold 10 --subset 1000 --log_interval 100
    """
    main()
