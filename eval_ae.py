import argparse
import json
import sys
import random
import logging
import numpy as np

import torch
from torch.autograd import Variable

from models import Seq2Seq
from utils import Corpus, to_gpu, batchify

from tqdm import tqdm

'''
TODO clear pushover
Example Usage:
(eval)
python eval_ae.py --data_file chunks.json --dict_file vocab.txt --ae_args output/ae/args.json --model output/ae/autoencoder_model_12.pt --len_samples 1000 --batch_size 64 --split 0.1 --cuda
(debug)
python eval_ae.py --data_file sample.json --dict_file vocab.txt --ae_args output/ae/args.json --model output/ae/autoencoder_model_13.pt --batch_size 2 --split 0
'''


parser = argparse.ArgumentParser('Train an autoencoder.')
# Path Arguments
parser.add_argument('--data_file', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--dict_file', type=str, required=True,
                    help='location of the dictionary file')
parser.add_argument('--err_f', type=str, default='ae_err.txt',
                    help='filename for error samples')
parser.add_argument('--len_f', type=str, default='ae_len.txt')
parser.add_argument('--len_samples', type=int, default=100)
parser.add_argument('--ae_args', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
# Training Arguments
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--split', type=float, default=0,
                    help='the ratio of validation data.'
                         'set it to 0 to switch off validating')
# other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

args = parser.parse_args()
print(vars(args))

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
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)


def main():
    # prepare corpus
    ae_args = json.load(open(args.ae_args))
    corpus = Corpus(args.data_file,
                    args.dict_file,
                    vocab_size=ae_args['vocab_size'])
    autoencoder = Seq2Seq(emsize=ae_args['emsize'],
                          nhidden=ae_args['nhidden'],
                          ntokens=ae_args['ntokens'],
                          nlayers=ae_args['nlayers'],
                          hidden_init=ae_args['hidden_init'],
                          max_len=ae_args['max_len'],
                          gpu=args.cuda)
    autoencoder.load_state_dict(torch.load(args.model))
    if args.cuda:
        autoencoder.cuda()
    autoencoder.eval()

    if args.split:
        train, valid = corpus.get_data(split=args.split)
    else:
        valid = corpus.get_data()
    samples = batchify(random.sample(valid, args.len_samples),
                       args.batch_size, shuffle=False)
    valid = batchify(valid, args.batch_size, shuffle=False)

    word_accuracies = []
    sent_accuracies = []

    f = open(args.err_f, 'w')
    for i, batch in enumerate(tqdm(valid, desc='acc')):
        source, target, length = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))
        length = to_gpu(args.cuda, Variable(length, volatile=True))

        # output: batch x seq_len x ntokens
        code = autoencoder.encode(source)
        max_indices = autoencoder.generate(code, length).contiguous()

        # ============word accuracy============
        word_accuracies.extend(     # strip the last <eos>
            max_indices.view(-1).eq(target[:, :-1].contiguous().view(-1)).data.cpu().tolist())

        # ==============generate examples==================
        max_indices = max_indices.data.cpu().numpy()
        target = target.data.cpu().numpy()

        for t, idx in zip(target, max_indices):
            # real sentence
            real = "".join([corpus.dictionary.idx2word[x]
                            for x in t if x >= corpus.dictionary.offset])
            # autoencoder output sentence
            gen = "".join([corpus.dictionary.idx2word[x]
                           for x in idx if x >= corpus.dictionary.offset])
            correct = real == gen
            sent_accuracies.append(correct)
            if not correct:
                f.write('{} | {}\n'.format(real, gen))
    f.close()

    log.info('word acc: {} sent acc: {}'.format(
        np.mean(word_accuracies), np.mean(sent_accuracies)))

    f = open(args.len_f, 'w')
    for i, batch in enumerate(tqdm(samples, desc='len')):
        source, target, length = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        target = target.view_as(source).data.cpu().numpy()

        one = torch.LongTensor([1]).expand_as(length)
        indices = []
        for j in range(-2, 3):
            length_ = torch.max(length + j, one)
            length_ = to_gpu(args.cuda, Variable(length_, volatile=True))
            code = autoencoder.encode(source)
            max_indices = autoencoder.generate(code, length_)
            indices.append(max_indices.data.cpu().numpy())

        for k, target_ in enumerate(target):
            # real sentence
            real = "".join([corpus.dictionary.idx2word[x]
                            for x in target_ if x >= corpus.dictionary.offset])
            f.write('origin: {}\n'.format(real))
            for j in range(-2, 3):
                idx = indices[j][k]
                # autoencoder output sentence
                gen = "".join([corpus.dictionary.idx2word[x]
                               for x in idx if x >= corpus.dictionary.offset])
                f.write('{} {}\n'.format(j if j < 0 else '+' + str(j), gen))
    f.close()

if __name__ == '__main__':
    main()
