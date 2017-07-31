import argparse
import logging
import os
import sys
from datetime import datetime
import numpy as np
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils import to_gpu, Corpus, BatchGen
from models import Seq2Seq, MLP_D, MLP_G

'''
Example Usage:
python train.py --data_file chunks.json --dict_file vocab.txt --ae_model output/ae/autoencoder_model_5.pt --ae_args output/ae/args.json --outf gan --batch_size 64 --log_interval 200 --updates 200000 --cuda
'''

parser = argparse.ArgumentParser(description='GAN for Lyrics Generation')
# Path Arguments
parser.add_argument('--data_file', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--dict_file', type=str, required=True,
                    help='location of the dictionary file')
parser.add_argument('--ae_model', type=str, required=True,
                    help='pre-trained autoencoder model file.')
parser.add_argument('--ae_args', type=str, required=True)
parser.add_argument('--outf', type=str, default='gan',
                    help='output directory name')

# Model Arguments
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--arch_g', type=str, default='300-300',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='300-300',
                    help='critic/discriminator architecture (MLP)')

# Training Arguments
parser.add_argument('--updates', type=int, default=3e5,
                    help='number of model updates')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-05,
                    help='critic/discriminator learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--N', type=int, default=5,
                    help='N-gram order for training n-gram language model')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=10000)

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

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
    state_dict = torch.load(args.ae_model)
    with open(args.ae_args) as f:
        ae_args = json.load(f)

    corpus = Corpus(args.data_file,
                    args.dict_file,
                    vocab_size=ae_args['vocab_size'])
    autoencoder = Seq2Seq(emsize=ae_args['emsize'],
                          nhidden=ae_args['nhidden'],
                          ntokens=ae_args['ntokens'],
                          nlayers=ae_args['nlayers'],
                          noise_radius=ae_args['noise_radius'],
                          hidden_init=ae_args['hidden_init'],
                          dropout=ae_args['dropout'],
                          gpu=args.cuda)
    autoencoder.load_state_dict(state_dict)
    for param in autoencoder.parameters():
        param.requires_grad = False
    # save arguments
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    log.info('[Data and AE model loaded.]')

    gan_gen = MLP_G(ninput=args.nhidden, noutput=args.nhidden, layers=args.arch_g)
    gan_disc = MLP_D(ninput=2 * args.nhidden, noutput=1, layers=args.arch_d)
    optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                                 lr=args.lr_gan_g,
                                 betas=(args.beta1, 0.999))
    optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                                 lr=args.lr_gan_d,
                                 betas=(args.beta1, 0.999))
    criterion_ce = nn.CrossEntropyLoss()

    if args.cuda:
        autoencoder = autoencoder.cuda()
        gan_gen = gan_gen.cuda()
        gan_disc = gan_disc.cuda()
        criterion_ce = criterion_ce.cuda()

    one = to_gpu(args.cuda, torch.FloatTensor([1]))
    mone = one * -1
    train_pairs = BatchGen(corpus.get_chunks(size=2), args.batch_size)

    def train_gan_g(batch):
        gan_gen.train()
        gan_gen.zero_grad()

        source, _ = batch
        source = to_gpu(args.cuda, Variable(source))
        source_hidden = autoencoder(source, noise=False, encode_only=True)

        fake_hidden = gan_gen(source_hidden)
        errG = gan_disc(source_hidden, fake_hidden)

        # loss / backprop
        errG.backward(one)
        optimizer_gan_g.step()

        return errG

    def train_gan_d(batch):
        # clamp parameters to a cube
        for p in gan_disc.parameters():
            p.data.clamp_(-args.gan_clamp, args.gan_clamp)

        gan_disc.train()
        gan_disc.zero_grad()

        # positive samples ----------------------------
        # generate real codes
        source, target = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))

        # batch_size x nhidden
        source_hidden = autoencoder(source, noise=False, encode_only=True)
        target_hidden = autoencoder(target, noise=False, encode_only=True)

        # loss / backprop
        errD_real = gan_disc(source_hidden, target_hidden)
        errD_real.backward(one)

        # negative samples ----------------------------

        # loss / backprop
        fake_hidden = gan_gen(source_hidden)
        errD_fake = gan_disc(source_hidden.detach(), fake_hidden.detach())
        errD_fake.backward(mone)

        optimizer_gan_d.step()
        errD = -(errD_real - errD_fake)

        return errD, errD_real, errD_fake

    niter = 0
    start_time = datetime.now()

    for t in range(args.updates):
        niter += 1

        # train discriminator/critic
        for i in range(args.niters_gan_d):
            # feed a seen sample within this epoch; good for early training
            errD, errD_real, errD_fake = \
                train_gan_d(next(train_pairs))

        # train generator
        for i in range(args.niters_gan_g):
            errG = train_gan_g(next(train_pairs))

        if niter % args.log_interval == 0:
            eta = str((datetime.now() - start_time) / (t + 1) * (args.updates - t - 1)).split('.')[0]
            log.info('[{}/{}] Loss_D: {:.6f} (real: {:.6f} '
                     'fake: {:.6f}) Loss_G: {:.6f} ETA: {}'
                     .format(niter, args.updates, errD.data.cpu()[0], errD_real.data.cpu()[0],
                             errD_fake.data.cpu()[0], errG.data.cpu()[0], eta))
        if niter % args.save_interval == 0:
            save_model(gan_gen, out_dir, 'gan_gen_model_{}.pt'.format(t))
            save_model(gan_disc, out_dir, 'gan_disc_model_{}.pt'.format(t))


def save_model(model, dir, filename):
    with open(os.path.join(dir, filename), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__ == '__main__':
    main()
