import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import to_gpu
import json


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(
                    layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, max_len=20, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)
        # one more <sos> or <eos>
        max_len += 1
        emsize_len = int(np.ceil(np.log(max_len) / np.log(2)))
        self.embedding_length = nn.Embedding(max_len, emsize_len)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize + nhidden + emsize_len
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)
        self.embedding_length.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def forward(self, indices, length, noise):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, noise)

        embed_len = self.embedding_length(length.unsqueeze(1))
        hidden = torch.cat([hidden,
                            embed_len.squeeze(1)],
                           dim=1)

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices)

        return decoded

    def encode(self, indices, noise=False):
        embeddings = self.embedding(indices)

        # Encode
        packed_output, state = self.encoder(embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        hidden = torch.div(hidden, norms.expand_as(hidden))

        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)

        output, state = self.decoder(augmented_embeddings, state)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, length, sample=False, temp=1.0):
        """Generate through decoder; no backprop"""
        embed_len = self.embedding_length(length.unsqueeze(1))
        hidden = torch.cat([hidden,
                            embed_len.squeeze(1)],
                           dim=1)

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(torch.max(length.data)):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                # sampling
                probs = F.softmax(overvocab / temp)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


def load_ae(ae_args_file, ae_model, vocab_file):
    ae_args = json.load(open(ae_args_file, "r"))
    word2idx = json.load(open(vocab_file, "r"))
    idx2word = {v: k for k, v in word2idx.items()}
    autoencoder = Seq2Seq(emsize=ae_args['emsize'],
                          nhidden=ae_args['nhidden'],
                          ntokens=ae_args['ntokens'],
                          nlayers=ae_args['nlayers'],
                          hidden_init=ae_args['hidden_init'])
    autoencoder.load_state_dict(torch.load(ae_model))
    return ae_args, autoencoder, idx2word


def load_models(ae_args_file, gan_args_file, vocab_file, ae_model, g_model, d_model):
    ae_args = json.load(open(ae_args_file, "r"))
    gan_args = json.load(open(gan_args_file, 'r'))
    word2idx = json.load(open(vocab_file, "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=ae_args['emsize'],
                          nhidden=ae_args['nhidden'],
                          ntokens=ae_args['ntokens'],
                          nlayers=ae_args['nlayers'],
                          hidden_init=ae_args['hidden_init'],
                          gpu=True)
    gan_gen = MLP_G(ninput=gan_args['nhidden'],
                    noutput=gan_args['nhidden'],
                    layers=gan_args['arch_g'])
    gan_disc = MLP_D(ninput=2 * gan_args['nhidden'],
                     noutput=1,
                     layers=gan_args['arch_d'])

    autoencoder.load_state_dict(torch.load(ae_model))
    gan_gen.load_state_dict(torch.load(g_model))
    gan_disc.load_state_dict(torch.load(d_model))
    return ae_args, gan_args, idx2word, autoencoder, gan_gen, gan_disc


def decode_idx(vocab, idx):
    # generated sentence
    words = [vocab[x] for x in idx]
    # truncate sentences to first occurrence of <eos>
    truncated_sent = []
    for w in words:
        if w == '<sos>' or w == '<pad>':
            continue
        if w != '<eos>':
            truncated_sent.append(w)
        else:
            break
    sent = "".join(truncated_sent)
    return sent


def generate_from_hidden(autoencoder, hidden, vocab, sample, maxlen):
    # autoencoder.eval()
    max_indices = autoencoder.generate(hidden=hidden,
                                       maxlen=maxlen,
                                       sample=sample)
    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = "".join(truncated_sent)
        sentences.append(sent)
    return sentences


def generate(autoencoder, gan_gen, inp, vocab, sample, maxlen):
    """
    Assume inp is batch_size x max_sen_len
    """
    gan_gen.eval()
    autoencoder.eval()

    # generate from the previous line
    fake_hidden = gan_gen(autoencoder.encode(inp))
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=maxlen,
                                       sample=sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = "".join(truncated_sent)
        sentences.append(sent)
    return sentences
