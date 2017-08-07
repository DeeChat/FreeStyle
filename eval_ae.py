import random
import torch
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify
from models import load_ae


def evaluate_autoencoder(autoencoder, outf, corpus, sample_size):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()

    source, target = batchify(corpus.get_data(), sample_size, shuffle=True)[0]
    source = Variable(source, volatile=True)  # to_gpu(True,
    target = Variable(target, volatile=True)

    # output: batch x seq_len x ntokens
    output = autoencoder(source, noise=False)

    with open(outf, "w") as f:
        max_values, max_indices = torch.max(output, 2)
        max_indices = \
            max_indices.view(output.size(0), -1).data.cpu().numpy()
        target = target.view(output.size(0), -1).data.cpu().numpy()
        for t, idx in zip(target, max_indices):
            # real sentence
            chars = "".join([corpus.dictionary.idx2word[x] for x in t if x >= 4])
            f.write(chars)
            f.write("\n")
            # autoencoder output sentence
            chars = "".join([corpus.dictionary.idx2word[x] for x in idx if x >= 4])
            f.write(chars)
            f.write("\n\n")


def main(args):
    random.seed(args.seed)
    ae_args, autoencoder, idx2word = load_ae(args.ae_args, args.ae_model, args.vocab_file)
    corpus = Corpus(args.data_path,
                    args.dict_file,
                    vocab_size=len(idx2word))
    # autoencoder.cuda()
    evaluate_autoencoder(autoencoder, args.outf, corpus, args.sample_size)


if __name__ == '__main__':
    # TODO make it GPU evaluable
    import argparse
    parser = argparse.ArgumentParser(description='AE evaluation.')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dict_file', type=str, required=True)
    parser.add_argument('--ae_args', type=str, required=True)
    parser.add_argument('--ae_model', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--outf', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1111)
    args = parser.parse_args()
    main(args)
