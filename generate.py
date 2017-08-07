import argparse
import numpy as np
import random

import torch
from torch.autograd import Variable

from models import load_models, generate, decode_idx
from utils import Corpus, BatchGen

###############################################################################
# Generation methods
###############################################################################


def main(args):
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")

    ###########################################################################
    # Load the models
    ###########################################################################

    ae_args, gan_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.ae_args, args.gan_args, args.vocab_file,
                      args.ae_model, args.g_model, args.d_model)

    ###########################################################################
    # Generation code
    ###########################################################################

    # Generate sentences
    corpus = Corpus(args.data_path,
                    args.dict_file,
                    vocab_size=len(idx2word))

    source, _ = next(BatchGen(corpus.get_chunks(size=2), args.ngenerations))
    prev_sent = [decode_idx(corpus.dictionary, sent) for sent in source.tolist()]
    source = Variable(source, volatile=True)
    sentences = generate(autoencoder, gan_gen, inp=source,
                         vocab=idx2word, sample=args.sample,
                         maxlen=args.maxlen)

    if not args.noprint:
        print("\nSentence generations:\n")
        for prev, sent in zip(prev_sent, sentences):
            print(prev)
            print("    ", sent)
            print("")
    with open(args.outf, "w") as f:
        f.write("Sentence generations:\n\n")
        for prev, sent in zip(prev_sent, sentences):
            f.write(prev + '\n')
            f.write("-> " + sent + '\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
    # parser.add_argument('--model_path', type=str, required=True,
    #                     help='directory to load models from')
    parser.add_argument('--ae_args', type=str, required=True)
    parser.add_argument('--gan_args', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--ae_model', type=str, required=True)
    parser.add_argument('--g_model', type=str, required=True)
    parser.add_argument('--d_model', type=str, required=True)
    parser.add_argument('--ngenerations', type=int, default=10,
                        help='Number of sentences to generate')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dict_file', type=str, required=True)
    parser.add_argument('--outf', type=str, default='./generated.txt',
                        help='filename and path to write to')
    parser.add_argument('--noprint', action='store_true',
                        help='prevents examples from printing')
    parser.add_argument('--maxlen', type=int, default=15)
    parser.add_argument('--sample', action='store_true',
                        help='sample when decoding for generation')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()
    main(args)
