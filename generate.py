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


# def interpolate(ae, gg, z1, z2, vocab,
#                 steps=5, sample=None, maxlen=None):
#     """
#     Interpolating in z space
#     Assumes that type(z1) == type(z2)
#     """
#     if type(z1) == Variable:
#         noise1 = z1
#         noise2 = z2
#     elif type(z1) == torch.FloatTensor or type(z1) == torch.cuda.FloatTensor:
#         noise1 = Variable(z1, volatile=True)
#         noise2 = Variable(z2, volatile=True)
#     elif type(z1) == np.ndarray:
#         noise1 = Variable(torch.from_numpy(z1).float(), volatile=True)
#         noise2 = Variable(torch.from_numpy(z2).float(), volatile=True)
#     else:
#         raise ValueError("Unsupported input type (noise): {}".format(type(z1)))

#     # interpolation weights
#     lambdas = [x*1.0/(steps-1) for x in range(steps)]

#     gens = []
#     for L in lambdas:
#         gens.append(generate(ae, gg, (1-L)*noise1 + L*noise2,
#                              vocab, sample, maxlen))

#     interpolations = []
#     for i in range(len(gens[0])):
#         interpolations.append([s[i] for s in gens])
#     return interpolations


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

    model_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.model_path)

    ###########################################################################
    # Generation code
    ###########################################################################

    # Generate sentences
    corpus = Corpus(args.data_path,
                    args.dict_file,
                    vocab_size=len(idx2word),
                    subset=args.ngenerations * 3)

    source, _ = next(BatchGen(corpus.pairs, args.ngenerations))
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
        for sent in sentences:
            f.write(sent + "\n")

    # Generate interpolations
    # if args.ninterpolations > 0:
    #     noise1 = torch.ones(args.ninterpolations, model_args['z_size'])
    #     noise1.normal_()
    #     noise2 = torch.ones(args.ninterpolations, model_args['z_size'])
    #     noise2.normal_()
    #     interps = interpolate(autoencoder, gan_gen,
    #                           z1=noise1,
    #                           z2=noise2,
    #                           vocab=idx2word,
    #                           steps=args.steps,
    #                           sample=args.sample,
    #                           maxlen=model_args['maxlen'])

    #     if not args.noprint:
    #         print("\nSentence interpolations:\n")
    #         for interp in interps:
    #             for sent in interp:
    #                 print(sent)
    #             print("")
    #     with open(args.outf, "a") as f:
    #         f.write("\nSentence interpolations:\n\n")
    #         for interp in interps:
    #             for sent in interp:
    #                 f.write(sent+"\n")
    #             f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
    parser.add_argument('--model_path', type=str, required=True,
                        help='directory to load models from')
    # parser.add_argument('--temp', type=float, default=1,
    #                     help='softmax temperature (lower --> more discrete)')
    parser.add_argument('--ngenerations', type=int, default=10,
                        help='Number of sentences to generate')
    # parser.add_argument('--ninterpolations', type=int, default=5,
    #                     help='Number z-space sentence interpolation examples')
    # parser.add_argument('--steps', type=int, default=5,
    #                     help='Number of steps in each interpolation')
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
    print(vars(args))
    main(args)
