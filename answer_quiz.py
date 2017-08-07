import argparse
import numpy as np
import random
import json

import torch
from torch.autograd import Variable

from models import load_models, generate_from_hidden
from utils import Dictionary
from tqdm import tqdm

'''
Example Usage:
python answer_quiz.py --quiz_file quiz.json --ae_model output/ae/autoencoder_model_5.pt --g_model output/gan/gan_gen_model_99999.pt --d_model output/gan/gan_disc_model_99999.pt --outf answer.json
'''


def main(args):
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ae_args, gan_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.ae_args, args.gan_args, args.vocab_file,
                      args.ae_model, args.g_model, args.d_model)
    word2idx = {v: k for k, v in idx2word.items()}
    gan_gen.cuda()
    autoencoder.cuda()
    gan_gen.eval()
    autoencoder.eval()

    with open(args.quiz_file) as f:
        quizzes = json.load(f)

    def gen_batches(data, batch_size=args.batch_size):
        for i in range(int(np.ceil(len(data) / batch_size))):
            yield data[i: i + batch_size]

    def pad_data(data):
        maxlen = max(map(len, data))
        return [x + (maxlen - len(x)) * [Dictionary.pad] for x in data]

    answers = []
    i = 0
    for batch in tqdm(gen_batches(quizzes), total=int(np.ceil(len(quizzes) / args.batch_size))):
        source_idx = torch.LongTensor(pad_data([[Dictionary.sos] + [word2idx[w] for w in quiz] for quiz in batch])).cuda()
        source = autoencoder(Variable(source_idx, volatile=True), noise=False, encode_only=True)
        sentences = []
        for _ in range(args.gen_cnt):
            hidden = gan_gen(source)
            sentences.append(generate_from_hidden(autoencoder, hidden, vocab=idx2word,
                                                  sample=args.sample, maxlen=args.maxlen))
            source = hidden
        sentences = list(zip(*sentences))
        for quiz, ans in zip(batch, sentences):
            i += 1
            answers.append({'id': i, 'quiz': ''.join(quiz), 'answer': ans})
    with open(args.outf, 'w') as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate samples.')
    parser.add_argument('--ae_args', type=str, default='output/ae/args.json')
    parser.add_argument('--gan_args', type=str, default='output/gan/args.json')
    parser.add_argument('--vocab_file', type=str, default='output/ae/vocab.json')
    parser.add_argument('--ae_model', type=str, required=True)
    parser.add_argument('--g_model', type=str, required=True)
    parser.add_argument('--d_model', type=str, required=True)
    parser.add_argument('--quiz_file', type=str, required=True)
    parser.add_argument('--gen_cnt', type=int, default=11)
    parser.add_argument('--outf', type=str, default='./answer.json')
    parser.add_argument('--maxlen', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sample', action='store_true',
                        help='sample when decoding for generation')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()
    main(args)
