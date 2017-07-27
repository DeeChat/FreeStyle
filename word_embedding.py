from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os
import argparse


def get_sentences(dir):
    sentences = []
    for filename in os.listdir(dir):
        if os.path.isdir(dir + '\\' + filename):
            for fn in os.listdir(dir + '\\' + filename):
                with open(dir + '\\' + filename + '\\' + fn, encoding='utf8') as f:
                    for line in f:
                        if line.find(',') != -1:
                            line = line.strip()
                            sentence = line.split(',')
                            sentences.append(sentence)
    return sentences


def get_embeddings(sentences, dim=100):
    model = Word2Vec(sentences, size=dim)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=100, help='specify the dimension for embeddings')
    args = parser.parse_args()
    path = 'D:\\processed_lyric\\processed_lyric'
    sentences = get_sentences(path)
    embeddings = get_embeddings(sentences, args.dim)
    embeddings.wv.save_word2vec_format('embeddings.txt')
    # embeddings = KeyedVectors.load_word2vec_format('embeddings.txt')
    # print(embeddings['悲伤'])
