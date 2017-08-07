from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os
import json
import argparse


def get_sentences(data_path, wv_dict_path):
    with open(data_path) as f:
        data = json.load(f)['data']

    sentences = []

    # convert word like blank ' ' to id and avoid be striped
    word2idx = {}

    idx = 0
    for song in data:
        for s in song['text']:
            idlist = []
            for w in s:
                if w not in word2idx:
                    word2idx[w] = str(idx)
                    idx += 1
                idlist.append(word2idx[w])
            sentences.append(idlist)
    with open(wv_dict_path, 'w') as f:
        json.dump(word2idx, f, ensure_ascii = False)
    return sentences

def get_embeddings(sentences, dim=300):
    model = Word2Vec(sentences, size=dim)
    return model

if __name__ == '__main__':
    '''
    usage:
    python word_embedding.py --data_file data/data.json --out_file data/new_word_embeddings.txt --wv_dict_file data/wv_dict.json
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=300, help='specify the dimension for embeddings')
    parser.add_argument('--data_file', type=str, default='data.json')
    parser.add_argument('--wv_dict_file', type=str, default='wv_dict.json')
    parser.add_argument('--out_file', type=str, default='embeddings.txt')
    args = parser.parse_args()
    sentences = get_sentences(args.data_file, args.wv_dict_file)
    embeddings = get_embeddings(sentences, args.dim)
    embeddings.wv.save_word2vec_format(args.out_file)

    # test
    with open(args.wv_dict_file, 'r') as f:
        word2idx = json.load(f)
    embeddings = KeyedVectors.load_word2vec_format(args.out_file)
    print(embeddings[word2idx['悲伤']])

