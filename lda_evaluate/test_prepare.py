import re
import os
import json
from glob import glob
from tqdm import tqdm
from opencc import OpenCC
import jieba
import argparse

import requests

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='lda_train.dat.theta')
parser.add_argument('--output_file', type=str, default='evaluate.out')
args = parser.parse_args()

def main():
    jieba.set_dictionary('../extra_dict/dict.txt.big')
    list(jieba.cut('分词器预热。'))

    stop_words = {}
    with open('stopwords.dat', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                stop_words[line.strip()] = 1

    files = glob('answer/*.json', recursive=True)
    articles = []
    for file in tqdm(files):
        with open(file) as f:
            data = json.load(f)

        for q in data:
            print('id = ', q['id'])
            text = []
            for line in q['answer']:
                tokens = [token for token in jieba.cut(line) if token and token not in stop_words]
                if len(tokens) > 0:
                    text += tokens
            text = ' '.join(w for w in text).strip()
            if len(text) > 0:
                articles.append(text)

    with open('test.dat', 'w') as f:
        f.write('%d\n' % len(articles))
        for text in articles:
            f.write('%s\n' % text)

if __name__ == '__main__':
    '''
    usage:
    '''
    main()