
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default='data.json')
parser.add_argument('--lda_train_file', type=str, default='lda_train.dat')
parser.add_argument('--stop_file', type=str, default='stopwords.dat')
args = parser.parse_args()

def main():
    with open(args.data_file) as f:
        data = json.load(f)['data']

    stop_words = {}
    with open(args.stop_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                stop_words[line.strip()] = 1

    articles = []
    for song in tqdm(data):
        text = []
        for s in song['text']:
            text += list(filter(lambda x: x not in stop_words, s))
        text = ' '.join(w for w in text).strip()
        if len(text) > 0:
            articles.append(text)

    with open(args.lda_train_file, 'w') as f:
        f.write('%d\n' % len(articles))
        for text in articles:
            f.write('%s\n' % text)


if __name__ == '__main__':
    '''
    usage:
    python train_prepare.py --data_file data/data.json --stop_file data/stopwords.dat --lda_train_file data/lda_train.dat
    python train_prepare.py --data_file ../netease-data/netease-data.json --stop_file stopwords.dat --lda_train_file ../netease-data/netease-lda-train.dat

    '''
    main()
    cnt = 0
    with open('../netease-data/netease-lda-train.dat', 'r') as f:
        for line in f.readlines():
            cnt += 1
            line = line.strip()
            if len(line) == 0:
                print('line ', cnt, 'is empty')


