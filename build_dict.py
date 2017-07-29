import json
from collections import Counter


def main():
    with open('data.json') as f:
        data = json.load(f)['data']

    def gen_tf(data):
        for song in data:
            for line in song['text']:
                for w in line:
                    yield w

    def gen_df(data):
        for song in data:
            for w in set(w for line in song['text'] for w in line):
                yield w

    tf_counter = Counter(gen_tf(data))
    df_counter = Counter(gen_df(data))
    vocab = sorted(df_counter,
                   key=lambda x: (df_counter.get(x), tf_counter.get(x)),
                   reverse=True)
    with open('vocab.txt', 'w') as f:
        for w in vocab:
            f.write('{},{},{}\n'.format(w, df_counter.get(w), tf_counter.get(w)))

if __name__ == '__main__':
    main()
