import json


def main(args):
    with open(args.data_file) as f:
        data = json.load(f)['data']

    with open(args.vocab_file) as f:
        vocab = []
        for line in f:
            vocab.append(line.split(',')[0])

    vocab = set(vocab[:args.vocab_size])

    def gen(data):
        for song in data:
            chunk = []
            for line in song['text']:
                if set(line) <= vocab:
                    chunk.append(line)
                elif len(chunk) > 0:
                    yield chunk
                    chunk = []
            if len(chunk) > 0:
                yield chunk
    chunks = list(gen(data))
    print('extract {} chunks with vocab size {}.'.format(len(chunks), len(vocab)))
    with open(args.outf, 'w') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=0)
    if args.sample_file:
        with open(args.sample_file, 'w') as f:
            json.dump(chunks[:args.sample_size], f, ensure_ascii=False, indent=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=12000)
    parser.add_argument('--data_file', type=str, default='data.json')
    parser.add_argument('--vocab_file', type=str, default='vocab.txt')
    parser.add_argument('--outf', type=str, default='chunks.json')
    parser.add_argument('--sample_file', type=str, default='sample.json')
    parser.add_argument('--sample_size', type=int, default=100)
    args = parser.parse_args()
    main(args)
