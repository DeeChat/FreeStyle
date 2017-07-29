import json


def main():
    with open('data.json') as f:
        data = json.load(f)['data']

    with open('vocab.txt') as f:
        vocab = []
        for line in f:
            vocab.append(line.split(',')[0])

    vocab = set(vocab[:12000])

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
    with open('chunks.json', 'w') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=0)
    # from pprint import pprint
    # with open('chunks.txt', 'w') as f:
    #     pprint(chunks, stream=f, compact=True)


if __name__ == '__main__':
    main()
