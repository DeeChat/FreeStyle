import json
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import whoosh.index as index
from whoosh import qparser
from whoosh.qparser import QueryParser
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

chencherry = SmoothingFunction()
ix = index.open_dir('indexdir')


def search(line, n):
    # group = qparser.OrGroup.factory(0.8)
    with ix.searcher() as searcher:
        query = QueryParser("line", ix.schema, group=qparser.OrGroup).parse(line)
        results = searcher.search(query, limit=n)
        if not results.is_empty():
            return [list(hit['line']) for hit in results[:n]]
        else:
            return [[]]


def score(line):
    reference = search(''.join(line), n=200)
    return sentence_bleu(reference, line,
                         smoothing_function=chencherry.method7,
                         weights=(0, 0.2, 0.4, 0.4))


def main(args):
    import os
    os.makedirs('score', exist_ok=True)
    with open(args.inputf) as f:
        answers = json.load(f)
    answers = [list(line) for answer in answers for line in answer['answer']]

    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        futures = [pool.submit(score, x) for x in answers]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    for future in tqdm(futures):
        out.append(future.result())
    with open(args.outf, 'w') as f:
        for a, s in zip(answers, out):
            f.write('{} {}\n'.format(''.join(a), s))
    print(np.mean(out))


def demo():
    hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
                   'ensures', 'that', 'the', 'military', 'always',
                   'obeys', 'the', 'commands', 'of', 'the', 'party']
    hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',
                   'forever', 'hearing', 'the', 'activity', 'guidebook',
                   'that', 'party', 'direct']
    reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
                  'ensures', 'that', 'the', 'military', 'will', 'forever',
                  'heed', 'Party', 'commands']
    reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
                  'guarantees', 'the', 'military', 'forces', 'always',
                  'being', 'under', 'the', 'command', 'of', 'the',
                  'Party']
    reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
                  'army', 'always', 'to', 'heed', 'the', 'directions',
                  'of', 'the', 'party']
    print((sentence_bleu([reference1, reference2, reference3], hypothesis1)))
    print((sentence_bleu([reference1, reference2, reference3], hypothesis2)))
    print((sentence_bleu([reference1, reference2, reference3], hypothesis1, smoothing_function=chencherry.method7)))
    print((sentence_bleu([reference1, reference2, reference3], hypothesis2, smoothing_function=chencherry.method7)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputf', required=True)
    parser.add_argument('--outf', required=True)
    args = parser.parse_args()
    main(args)
