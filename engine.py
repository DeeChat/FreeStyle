import os
import json
import whoosh.index as index
from whoosh.fields import Schema, NGRAM
from whoosh import qparser
from whoosh.qparser import QueryParser
from tqdm import tqdm


def demo_zh():
    text = ['每一个星星都是眼泪',
            '说不清是对是错',
            '何必要为难自己',
            '时光过欣赏你更多',
            '我快要 长大了',
            '可是有谁不会',
            '是我的冷漠',
            '抬头怎开口']
    os.makedirs('indexdir', exist_ok=True)
    schema = Schema(line=NGRAM(stored=True, queryor=True, minsize=2, maxsize=2))
    ix = index.create_in("indexdir", schema)
    writer = ix.writer()
    for line in text:
        print(line)
        writer.add_document(line=line)
    writer.commit()
    print('-------------')
    queries = ['冷漠',
               '要 长大',
               '要长大',
               ' ',
               '为难自己']
    with ix.searcher() as searcher:
        for q in queries:
            query = QueryParser("line", ix.schema).parse(q)
            print('query: {}'.format(q))
            print('result: ', end='')
            results = searcher.search(query)
            if len(results) > 0:
                print(results[0]['line'])
            else:
                print('NULL')
            print()
    print('------------')
    queries = ['为何要为难自己',
               '该怎开口',
               '怎开口',
               '开口如何说',
               '星星都是我的眼泪']
    with ix.searcher() as searcher:
        for q in queries:
            query = QueryParser("line", ix.schema, group=qparser.OrGroup).parse(q)
            print('query: {}'.format(q))
            results = searcher.search(query)
            print('result: ', end='')
            if len(results) > 0:
                print(results[0]['line'])
            else:
                print('NULL')
            print()


def main():
    with open('chunks.json') as f:
        data = json.load(f)
    data = [''.join(line) for chunk in data for line in chunk]
    print(len(data), '-> ', end='')
    data = list(set(data))
    print(len(data))

    os.makedirs('indexdir', exist_ok=True)
    schema = Schema(line=NGRAM(stored=True, queryor=True, minsize=2, maxsize=2))
    ix = index.create_in("indexdir", schema)
    writer = ix.writer()
    for line in tqdm(data):
        writer.add_document(line=line)
    writer.commit()


def test():
    text = ['可惜不是你 陪我到最后',
            '如果你说你不爱我',
            '我想要怒放的生命',
            '好吗 一句话就湿了眼眶',
            '偶尔会有下雨的天气']
    ix = index.open_dir('indexdir')
    # group = qparser.OrGroup.factory(0.8)
    with ix.searcher() as searcher:
        for line in text:
            # line = ' '.join([x + 'x' for x in list(line)])
            query = QueryParser("line", ix.schema, group=qparser.OrGroup).parse(line)
            results = searcher.search(query)
            print(line)
            print('----------')
            print(len(results))
            if len(results) > 0:
                results = [hit['line'] for hit in results[:5]]
                for result in results:
                    print(result)
            print()


if __name__ == '__main__':
    # demo_zh()
    main()
    # test()
