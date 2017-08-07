import re
import json
import numpy as np
from xpinyin import Pinyin

'''
real data: 0.111239683366
answer/0.json 0.105717329545
answer/1.json 0.118030894886
answer/2.json 0.0961558948864
answer/3.json 0.106711647727
answer/4.json 0.131356534091
answer/5.json 0.123481889205
answer/6.json 0.133212002841
answer/7.json 0.126109730114
answer/8.json 0.133132102273
answer/9.json 0.127672230114
answer/10.json 0.120854048295
answer/11.json 0.118084161932
answer/12.json 0.120498934659
answer/13.json 0.127982954545
answer/14.json 0.118013139205
answer/15.json 0.127379261364
answer/16.json 0.117595880682
answer/17.json 0.124795809659
answer/18.json 0.117009943182
answer/19.json 0.124564985795
'''

p = Pinyin()
vowel = re.compile(r'[aeiouv]')


def get_rhyme(char):
    py = p.get_pinyin(char)
    try:
        idx = vowel.search(py).start()
    except AttributeError as e:
        return py
    return py[idx:]


def rhyme_score(lines):
    rhymes = []
    chars = []
    score = 0
    for line in lines:
        char = line.strip()[-1:]
        rhyme = get_rhyme(char)
        if len(rhymes) == 4:
            rhymes.pop(0)
            chars.pop(0)
        if rhyme in rhymes and char not in chars:
            score += 1
        rhymes.append(rhyme)
        chars.append(char)
    return score / (len(lines) - 1)


def test():
    text = ['这一刻 突然觉得好熟悉',
            '像昨天 今天同时在放映',
            '我这句语气 原来好像你',
            '不就是我们爱过的证据']
    print(rhyme_score(text))


def main():
    with open('chunks.json') as f:
        data = json.load(f)
    data = [''.join(line) for chunk in data if len(chunk) > 4 for line in chunk]
    score = np.mean([rhyme_score(chunk) for chunk in data])
    print('real data:', score)

    for i in range(20):
        file = 'answer/{}.json'.format(i)
        with open(file) as f:
            data = json.load(f)
        answers = [[answer['quiz']] + answer['answer'] for answer in data]
        score = np.mean([rhyme_score(answer) for answer in answers])
        print(file, score)


if __name__ == '__main__':
    main()
    # test()
