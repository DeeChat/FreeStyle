import re
import os
import json
from glob import glob
from tqdm import tqdm
from opencc import OpenCC
import jieba

opencc = OpenCC()
jieba.set_dictionary('resources/dict.txt.big')
list(jieba.cut('分词器预热。'))
print()


# TODO add max len limit
def main():
    files = glob('lyrics/**/*.lrc', recursive=True)
    print('{} lyrics detected.'.format(len(files)))
    files = list(filter(lambda x: not re.search(r'[^\u4E00-\u9FA5\-]', os.path.basename(x)[:-4]), files))
    print('filter down to {}.'.format(len(files)))
    data = []
    for file in tqdm(files):
        try:
            singer, song = os.path.basename(file)[:-4].split('-')
        except ValueError as e:
            continue
        text = []
        with open(file) as f:
            skip = False
            for line in f:
                if line == '' or line[0] != '[' or regex_sw.search(line) or singer in line:
                    continue
                line = regex_time.sub('', line)
                line = line.strip()
                line = regex_space.sub(' ', line)
                line = regex_punc.sub(' ', line)
                line = line.strip()
                if regex_skip.search(line):
                    skip = True
                    break
                line_sc = opencc.convert(line)
                if line != line_sc:
                    skip = True
                    break
                tokens = [token for token in jieba.cut(line) if token]
                if len(tokens) > 0:
                    text.append(tokens)
            if skip:
                continue
        if len(text) > 0:
            data.append({'singer': singer, 'song': song, 'text': text})
    with open('data.json', 'w') as f:
        json.dump({'data': data}, f, ensure_ascii=False)
    print('{} songs saved.'.format(len(data)))
    from pprint import pprint
    with open('data.txt', 'w') as f:
        pprint(data, stream=f, compact=True)
regex_sw = re.compile(r'丶|☆|﹕|>|找歌词|作曲|作词|编曲|原唱|=|＝|监制|\@|\-|编辑人|《|www|QQ|：|歌词|by|演唱|－|―|;|\*|music|MUSIC|END|end|OH|］|<|［')
# regex_sw = re.compile(r'找歌词|作曲|作词|编曲|监制|\@|\-|编辑人|《|www|QQ|－|：|歌词|by|演唱|LRC')
regex_time = re.compile(r'\[.*\]([\(（].+[）\)])?')
# regex_punc = re.compile(r'[{}]+'.format('!,;.:?"\'！~～（）()，。…：#$%&*'))
regex_punc = re.compile(r'[^A-z\d \u4E00-\u9FA5]')
# regex_skip = re.compile(r'[A-z\d\u3040-\u309F\u30A0-\u30FF毋拢阮屎馀廿阙国啦]+')
regex_skip = re.compile(r'[A-z\d\u3040-\u309F\u30A0-\u30FF]+')  # English letters, numbers, hiragana and katakana
regex_space = re.compile(r'\s+')


if __name__ == '__main__':
    main()
