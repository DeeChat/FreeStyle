import re
import os
import json
from glob import glob
from tqdm import tqdm
from opencc import OpenCC
import jieba


regex_sw = re.compile(r'丶|☆|﹕|>|找歌词|作曲|作词|编曲|=|＝|监制|\@|\-|编辑人|《|www|QQ|：|歌词|by|演唱|－|―|;|\*|music|MUSIC|END|end|OH|］|<|［')
# regex_sw = re.compile(r'找歌词|作曲|作词|编曲|监制|\@|\-|编辑人|《|www|QQ|－|：|歌词|by|演唱|LRC')
regex_time = re.compile(r'\[.*\]')
# regex_punc = re.compile(r'[{}]+'.format('!,;.:?"\'！~～（）()，。…：#$%&*'))
regex_punc = re.compile(r'[^A-d\d \u4E00-\u9FA5]')
regex_skip = re.compile(r'[A-z\d\u3040-\u309F\u30A0-\u30FF]+')  # English letters, numbers, hiragana and katakana
regex_space = re.compile(r'\s+')

def main():
    '''
    将所有lrc文件预处理, 最后整合到一个完整的json文件
    '''

    opencc = OpenCC()
    jieba.set_dictionary('extra_dict/dict.txt.big')

    artists = {}
    with open('netease-data/artists-id-list.csv', 'r') as f:
        for line in f.readlines():
            artist_id, artist_name = list(map(lambda x: x.strip(), line.split(',', 1)))
            artists[artist_id] = artist_name

    songs = {}
    with open('netease-data/all-song-list.csv', 'r') as f:
        for line in f.readlines():
            song_id, song_name = list(map(lambda x: x.strip(), line.split(',', 1)))
            songs[song_id] = song_name

    files = glob('netease-data/netease_lyrics/*/*.lrc', recursive=True)
    print('{} lyrics detected.'.format(len(files)))
    files = list(filter(lambda x: not re.search(r'\+|\(|（|[A-z]', os.path.basename(x)[:-4]), files))
    print('filter down to {}.'.format(len(files)))
    data = []
    for file in tqdm(files):
        try:
            artist_id, song_id = file[:-4].split('-')[-1].split('/')
            artist_name, song_name = artists[artist_id], songs[song_id]
        except ValueError as e:
            print('pass filepath error', file, e)
            continue
        text = []
        with open(file) as f:
            skip = False
            for line in f:
                if line == '' or line[0] != '[' or regex_sw.search(line) or artist_name in line:
                    continue
                line = regex_time.sub('', line)
                line = regex_punc.sub('', line)
                if regex_skip.search(line):
                    skip = True
                    break
                line = line.strip()
                line = regex_space.sub(' ', line)
                line = opencc.convert(line)
                tokens = [token for token in jieba.cut(line) if token]
                if len(tokens) > 0:
                    text.append(tokens)
            if skip:
                continue
        if len(text) > 0:
            data.append({'artist_id': artist_id,
                         'artist_name': artist_name,
                         'song_id': song_id,
                         'song_name': song_name,
                         'text': text})
    with open('netease-data/netease-data.json', 'w') as f:
        json.dump({'data': data}, f, ensure_ascii=False)
    print('{} songs saved.'.format(len(data)))
    from pprint import pprint
    with open('netease-data/data.txt', 'w') as f:
        pprint(data, stream=f, compact=True)


if __name__ == '__main__':
    main()
