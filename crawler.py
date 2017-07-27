from bs4 import BeautifulSoup
import os
from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.parse import quote

def get_lrc_url(domain, loc):
    """获取歌手单页的全部歌词文件URL"""

    doc = urlopen(domain + loc).read()
    soup = BeautifulSoup(doc, 'html5lib')

    link_list = []
    for a in soup.find_all('a', class_='ico-lrc'):
        # print(a['href'])
        link_list.append(a['href'])
    return link_list


def get_singer_lrc(domain, loc):
    """获取歌手的全部歌词文件URL"""

    link_list = []
    while True:
        link_list += get_lrc_url(domain, loc)
        doc = urlopen(domain + loc).read()
        soup = BeautifulSoup(doc, 'html5lib')
        div = soup.find('div', class_='pages')
        a_list = div.find_all('a')
        if not a_list:
            break
        if a_list[-1].string == '下一页 »':
            loc = a_list[-1]['href']
        else:
            break
    return link_list


def download_file(domain, singer, lrc_list):
    """下载URL列表里的全部歌词文件"""

    path = "D:\\lyrics\\" + singer + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
    for url in lrc_list:
        lrc_url = domain + quote(url)  # 解决带中文的URL
        filename = path + url.split('/')[1]
        try:
            urlretrieve(lrc_url, filename=filename)
        except OSError:
            continue


if __name__ == '__main__':
    domain = 'http://www.lrcgc.com/'
    artists = ['artist-11.html', 'artist-12.html', 'artist-13.html', 'artist-21.html',
               'artist-22.html', 'artist-23.html']
    for artist in artists:
        doc = urlopen(domain + artist).read()
        soup = BeautifulSoup(doc, 'html5lib')
        for ul in soup.find_all('ul', class_='cc'):
            for a in ul.find_all('a'):
                loc = a['href']
                singer = a.string
                lrc_list = get_singer_lrc(domain, loc)
                download_file(domain, singer, lrc_list)
                print("%s done" % singer)
