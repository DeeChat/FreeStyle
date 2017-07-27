#! usr/bin/python
#coding:utf-8
#Author:Xuejun Cheng
import re
import os
import os.path
class reprocessing(object):
    def __init__(self,rootdir):
        self.rootdir = rootdir
    # 去除题目中所有包括Live等关键字文件（可以添加）
    def keywords_delete(self):
        '''
        	Dedupilicate lyric files, reserve the original version
        	while discarding other versions (live,粤语,etc..)

        	path: path of folder to dedupulicate
        	'''
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                s = os.path.join(parent, filename)
                if re.search(r'live|\+|\(|\（', filename) is not None:
                    os.remove(s)

    # 去除所有时间戳
    def time_delete(self):
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                s = os.path.join(parent, filename)
                f = open(s, 'r', encoding='utf8')
                line = f.readlines()
                f1 = open(s, 'w+', encoding='utf8')
                al = re.compile(r'\[.*?\]')
                try:
                    for i in range(0, len(line)):
                        result = re.sub(al, "\n", line[i])
                        f1.write(result)
                except IndexError:
                    print("passttime_delete")
                f.close()
                f1.close()

    #去除歌词原始文本中带相关字符的行删除(可以添加)

    def char_line_delete(self):
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                s = os.path.join(parent, filename)
                f = open(s, 'r+', encoding='utf8')
                line = f.readlines()
                f1 = open(s, 'w+', encoding='utf8')
                try:
                    for i in range(0, len(line)):
                        if re.search(r'丶|∶|☆|\d|﹕|>|找歌词|作曲|作词|编曲|监制|\@|\-|编辑人|《|www|QQ|：|歌词|by|演唱|:|－|;|\*|music|MUSIC|END|end|OH|\］|<|\［', line[i]) is not None:
                            del line[i]
                        else:
                            f1.write(line[i])
                except IndexError:
                    print("passchar_line_delete")
                f.close()
                f1.close()
#去除歌词原始文本中带相关字符的位置清空（可以继续添加）
    def char_line_replace(self):
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                s = os.path.join(parent, filename)
                f = open(s, 'r+', encoding='utf8')
                line = f.readlines()
                f1 = open(s, 'w+', encoding='utf8')
                al = re.compile(r'〈|ˉ|\｜|­|•|﻿|\．|\/|·|、|�|\,|_|#|，|,|；|！|\？|\?|\(.*?\)|\.|。|…|\<.*?\>|\（.*?\）|~|!|&|\'|’|“|”|"|～|\）|\)')
                try:
                    for i in range(0, len(line)):
                        result = re.sub(al, "", line[i])
                        f1.write(result)
                except IndexError:
                    print("pass_char_line_replace")
                f.close()
                f1.close()

  #去除歌词中所有的空格和所有空行

    def null_delete(self):
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                s = os.path.join(parent, filename)
                f = open(s, 'r+', encoding='utf8')
                line = f.readlines()
                #删除空格
                f1 = open(s, 'w+', encoding='utf8')
                al = re.compile(r'\s')
                try:
                    for i in range(0, len(line)):
                        result = re.sub(al, "", line[i])
                        f1.write(result + '\n')
                except IndexError:
                    print("pass_null_delete")
                f.close()
                f1.close()
                #删除空行
                f = open(s, 'r+', encoding='utf8')
                line = f.readlines()
                f2 = open(s, 'w+', encoding='utf8')
                try:
                    for i in range(0, len(line)):
                        data = line[i].strip()
                        if len(data) != 0:
                            f2.write(data)
                            f2.write('\n')
                except IndexError:
                    print("pass_null_detete")
                f.close()
                f2.close()

   #统计当前的歌词文件数目
    def countofdocument(self):
        number = 0
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                number = number + 1
        return number

    #去除所有英文歌曲
    def englishlyric_delete(self):
        def is_chinese(uchar):
            '''
            :param uchar:
            :return:
            '''
            varible=0
            for element in uchar.strip('\n'):
                if element < u'\u4e00' or element > u'\u9fa5':
                    varible=1
                    break
            return varible
        result = []
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                s = os.path.join(parent, filename)
                f = open(s, 'r+', encoding='utf8')
                line = f.readlines()
                a = 0
                for i in range(0, len(line)):
                    a =a+ is_chinese(line[i])
                if a > 0:
                    result.append(s)
                f.close()
        for t in result:
            os.remove(t)

    #去除空文件和空文件夹
    def null_docfile_delete(self):
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for dirname in dirnames:
                if not os.listdir(os.path.join(parent, dirname)):
                    os.rmdir(os.path.join(parent, dirname))
            for filename in filenames:
                if os.path.getsize(os.path.join(parent, filename)) == 0:
                    os.remove(os.path.join(parent, filename))

    #删除第一行空行和第二行如果字符长度小于5
    def firstline_delete(self):
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                # 去除第一行空行
                s = os.path.join(parent, filename)
                f = open(s, 'r+', encoding='utf8')
                line = f.readlines()
                f1 = open(s, 'w+', encoding='utf8')
                try:
                    for i in range(0, len(line)):
                        if i == 0:
                            del line[i]
                        else:
                            if i == 1:
                                if len(line[i]) < 5:
                                    del line[i]
                                else:
                                    f1.write(line[i])
                            else:
                                f1.write(line[i])
                except IndexError:
                    print("passfirst_line")
                f.close()
                f1.close()
    #遇见中英混杂歌词将英文歌词位置清空
    def eng_empty(self):
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                s = os.path.join(parent, filename)
                f = open(s, 'r+', encoding='utf8')
                line = f.readlines()
                f1 = open(s, 'w+', encoding='utf8')
                al = re.compile('[A-Za-z]')
                try:
                    for i in range(0, len(line)):
                        result = re.sub(al, '', line[i])
                        f1.write(result)
                except IndexError:
                    print("pass_eng_empty")
                f.close()
                f1.close()
    #遇见中英混杂歌曲将该歌曲删除
    def eng_delete(self):
        resultnew = []
        for parent, dirnames, filenames in os.walk(self.rootdir):
            for filename in filenames:
                s = os.path.join(parent, filename)
                f = open(s, 'r+', encoding='utf8')
                line = f.readlines()
                a = 1
                try:
                    for i in range(0, len(line)):
                        if re.search(r'[A-Za-z]', line[i]) is not None:
                            a = 0
                            break
                    if a == 0:
                        resultnew.append(s)
                except IndexError:
                    print("pass_eng_delete")
                f.close()
        for t in resultnew:
            os.remove(t)
    #中文分词


if __name__=="__main__":
        dir = "./lyric/new"  # 指明被遍历的文件夹
        # reprocessing(dir).keywords_delete()
        # reprocessing(dir).time_delete()
        # reprocessing(dir).char_line_delete()
        # reprocessing(dir).char_line_replace()
        reprocessing(dir).englishlyric_delete()
        # # reprocessing(dir).firstline_delete()
        # # reprocessing(dir).eng_empty()
        # # reprocessing(dir).null_delete()
        # reprocessing(dir).null_docfile_delete()
        # print(reprocessing(dir).countofdocument())

