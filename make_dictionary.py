#! usr/bin/python
# coding: utf-8
# Author: Pan Li (pl1748@nyu.edu)

import re
import os
import codecs
from collections import Counter
import pandas as pd
import numpy as np

def make_dictionary(path):
	'''
	Transform lyric line into word list for 
	further calculation

	path: path of the lyric folder
	'''
    wordlist = []
    files = os.listdir(path)
    num_files = len(files)
    for file in files:
        with codecs.open(path + '/' + file, 'r', 'utf-8') as fopen:
            raw_lyric = fopen.readlines()
            raw_lyric = [x.split(',') for x in raw_lyric]
            append_lyric = [wordlist.append(x) for y in raw_lyric for x in y]
    return wordlist

def concat_dictionary(path):
	'''
	Concat word lists to construct the dictionary

	path: path of the word lists
	'''
	folders = os.listdir(path)
    words = []
    dictionary = {}
    for folder in folders:
        words.append(make_dictionary(path + '/' + folder))
    wordlist = [x for y in words for x in y]
    wordlist = Counter(wordlist)
    word_order = dict(sorted(wordlist.items(), key=lambda d: d[1], reverse=True))
    word_list = list(word_order.keys())
    for index in range(len(word_list)):
        dictionary[word_list[index]] = index
    return dictionary

def make_index(input_path,dictionary,output_path):
	'''
	Assign index for each word in the dictionary
	with the order of frequency

	input_path: path of the lyric folder
	dictionary: consturcted dictionary including 
	words and their frequency
	output_path: path of the output numerical lyrics 
	'''
    files = os.listdir(input_path)
    os.mkdir(output_path)
    for file in files:
        with codecs.open(input_path +'/'+ file,'r','utf-8') as fread:
            lyrics = fread.readlines()
            if len(lyrics)>=2:
                with codecs.open(output_path +'/'+ file,'w','utf-8') as fwrite:
                    for index in range(len(lyrics)):
                        lyric = lyrics[index]
                        lyric = lyric.split(',')
                        lyric_num = [dictionary[word] for word in lyric]
                        lyric_num = [str(word) for word in lyric_num]
                        lyric_num = ','.join(lyric_num)
                        fwrite.write(lyric_num +'\n')
    return True

if __name__ == '__main__':
    file_path = '.../processed_lyric'
    output_path = '.../numerical_lyric'
    dictionary = concat_dictionary(file_path)
    os.mkdir(output_path)
    folders = os.listdir(file_path)
    for folder in folders:
        make_index(file_path +'/'+ folder, dictionary, output_path +'/'+ folder)
    #with codecs.open('C:/Users/Admin/Desktop/result.txt','w','utf-8') as f:
    #    f.write(str(word_order))
