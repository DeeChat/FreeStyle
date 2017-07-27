#! usr/bin/python
# coding: utf-8
# Author: Pan Li (pl1748@nyu.edu)

import os
import codecs
from collections import Counter
import pandas as pd

def make_dictionary(path):
    '''
    Transform lyric line into word list for 
    further calculation
    
    path: path of the lyric folder
    '''
    wordlist = []
    file_wordlist = []
    files = os.listdir(path)
    for file in files:
        with codecs.open(path + '/' + file, 'r', 'utf-8') as fopen:
            raw_lyric = fopen.readlines()
            raw_lyric = [x.split(',') for x in raw_lyric]
            [wordlist.append(x) for xx in raw_lyric for x in xx]
            file_wordlist=list(set(wordlist))
    return [wordlist,file_wordlist]

def concat_dictionary(path):
    '''
	Concat word lists to construct the dictionary

	path: path of the word lists
    '''
    folders = os.listdir(path)
    words = []
    file_words = []
    for folder in folders:
        words.extend(make_dictionary(path + '/' + folder)[0])
        file_words.extend(make_dictionary(path + '/' + folder)[1])
    for index in range(len(words)):
        words[index]=words[index].replace('\n','')
    for index in range(len(file_words)):
    	file_words[index] = file_words[index].replace('\n','')
    wordlist = dict(Counter(words))
    file_wordlist = Counter(file_words)
    word_order = sorted(file_wordlist.items(), key=lambda d: d[1], reverse=True)
    with codecs.open('df_dictionary','w','utf-8') as f:
    	for index in range(len(word_order)):
        	f.write(str(word_order[index][0]) + ',' +
        		str(word_order[index][1])+','+str(wordlist[word_order[index][0]])
        		+','+str(index)+'\n')
    return True

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
                        lyric = lyric.replace('\n','')
                        lyric = lyric.split(',')
                        lyric_num = [dictionary[word] for word in lyric]
                        lyric_num = [str(word) for word in lyric_num]
                        lyric_num = ','.join(lyric_num)
                        fwrite.write(lyric_num +'\n')
    return True

if __name__ == '__main__':
    file_path = 'processed_lyric'
    output_path = 'numerical_lyric'
    dictionary = {}
    concat_dictionary(file_path)
    df=pd.read_csv('df_dictionary',names=['word','df','tf','index'])
    df_sort=df.sort_values(by=['df','tf'],ascending=[0,0])
    df_sort.to_csv('df_dictionary',header=False,index=False) 
    for index in df_sort.index:
        dictionary[df_sort.loc[index,'word']]=df_sort.loc[index,'index']
    os.mkdir(output_path)
    folders = os.listdir(file_path)
    for folder in folders:
        make_index(file_path +'/'+ folder, dictionary, output_path +'/'+ folder)
